import caffe2_path
from caffe2.python import (
    core, workspace, model_helper, utils, brew, net_drawer
)
from mask_gru_cell import MaskGRU
from caffe2.python.optimizer import build_adam
from data_reader import build_input_reader
import numpy as np
import logging

logging.basicConfig()
log = logging.getLogger("mask_rnn")
log.setLevel(logging.DEBUG)

# Default set() here is intentional as it would accumulate values like a global
# variable
def CreateNetOnce(net, created_names=set()): # noqa
    name = net.Name()
    if name not in created_names:
        created_names.add(name)
        workspace.CreateNet(net)

class MaskRNN(object):
    def __init__(
        self,
        model_name, 
        db_name,
        seq_size,
        batch_size, 
        input_dim,
        output_dim,
        hidden_size,
        ):
    	'''
        The db contains: (T: seq length, N: batch size, D: input dim, E: output dim)
    	   seq_lengths: np.array of (N, 1) with each element is the seq length.
           input_blob: the concat (axis = 2) of:
            - inputs: np.float32 T * N * D
            - inputs_last: np.float32 T * N * D
            - inputs_mean: np.float32 T * N * D
            - masks: np.float32 T * N * D (same size as the inputs)
            - interval: np.float32 T * N * D (same size as the inputs)
           target: np.float32 T * N * E
    	'''
        workspace.ResetWorkspace()
        self.model_name = model_name
        self.db_name = db_name
        self.seq_size = seq_size
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.net_store = {}

    def build_net(
        self,
        base_learning_rate=0.1  # base_learning_rate * seq_size
        ):
        log.debug('Start Building Mask-RNN')
        model = model_helper.ModelHelper(name="mask_rnn")

        hidden_init= model.net.AddExternalInputs(
            'hidden_init',
        )

        # Add external inputs (read directly from the database)
        seq_lengths, _input_blob, _target = build_input_reader(
            model, self.db_name, 'minidb', 
            ['seq_lengths', 
             'input_blob_batch_first', 
             'target_batch_first'], 
            batch_size = self.batch_size, data_type='train'
        )
        # In order to put into batches, the input_blob is
        # [BATCH_SIZE, SEQ_LEN, INPUT_DIM] 
        # i.e. the first dim is the batch size 
        # However the required input dim is:
        # [SEQ_LEN, BATCH_SIZE, INPUT_DIM]
        input_blob = model.net.Transpose(
            [_input_blob], 'input_blob', axes=[1, 0, 2])
        target = model.net.Transpose(
            [_target], 'target', axes=[1, 0, 2])

        hidden_output_all, self.hidden_output = MaskGRU(
            model, input_blob, seq_lengths, [hidden_init],
            self.input_dim, self.hidden_size, scope="MaskRNN"
        )

        # axis is 2 as first two are T (time) and N (batch size).
        output = brew.fc(
            model,
            hidden_output_all,
            None,
            dim_in=self.hidden_size,
            dim_out=self.output_dim,
            axis=2
        )

        # Get the predict net
        (self.net_store['predict'], 
            self.external_inputs) = model_helper.ExtractPredictorNet(
            model.net.Proto(),
            [input_blob, seq_lengths, hidden_init],
            [output],
        )

        # TODO: print out output and output_reshaped, check dimension

        # Then, we add loss and gradient ops
        # We treat them as one big batch of size T * N
        output_reshaped, _ = model.net.Reshape(
            output, ['output_reshaped', '_output_shape'], 
            shape=[-1, self.output_dim])
        target, _ = model.net.Reshape(
            target, ['target_reshaped', '_target_shape'], 
            shape=[-1, self.output_dim])

        l2_dist = model.net.SquaredL2Distance(
            [output_reshaped, target], 'l2_dist')
        loss = model.net.AveragedLoss(l2_dist, 'loss')

        # Training net
        model.AddGradientOperators([loss])
        build_adam(
            model,
            base_learning_rate=base_learning_rate*self.seq_size,
        )

        self.model = model
        self.predictions = output
        self.loss = loss

        # Create a net to copy hidden_output to hidden_init
        prepare_state = core.Net("prepare_state")
        prepare_state.Copy(self.hidden_output, hidden_init)
        self.net_store['prepare'] = prepare_state

    def train(
        self, 
        iters,
        iters_to_report=0, 
        ):
        log.debug("Training model")

        workspace.RunNetOnce(self.model.param_init_net)
        # initialize the output states which will be copied to input
        # states within the loop below
        workspace.FeedBlob(self.hidden_output, np.zeros(
            [1, self.batch_size, self.hidden_size], dtype=np.float32
        ))
        # Create the prepare net and train net
        workspace.CreateNet(self.net_store['prepare'])

        for i in range(iters):
            print('>>> iter: ' + str(i))
            # Reset output state
            workspace.FeedBlob(self.hidden_output, np.zeros(
                [1, self.batch_size, self.hidden_size], dtype=np.float32
            ))
            # Copy hidden_ouput to hidden_init
            workspace.RunNet(self.net_store['prepare'].Name())
            CreateNetOnce(self.model.net)
            workspace.RunNet(self.model.net.Name())

    def draw_nets(self):
        for net_name in self.net_store:
            net = self.net_store[net_name]
            graph = net_drawer.GetPydotGraph(net.Proto().op, rankdir='TB')
            with open(self.model_name + '_' + net.Name() + ".png",'wb') as f:
                f.write(graph.create_png())
            with open(self.model_name + '_' + net.Name() + "_proto.txt",'wb') as f:
                f.write(str(net.Proto()))

def main():
    SEQ_LEN = 5
    NUM_EXAMPLE = 10
    INPUT_DIM = 2
    OUTPUT_DIM = 1
    my_model = MaskRNN(
        'MaskRNN_test',
        'test.minidb',
        seq_size=SEQ_LEN,
        batch_size=10,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        hidden_size=6,
    )
    my_model.build_net(base_learning_rate=0.1)
    my_model.draw_nets()
    my_model.train(
        iters=1
    )

if __name__ == '__main__':
    main()