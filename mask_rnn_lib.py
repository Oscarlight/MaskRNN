import caffe2_path
from caffe2.python import (
    core, workspace, model_helper, utils, brew
)
from caffe2.python.gru_cell import GRU
from caffe2.python.optimizer import build_adam
import logging
logging.basicConfig()
log = logging.getLogger("mask_rnn")
log.setLevel(logging.DEBUG)

class MaskRNN(object):
    def __init__(self, 
        db_name,
        batch_size, 
        input_dim,
        output_dim,
        hidden_size,
        iters,
        iters_to_report,
        ):
    	'''
        The db contains: (T: seq length, N: batch size, D: input dim, E: output dim)
    	   seq_lengths: np.array of (N, 1) with each element is the seq length.
           input_blob: the concat (axis = 2) of:
            - inputs: np.float32 T * N * D
            - inputs_mean: np.float32 T * N * D
            - masks: np.float32 T * N * D (same size as the inputs)
            - interval: np.float32 T * N * D (same size as the inputs)
           target: np.float32 T * N * E
    	'''
        self.db_name = db_name
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.iters = iters
        self.iters_to_report = iters_to_report

    def build_net(self):
        log.debug('Start Building Mask-RNN')
        model = model_helper.ModelHelper(name="mask_rnn")


        seq_lengths, input_blob, \
            hidden_init, target = model.net.AddExternalInputs(
                'seq_lengths',
                'input_blob',
                'hidden_init',
                'target',
        )

        hidden_output_all, self.hidden_output = MaskGRU(
            model, input_blob, seq_lengths, [hidden_init],
            self.input_dim, self.batch_size, scope="MaskRNN"
        )

        output = brew.fc(
            model,
            hidden_output_all,
            None,
            dim_in=self.hidden_size,
            dim_out=self.output_dim,
            axis=2
        )
        # axis is 2 as first two are T (time) and N (batch size).
        # We treat them as one big batch of size T * N
        output_reshaped, _ = model.net.Reshape(
            output, ['output_reshaped', '_'], shape=[-1, self.input_dim])

        # Create a copy of the current net. We will use it on the forward
        # pass where we don't need loss and backward operators
        self.predit_net = core.Net(model.net.Proto())

        # Add loss
        l2_dist = model.net.SquaredL2Distance(
            [output_reshaped, target], 'l2_dist')
        loss = model.net.AveragedLoss(l2_dist, 'loss')

        # Training net
        model.AddGradientOperators([loss])
        build_adam(
            model,
            base_learning_rate=0.1 * self.seq_length,
        )

        self.model = model
        self.predictions = output
        self.loss = loss

        # Create a net to copy hidden_output to hidden_init
        self.prepare_state = core.Net("prepare_state")
        self.prepare_state.Copy(self.hidden_output, hidden_init)
        # print(str(self.prepare_state.Proto()))

    def train(self):
        log.debug("Training model")

        workspace.RunNetOnce(self.model.param_init_net)
        # Writing to output states which will be copied to input
        # states within the loop below
        workspace.FeedBlob(self.hidden_output, np.zeros(
            [1, self.batch_size, self.hidden_size], dtype=np.float32
        ))
        workspace.CreateNet(self.prepare_state)
        # Copy hidden_ouput to hidden_init
        workspace.RunNet(self.prepare_state.Name())

        # Add external inputs
        inputs = build_input_reader(self.model, self.db_name, 'minidb', 
            ['seq_lengths', 'input_blob', 'target'], 
            batch_size = 3, data_type='train')
		workspace.FeedBlob('seq_lengths', inputs[0])
        workspace.FeedBlob('input_blob', inputs[1]) # concat of ...
        workspace.FeedBlob('target', inputs[2])

        CreateNetOnce(self.model.net)

        for i in range(self.iters):
            workspace.RunNet(self.model.net.Name())
            workspace.RunNet(self.prepare_state.Name())


def main():
    my_model = MaskRNN(
        'test.minidb',
        batch_size=3,
        input_dim=10,
        output_dim=1,
        hidden_size=4,
        iters_to_report=1,
    )
    my_model.build_net()

if __name__ == '__main__':
    main()