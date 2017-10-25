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
    	(T: seq size, N: batch size, D: input dim)
    	seq_lengths: np.array of (N, 1) with each element is the seq length.
    	'''
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.iters = iters
        self.iters_to_report = iters_to_report

    def build_net(self):
        log.debug('Start Building Mask-RNN')
        model = model_helper.ModelHelper(name="mask_rnn")

        input_blob, seq_lengths, \
            hidden_init, target = model.net.AddExternalInputs(
                'input_blob',
                'seq_lengths',
                'hidden_init',
                'target',
        )
        # the input sequence in a format T x N x D
        # where T: sequence size (i.e. time), N: batch size and D: input dimension
        hidden_output_all, self.hidden_output = GRU(
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
		workspace.FeedBlob("seq_lengths", self.seq_lengths)
		# inputs: np.float32 [self.seq_length, self.batch_size, self.input_dim]
		# target: np.float32 [self.seq_length, self.batch_size, self.output_dim]
        workspace.FeedBlob('input_blob', inputs)
        workspace.FeedBlob('target', target)

        CreateNetOnce(self.model.net)
        workspace.RunNet(self.model.net.Name())



def main():        # print(model)
    my_model = MaskRNN(
        seq_length=5,
        batch_size=3,
        input_dim=10,
        output_dim=1,
        hidden_size=4,
        iters_to_report=1,
    )
    my_model.build_net()

if __name__ == '__main__':
    main()