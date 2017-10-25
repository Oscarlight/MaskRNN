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
        seq_lengths,
        batch_size, 
        input_dim,
        output_dim,
        hidden_size,
        iters_to_report,
        ):
    
        self.seq_lengths = seq_lengths
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
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

        # As though we predict the same probability for each character
        smooth_loss = -np.log(1.0 / self.D) * self.seq_length
        last_n_iter = 0
        last_n_loss = 0.0
        num_iter = 0

        # Writing to output states which will be copied to input
        # states within the loop below
        workspace.FeedBlob(self.hidden_output, np.zeros(
            [1, self.batch_size, self.hidden_size], dtype=np.float32
        ))
        workspace.CreateNet(self.prepare_state)

        # We iterate over text in a loop many times. Each time we peak
        # seq_length segment and feed it to LSTM as a sequence
        last_time = datetime.now()
        progress = 0
        while True:
            workspace.FeedBlob(
                "seq_lengths", self.seq_lengths
            )
            workspace.RunNet(self.prepare_state.Name())

            input = np.zeros(
                [self.seq_length, self.batch_size, self.D]
            ).astype(np.float32)
            target = np.zeros(
                [self.seq_length * self.batch_size]
            ).astype(np.int32)

            for e in range(self.batch_size):
                for i in range(self.seq_length):
                    pos = text_block_starts[e] + text_block_positions[e]
                    input[i][e][self._idx_at_pos(pos)] = 1
                    target[i * self.batch_size + e] =\
                        self._idx_at_pos((pos + 1) % N)
                    text_block_positions[e] = (
                        text_block_positions[e] + 1) % text_block_sizes[e]
                    progress += 1

            workspace.FeedBlob('input_blob', input)
            workspace.FeedBlob('target', target)

            CreateNetOnce(self.model.net)
            workspace.RunNet(self.model.net.Name())

            num_iter += 1
            last_n_iter += 1

            if num_iter % self.iters_to_report == 0:
                new_time = datetime.now()
                print("Characters Per Second: {}". format(
                    int(progress / (new_time - last_time).total_seconds())
                ))
                print("Iterations Per Second: {}". format(
                    int(self.iters_to_report /
                        (new_time - last_time).total_seconds())
                ))

                last_time = new_time
                progress = 0

                print("{} Iteration {} {}".
                      format('-' * 10, num_iter, '-' * 10))

            loss = workspace.FetchBlob(self.loss) * self.seq_length
            smooth_loss = 0.999 * smooth_loss + 0.001 * loss
            last_n_loss += loss

            if num_iter % self.iters_to_report == 0:
                self.GenerateText(500, np.random.choice(self.vocab))

                log.debug("Loss since last report: {}"
                          .format(last_n_loss / last_n_iter))
                log.debug("Smooth loss: {}".format(smooth_loss))

                last_n_loss = 0.0
                last_n_iter = 0

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