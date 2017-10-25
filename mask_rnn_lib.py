import caffe2_path
from caffe2.python import (
	core, workspace, model_helper, utils, brew
)
from caffe2.python.gru_cell import GRU
import logging
logging.basicConfig()
log = logging.getLogger("mask_rnn")
log.setLevel(logging.DEBUG)

class MaskRNN(object):
	def __init__(self, 
		seq_length, 
		batch_size, 
		input_dim,
		hidden_size,
		iters_to_report,
		):
		self.seq_length = seq_length
		self.batch_size = batch_size
		self.D = input_dim
		self.hidden_size = hidden_size
		self.iters_to_report = iters_to_report

	def build_net(self):
		log.debug('Start Building Mask-RNN')
		model = model_helper.ModelHelper(name="mask_rnn")
		input_blob, seq_lengths, hidden_init, target = \
            model.net.AddExternalInputs(
                'input_blob',
                'seq_lengths',
                'hidden_init',
                'target',
            )
        results = GRU(
        	model, input_blob, seq_lengths, [hidden_init],
        	self.D, self.batch_size
        	)
        print(results)

def main():
	parser = argparse.ArgumentParser(
        description="Caffe2: Char RNN Training"
    )
	parser.add_argument("--train_data", type=str, default=None,
                        help="Path to training data in a text file format",
                        required=True)
    parser.add_argument("--seq_length", type=int, default=25,
                        help="One training example sequence length")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Training batch size")

    args = parser.parse_args()
    my_model = MaskRNN(args)

if __name__ == '__main__':
	main()