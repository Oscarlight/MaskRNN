import exporter
import numpy as np
from caffe2.python import (
	workspace, core
)
SEQ_LEN = 19
BATCHSIZE = 655
HIDDEN_DIM = 256

def predict(model_name, seq_lens, inputs, index):
	workspace.FeedBlob('seq_lengths', seq_lens)
	workspace.FeedBlob('input_blob', inputs)

	# Create the prepare net
	# hidden_init= model.net.AddExternalInputs('hidden_init')
	hidden_init = 'hidden_init'
	hidden_output = 'MaskRNN/hidden_t_last'
	prepare_net = core.Net("prepare_state")
	prepare_net.Copy(hidden_output, hidden_init)
	workspace.FeedBlob(hidden_output, 
		np.zeros([1, BATCHSIZE, HIDDEN_DIM], dtype=np.float32))
	workspace.CreateNet(prepare_net)
	workspace.RunNet(prepare_net.Name())

	pred_net = exporter.load_net(
		model_name+str(index)+'_init',
		model_name+str(index)+'_predict',
	)
	workspace.RunNet(pred_net)


if __name__ == '__main__':
	data_path = './data/'
	train_seq_lens = np.load(data_path + 'my_train_SEQLEN.npy').astype(np.int32)
	train_features = np.load(data_path + 'my_train_features.npy').astype(np.float32)
	train_targets = np.load(data_path + 'my_train_target.npy').astype(np.float32)

	model_name = 'model1/MaskRNN'
	target = predict(model_name, train_seq_lens, train_features, 100)