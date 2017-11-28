import exporter
import numpy as np
from caffe2.python import (
	workspace, core
)
SEQ_LEN = 19
HIDDEN_DIM = 256

def predict(model_name, seq_lens, inputs, index):
	inputs = np.transpose(inputs, (1, 0, 2))
	workspace.FeedBlob('seq_lengths', seq_lens)
	workspace.FeedBlob('input_blob', inputs)
	workspace.FeedBlob('hidden_init',
		np.zeros([1, seq_lens.shape[0], HIDDEN_DIM], dtype=np.float32))

	pred_net = exporter.load_net(
		model_name+str(index)+'_init',
		model_name+str(index)+'_predict',
	)
	workspace.RunNet(pred_net)
	class_output = workspace.FetchBlob('class_softmax_output')
	reg_output = workspace.FetchBlob('mask_rnn_blob_0')
	class_output = np.transpose(class_output, (1, 0, 2))
	reg_output = np.transpose(reg_output, (1, 0, 2))
	return (class_output, reg_output)

def load_data(data_path, data_type):
	train_seq_lens = np.load(data_path + 'my_'+data_type+'_SEQLEN.npy').astype(np.int32)
	train_features = np.load(data_path + 'my_'+data_type+'_features.npy').astype(np.float32)
	train_targets = np.load(data_path + 'my_'+data_type+'_target.npy').astype(np.float32)
	train_start_end_index = np.load(data_path + 'my_'+data_type+'_start_end_index.npy')
	train_seq_lens = train_seq_lens[:, 1:2]
	return (train_seq_lens, train_features, train_targets, train_start_end_index)

def compute_regre_error(reg_output, reg_target, reg_mask, start_end_index):
	mse_dict = {0:[], 1:[], 2:[]}
	for i in range(reg_output.shape[0]): # each example
		start = start_end_index[i, 0]
		end = start_end_index[i, 1]
		for k in range(start, end+1): # in a seq
			for j in range(3): # for each regression targets
				if reg_mask[i, k, j] > 0.5:
					mse_dict[j].append((reg_output[i, k, j] - reg_target[i, k, j])**2)

	mse = []
	for j in range(3):
		mse.append(np.mean(np.array(mse_dict[j])))
	return mse

def compute_mAUC(class_output, class_target, class_mask, start_end_index):


if __name__ == '__main__':
	data_path = './data/new/'
	model_name = 'model2/MaskRNN'
	index = 100
	tar_mean = np.load(data_path + 'tar_mean.npy').astype(np.float32)
	tar_std = np.load(data_path + 'tar_std.npy').astype(np.float32)
	# train
	(train_seq_lens, train_features, 
		train_targets, train_start_end_index) = load_data(data_path, 'train')
	target = predict(model_name, train_seq_lens, train_features, index)
	train_regre_target = np.multiply(train_targets[:,:,3:6], tar_std) + tar_mean
	train_regre_output = np.multiply(target[1], tar_std) + tar_mean
	print(compute_regre_error(train_regre_output, train_regre_target, 
		train_targets[:,:,7:10], train_start_end_index))
	# test
	for data_type in ['valid','test']:
		(test_seq_lens, test_features, 
			test_targets, test_start_end_index) = load_data(data_path, data_type)		
		target = predict(model_name, test_seq_lens, test_features, index)
		test_regre_target = test_targets[:,:,3:6]
		test_regre_output = np.multiply(target[1], tar_std) + tar_mean
		print(compute_regre_error(test_regre_output, test_regre_target, 
			test_targets[:,:,7:10], test_start_end_index))
		quit()