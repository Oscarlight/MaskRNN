import exporter
import numpy as np
from caffe2.python import (
	workspace, core
)
from eval import MAUC, calcBCA
import matplotlib.pyplot as plt
import matplotlib
from pylab import rcParams
rcParams['figure.figsize'] = 7.6, 8
matplotlib.rcParams.update({'font.size': 22})

HIDDEN_DIM = 8
model_name = 'model10/MaskRNN'
save_path = 'hiddenDim8/'
# HIDDEN_DIM = 16
# model_name = 'model9/MaskRNN'
# save_path = 'hiddenDim16/'
# HIDDEN_DIM = 32
# model_name = 'model11/MaskRNN'
# save_path = 'hiddenDim32/'

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

def compute_mAUC_BCA(class_output, class_target, class_mask, start_end_index):
	estimLabels=[]; trueLabels=[]; data=[]
	for i in range(class_output.shape[0]): # each example
		start = start_end_index[i, 0]
		end = start_end_index[i, 1]
		for k in range(start, end+1): # in a seq
			if class_mask[i, k, 0] > 0.5: # class_mask
				estimLabels.append(np.argmax(class_output[i, k, :]))
				trueLabels.append(np.argmax(class_target[i, k, :]))
				# assert
				label, = np.where(class_target[i, k, :]==1.0)
				assert np.argmax(class_target[i, k, :]) == label[0]
				#
				data.append((label[0], class_output[i, k, :]))
	return [MAUC(data, 3), calcBCA(np.array(estimLabels), np.array(trueLabels), 3)]


if __name__ == '__main__':
	data_path = './data/npy4/'
	# index = 100
	tar_mean = np.load(data_path + 'tar_mean.npy').astype(np.float32)
	tar_std = np.load(data_path + 'tar_std.npy').astype(np.float32)
	print(tar_mean)
	print(tar_std)
	# quit()
	# train, valid, test
	error = {'train':[], 'valid':[], 'test':[]}
	# epochs = list(np.linspace(0, 90, num=10).astype(np.int32))
	epochs = list(np.linspace(0, 500, num=51).astype(np.int32))
	# epochs = list(np.linspace(100, 900, num=9).astype(np.int32)) + [999]
	# tests
	for data_type in ['train','valid','test']:
		(test_seq_lens, test_features, 
			test_targets, test_start_end_index) = load_data(data_path, data_type)		
		if data_type == 'train':
			test_regre_target = np.multiply(test_targets[:,:,3:6], tar_std) + tar_mean
		else:
			test_regre_target = test_targets[:,:,3:6]
		# test_regre_target = np.multiply(test_targets[:,:,3:6], tar_std) + tar_mean

		for index in epochs:
			print('>> computing for ' + str(index))
			target = predict(model_name, test_seq_lens, test_features, index)
			test_regre_output = np.multiply(target[1], tar_std) + tar_mean
			class_res = compute_mAUC_BCA(target[0], test_targets[:,:,0:3], 
				test_targets[:,:,6:7], test_start_end_index)
			reg_res = compute_regre_error(test_regre_output, test_regre_target, 
				test_targets[:,:,7:10], test_start_end_index)
			error[data_type].append(class_res + reg_res)

	# print(error)
	# quit()
	errors_by_type = {'mAUC':[], 'BCA':[], 'vennorm':[], 'adas13':[], 'mmse':[]}
	# assert len(error['train'])==10
	for j in range(len(error['train'])): # epochs
		for i, title in enumerate(['mAUC', 'BCA', 'vennorm', 'adas13', 'mmse']):
			error_tvt = [error[data_type][j][i] for data_type in ['train','valid','test']]
			errors_by_type[title].append(error_tvt)

	# print(errors_by_type)
	print('-'*50)
	print(str(HIDDEN_DIM))
	for i, title in enumerate(['mAUC', 'BCA', 'vennorm', 'adas13', 'mmse']):
		err_array = np.array(errors_by_type[title])
		# print(err_array)
		plt.plot(epochs, err_array[:,0], linewidth=2, label = 'train')
		plt.plot(epochs, err_array[:,1], linewidth=2, label = 'valid')
		plt.plot(epochs, err_array[:,2], linewidth=2, label = 'test')
		plt.legend()
		plt.title(title)
		plt.xlabel('epoch')
		print(title)
		if i > 1:
			print(np.min(err_array[:,1]), np.min(err_array[:,2]))
			plt.title(title + ' (MSE)' )
		else:
			plt.title(title)
			print(np.max(err_array[:,1]), np.max(err_array[:,2]))
		plt.savefig(save_path + model_name.split('/')[0] + '_' + title + '.pdf')
		plt.clf()



		