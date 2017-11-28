import caffe2_path
from data_reader import write_db
import numpy as np
import os

SEQ_LEN = 19
NUM_EXAMPLE = 655
INPUT_DIM = 745
CLASS_OUTPUT_DIM = 3
REGRE_OUTPUT_DIM = 3

data_path = './data/'
train_seq_lens = np.load(data_path + 'my_train_SEQLEN.npy').astype(np.int32)
train_features = np.load(data_path + 'my_train_features.npy').astype(np.float32)
train_targets = np.load(data_path + 'my_train_target.npy').astype(np.float32)
train_seq_lens = train_seq_lens[:, 1:2]
train_class_targets = train_targets[:, :, 0:3]
train_regre_targets = train_targets[:, :, 3:6]
train_class_target_masks = train_targets[:, :, 6:7]
train_regre_target_masks = train_targets[:, :, 7:10]

count = 0
for i in range(train_class_targets.shape[0]):
	for j in range(train_class_targets.shape[1]):
		if sum(train_class_targets[i, j, :]) == 0:
			train_class_targets[i, j, 0]=1/3.
			train_class_targets[i, j, 1]=1/3.
			train_class_targets[i, j, 2]=1/3.
			count+=1

print(SEQ_LEN * NUM_EXAMPLE, count)

count = 0
for i in range(train_regre_target_masks.shape[0]):
	for j in range(train_regre_target_masks.shape[1]):
		if train_regre_target_masks[i, j, 1] == 1:
			count+=1

print(SEQ_LEN * NUM_EXAMPLE, count)

quit()

assert train_seq_lens.shape == (NUM_EXAMPLE, 1)
assert train_features.shape == (NUM_EXAMPLE, SEQ_LEN, INPUT_DIM * 5)
assert train_class_targets.shape == (NUM_EXAMPLE, SEQ_LEN, CLASS_OUTPUT_DIM)
assert train_regre_targets.shape == (NUM_EXAMPLE, SEQ_LEN, REGRE_OUTPUT_DIM)
assert train_class_target_masks.shape == (NUM_EXAMPLE, SEQ_LEN, 1)
assert train_regre_target_masks.shape == (NUM_EXAMPLE, SEQ_LEN, REGRE_OUTPUT_DIM)

db_name = 'train_data.minidb'
if os.path.isfile(db_name): 
	os.remove(db_name)
write_db('minidb', db_name, 
	[
		train_seq_lens, 
		train_features, 
		train_class_targets,
		train_regre_targets,
		train_class_target_masks,
		train_regre_target_masks
	]
)