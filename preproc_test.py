import caffe2_path
from data_reader import write_db
import numpy as np
import os
# example input data
SEQ_LEN = 12
NUM_EXAMPLE = 10
INPUT_DIM = 2
CLASS_OUTPUT_DIM = 2
REGRE_OUTPUT_DIM = 2

# In order to put into batches, the input is
# [NUM_EXAMPLE, SEQ_LEN, INPUT_DIM] 
# i.e. the first dim is the num of example 
# However the required input dim is:
# [SEQ_LEN, NUM_EXAMPLE, INPUT_DIM]
seq_lens = np.random.randint(
	SEQ_LEN+1, size=(NUM_EXAMPLE, 1)
).astype(np.int32)

# padding zeros at the end of the list
x = np.random.rand(
	NUM_EXAMPLE, SEQ_LEN, INPUT_DIM
).astype(np.float32)
# Caution: use nanmean in the real world
x_mean = np.repeat(
	np.expand_dims(
		np.nanmean(x, axis=1),
		axis=1,
	),
	SEQ_LEN, axis=1
)
assert x_mean.shape == x.shape
# binary mask
mask = np.random.randint(
	2, size=(NUM_EXAMPLE, SEQ_LEN, INPUT_DIM)
).astype(np.float32) 

interval = np.random.randint(
	3, size=(NUM_EXAMPLE, SEQ_LEN, INPUT_DIM)
).astype(np.float32)

# target
_class_target_0 = np.zeros(
	(NUM_EXAMPLE, SEQ_LEN, CLASS_OUTPUT_DIM - 1)
).astype(np.float32)
_class_target_1 = np.ones(
	(NUM_EXAMPLE, SEQ_LEN, 1)
).astype(np.float32)

class_target = np.concatenate((_class_target_0, _class_target_1), axis=2)

print(class_target.shape)

class_target_mask = np.random.randint(
	2, size=(NUM_EXAMPLE, SEQ_LEN, 1)
).astype(np.float32)

regre_target = np.ones(
	(NUM_EXAMPLE, SEQ_LEN, REGRE_OUTPUT_DIM)
).astype(np.float32)

regre_target_mask = np.random.randint(
	2, size=(NUM_EXAMPLE, SEQ_LEN, REGRE_OUTPUT_DIM)
).astype(np.float32)


inputs = np.concatenate((x, x, x_mean, mask, interval), axis=2)
print(x.shape, x_mean.shape, mask.shape, interval.shape)
print(seq_lens.shape)
print(inputs.shape)

db_name = 'test.minidb'
if os.path.isfile(db_name): 
	os.remove(db_name)
write_db('minidb', db_name, 
	[seq_lens, inputs, class_target, regre_target, class_target_mask, regre_target_mask])