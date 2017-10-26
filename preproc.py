import caffe2_path
from data_reader import write_db
import numpy as np
import os
# example input data
# padding zeros at the end of the list
seq_lens = np.array([[4], [3], [2]]).astype(np.float32)
x = np.array(
	[[[1., 1.], [2., 1.], [3., 2.], [4., 4.]],
     [[0.5, 0.5], [0.7, 0.7], [0.9, 0.9], [0, 0]],
     [[2., 2.], [3.5, 3.5], [0, 0], [0, 0]]], dtype=np.float32)
# caution do nanmean in the real world
x_mean = np.repeat(
	np.expand_dims(
		np.nanmean(x, axis=1),
		axis=1,
	),
	4, axis=1
)
assert x_mean.shape == x.shape
mask = np.array(
	[[[1., 0.], [1., 1.], [1., 0.], [1., 1.]],
     [[1., 0.], [0., 1.], [1., 1.], [0., 0.]],
     [[0., 1.], [0., 0.], [0., 0.], [0., 0.]]], dtype=np.float32)
interval = np.array(
	[[[0, 0], [1, 1], [2, 0], [3, 2]],
     [[0, 0], [1, 1], [2, 2], [0, 0]],
     [[0, 0], [1, 1], [0, 0], [0, 0]]], dtype=np.float32)
target = np.array(
	[[[1], [2], [3], [4]],
     [[1], [2], [3], [4]],
     [[1], [2], [3], [4]]], dtype=np.float32)
inputs = np.concatenate((x, x_mean, mask, interval), axis=2)

db_name = 'test.minidb'
os.remove(db_name)
write_db('minidb', db_name, 
	[seq_lens, inputs, target])