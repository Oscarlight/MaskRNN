import numpy as np

# data_path = './data/new/'
# train_start_end_index = np.load(data_path + 'my_train_start_end_index.npy').astype(np.float32)
# print(train_start_end_index[1,1])

# train_targets = np.load(data_path + 'my_train_target.npy').astype(np.float32)
# tar_mean = np.load(data_path + 'tar_mean.npy').astype(np.float32)
# tar_std = np.load(data_path + 'tar_std.npy').astype(np.float32)

# print(tar_mean)
# print(tar_std)

# a = np.multiply(train_targets[:,:,3:6], tar_std) + tar_mean
# assert np.array_equal(a[:,:,0],train_targets[:,:,3] * tar_std[0] + tar_mean[0])
# assert np.array_equal(a[:,:,1],train_targets[:,:,4] * tar_std[1] + tar_mean[1])
# assert np.array_equal(a[:,:,2],train_targets[:,:,5] * tar_std[2] + tar_mean[2])
a = np.array([0, 1, 0])
b=np.argmax(a)
print(b)