import numpy as np

a = np.random.rand(3, 2, 1)
b = np.transpose(a, (1, 0, 2))

for i in range(3):
	for j in range(2):
		for k in range(1):
			assert a[i, j, k] == b[j, i, k]
