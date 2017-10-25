import sys
import os
user_dir = os.environ['HOME'] 

# Mingda
if user_dir == '/home/oscar':
	sys.path.append(user_dir + '/Documents/caffe2/build/')
if user_dir == '/Users/Mingda':
	sys.path.append(user_dir + '/Documents/Caffe2/caffe2/build/')
