import caffe2_path
from caffe2.python import (
    core, workspace
)
from mask_rnn_lib import MaskRNN
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import argparse

SEQ_LEN = 19
NUM_EXAMPLE = 655
INPUT_DIM = 745
CLASS_OUTPUT_DIM = 3
REGRE_OUTPUT_DIM = 3


parser = argparse.ArgumentParser(description='MaskRNN')
parser.add_argument("-model_path", type=str,
                    help="model4/")
parser.add_argument("-hidden_size", type=int,
                    help="")

args = parser.parse_args()
model_path = args.model_path

if not os.path.exists(model_path):
    os.makedirs(model_path)
my_model = MaskRNN(
    model_path + 'MaskRNN',
    'train_data.minidb',
    seq_size=SEQ_LEN,
    batch_size=655,
    input_dim=INPUT_DIM,
    class_output_dim=CLASS_OUTPUT_DIM,
    regre_output_dim=REGRE_OUTPUT_DIM,
    hidden_size=args.hidden_size,
)
my_model.build_net(base_learning_rate=0.001)
my_model.draw_nets()
my_model.train(
    iters=5000,
    iters_to_report=100,
)
# my_model.plot_loss_trend()

