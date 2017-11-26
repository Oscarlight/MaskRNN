import caffe2_path
from caffe2.python import (
    core, workspace, model_helper, utils, brew, net_drawer, 
)
from mask_gru_cell import MaskGRU
from caffe2.python.optimizer import build_adam
from data_reader import build_input_reader
import numpy as np
import logging
import matplotlib.pyplot as plt


