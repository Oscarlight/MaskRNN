import caffe2_path
from caffe2.python import (
    core, workspace, model_helper, utils, brew, net_drawer, 
)
# from caffe2.python.gru_cell import GRU
from mask_gru_cell import MaskGRU
from caffe2.python.optimizer import build_adam
from data_reader import build_input_reader
import exporter
import numpy as np
import logging
import pickle
import os
import matplotlib.pyplot as plt

logging.basicConfig()
log = logging.getLogger("mask_rnn")
log.setLevel(logging.DEBUG)

# Default set() here is intentional as it would accumulate values like a global
# variable
def CreateNetOnce(net, created_names=set()): # noqa
    name = net.Name()
    if name not in created_names:
        created_names.add(name)
        workspace.CreateNet(net)

class MaskRNN(object):
    def __init__(
        self,
        model_name, 
        db_name,
        seq_size,
        batch_size, 
        input_dim,
        class_output_dim,
        regre_output_dim,
        hidden_size,
        ):
    	'''
        The db contains: (T: seq length, N: batch size, D: input dim, E: output dim)
    	   seq_lengths: np.array of (N, 1) with each element is the seq length.
           input_blob: the concat (axis = 2) of:
            - inputs: np.float32 T * N * D
            - inputs_last: np.float32 T * N * D
            - inputs_mean: np.float32 T * N * D
            - masks: np.float32 T * N * D (same size as the inputs)
            - interval: np.float32 T * N * D (same size as the inputs)
           target: np.float32 T * N * E
    	'''
        workspace.ResetWorkspace()
        self.model_name = model_name
        self.db_name = db_name
        self.seq_size = seq_size
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.class_output_dim = class_output_dim
        self.regre_output_dim = regre_output_dim
        self.hidden_size = hidden_size
        self.net_store = {}
        self.reports = {
            'epoch' : [],
        }

    def build_net(
        self,
        base_learning_rate=0.1  # base_learning_rate * seq_size
        ):
        log.debug('>>> Building Mask-RNN')
        model = model_helper.ModelHelper(name="mask_rnn")

        hidden_init= model.net.AddExternalInputs(
            'hidden_init',
        )
        # TODO: do I still need this?
        model.net.AddExternalInputs(
            'input_blob',
            'seq_lengths',
            'target',
        )
        # Add external inputs (read directly from the database)
        # the dimension of class_target_mask: [BATCH_SIZE, SEQ_LEN, 1]
        # the dimension of regre_target_mask: [BATCH_SIZE, SEQ_LEN, regre_output_dim]
        (seq_lengths, _input_blob, 
            _class_target, _regre_target, 
            _class_target_mask, _regre_target_mask) = build_input_reader(
            model, self.db_name, 'minidb', 
            ['seq_lengths', 
             'input_blob_batch_first', 
             'class_target_batch_first',
             'regre_target_batch_first',
             'class_target_mask_batch_first',
             'regre_target_mask_batch_first'], 
            batch_size = self.batch_size, data_type='train'
        )

        # In order to put into batches, the input_blob is
        # [BATCH_SIZE, SEQ_LEN, INPUT_DIM] 
        # i.e. the first dim is the batch size 
        # However the required input dim is:
        # [SEQ_LEN, BATCH_SIZE, INPUT_DIM]
        input_blob = model.net.Transpose(
            [_input_blob], 'input_blob', axes=[1, 0, 2])
        class_target = model.net.Transpose(
            [_class_target], 'class_target', axes=[1, 0, 2])
        regre_target = model.net.Transpose(
            [_regre_target], 'regre_target', axes=[1, 0, 2])
        class_target_mask = model.net.Transpose(
            [_class_target_mask], 'class_target_mask', axes=[1, 0, 2])
        regre_target_mask = model.net.Transpose(
            [_regre_target_mask], 'regre_target_mask', axes=[1, 0, 2])

        hidden_output_all, self.hidden_output = MaskGRU(
            model, input_blob, seq_lengths, (hidden_init,),
            self.input_dim, self.hidden_size, scope="MaskRNN"
        )

        # axis is 2 as first two are T (time) and N (batch size)
        # multi-task learning: regression
        regre_output = brew.fc(
            model,
            hidden_output_all,
            None,
            dim_in=self.hidden_size,
            dim_out=self.regre_output_dim,
            axis=2
        )
        # multi-task learning: classification
        class_output = brew.fc(
            model,
            hidden_output_all,
            None,
            dim_in=self.hidden_size,
            dim_out=self.class_output_dim,
            axis=2
        )
        # softmax head for testing only
        class_softmax_output = model.net.Softmax(
            class_output, 'class_softmax_output', axis=2)

        # Get the predict net
        (self.net_store['predict'], 
            self.external_inputs) = model_helper.ExtractPredictorNet(
            model.net.Proto(),
            [input_blob, seq_lengths, hidden_init],
            [class_softmax_output, regre_output],
        )

        # Then, we add loss and gradient ops
        # We treat them as one big batch of size T * N
        # we use the logit of classification head
        class_output_reshaped, _ = model.net.Reshape(
            class_output, ['class_output_reshaped', '_class_output_shape'], 
            shape=[-1, self.class_output_dim])
        regre_output_reshaped, _ = model.net.Reshape(
            regre_output, ['regre_output_reshaped', '_regre_output_shape'], 
            shape=[-1, self.regre_output_dim])

        class_target_reshaped, _ = model.net.Reshape(
            class_target, ['class_target_reshaped', '_class_target_shape'], 
            shape=[-1, self.class_output_dim])
        regre_target_reshaped, _ = model.net.Reshape(
            regre_target, ['regre_target_reshaped', '_regre_target_shape'], 
            shape=[-1, self.regre_output_dim])

        class_target_mask_reshaped, _ = model.net.Reshape(
            class_target, ['class_target_mask_reshaped', '_class_target_mask_shape'], 
            shape=[-1, 1])
        regre_target_mask_reshaped, _ = model.net.Reshape(
            regre_target, ['regre_target_mask_reshaped', '_regre_target_mask_shape'], 
            shape=[-1, self.regre_output_dim])

        # stop gradient to label and mask
        class_target_reshaped = model.net.StopGradient(
            class_target_reshaped, 'stopped_class_target_reshaped'
        )
        regre_target_reshaped = model.net.StopGradient(
            regre_target_reshaped, 'stopped_regre_target_reshaped'
        )
        class_target_mask_reshaped = model.net.StopGradient(
            class_target_mask_reshaped, 'stopped_class_target_mask_reshaped'
        )
        regre_target_mask_reshaped = model.net.StopGradient(
            regre_target_mask_reshaped, 'stopped_regre_target_mask_reshaped'
        )

        # model.net.Print([class_output_reshaped], 'print', to_file=0)
        # classification error
        # combined softmax and log likelihood for numerical stability
        # weighted by class_target_mask_reshaped
        _, class_average_loss = model.net.SoftmaxWithLoss(
            [class_output_reshaped, class_target_reshaped, class_target_mask_reshaped],
            ['_train_softmax_ouput', 'class_average_loss'], label_prob=1
        )
        # regression error
        # mask need to be applied to *each* individual dimension of output vector
        regre_output_reshaped_list = model.net.Split(
            [regre_output_reshaped],
            ['regre_output_reshaped_' + str(i) for i in range(self.regre_output_dim)],
            axis=1, # has been reshaped to 2D tensor
        )
        regre_target_reshaped_list = model.net.Split(
            [regre_target_reshaped],
            ['regre_target_reshaped_' + str(i) for i in range(self.regre_output_dim)],
            axis=1, # has been reshaped to 2D tensor
        )        
        regre_target_mask_reshaped_list = model.net.Split(
            [regre_target_mask_reshaped],
            ['regre_target_mask_reshaped_' + str(i) for i in range(self.regre_output_dim)],
            axis=1, # has been reshaped to 2D tensor
        )   
        regre_average_loss_lst = []; i = 0
        for o, t, m in zip(
            regre_output_reshaped_list, 
            regre_target_reshaped_list, 
            regre_target_mask_reshaped_list):
            l2_dist = model.net.SquaredL2Distance(
                [o, t], 'l2_dist_' + str(i))
            m = model.net.Squeeze(
                m, 'squeezed_regre_target_mask_' + str(i), dims=[1])
            weighted_l2_dist = model.net.Mul(
                [l2_dist, m], 'weighted_l2_dist_' + str(i))
            regre_average_loss_lst.append(model.net.AveragedLoss(
                weighted_l2_dist, 'regre_average_loss_' + str(i)))
            i += 1

        assert i == self.regre_output_dim, 'output dim != # of loss split'

        # Training net
        model.AddGradientOperators([class_average_loss] + regre_average_loss_lst)
        build_adam(
            model,
            base_learning_rate=base_learning_rate*self.seq_size,
        )

        self.model = model
        self.predictions = [class_softmax_output, regre_output]
        self.loss = [class_average_loss] + regre_average_loss_lst
        for loss in self.loss:
            loss = str(loss)
            self.reports[loss] = []

        # Create a net to copy hidden_output to hidden_init
        prepare_state = core.Net("prepare_state")
        prepare_state.Copy(self.hidden_output, hidden_init)
        self.net_store['prepare'] = prepare_state
        self.net_store['train'] = core.Net(model.net.Proto())

    def train(
        self, 
        iters,
        iters_to_report=1, 
        ):
        log.debug(">>> Training Mask-RNN")

        workspace.RunNetOnce(self.model.param_init_net)
        # initialize the output states which will be copied to input
        # states within the loop below
        workspace.FeedBlob(self.hidden_output, np.zeros(
            [1, self.batch_size, self.hidden_size], dtype=np.float32
        ))
        # Create the prepare net and train net
        workspace.CreateNet(self.net_store['prepare'])

        for num_iter in range(iters):
            # Reset output state
            workspace.FeedBlob(self.hidden_output, np.zeros(
                [1, self.batch_size, self.hidden_size], dtype=np.float32
            ))
            # Copy hidden_ouput to hidden_init
            workspace.RunNet(self.net_store['prepare'].Name())
            CreateNetOnce(self.model.net)
            workspace.RunNet(self.model.net.Name())

            if num_iter % iters_to_report == 0:
                self.reports['epoch'].append(num_iter)
                for loss in self.loss:
                    loss = str(loss)
                    self.reports[loss].append(
                        workspace.FetchBlob(loss)
                    )

        print('>>> Saving test model')

        # Save Net
        exporter.save_net(
            self.net_store['predict'], 
            self.model.param_init_net, 
            self.model_name+'_init', self.model_name+'_predict'
        )

        # Save report
        with open(self.model_name + '_report.pickle',"wb") as pickle_file:
            pickle.dump(self.reports, pickle_file)


    def draw_nets(self, plot_train=False):
        for net_name in self.net_store:
            net = self.net_store[net_name]
            if net_name != 'train' or plot_train:
                graph = net_drawer.GetPydotGraph(net.Proto().op, rankdir='TB')
                with open(self.model_name + '_' + net.Name() + ".png",'wb') as f:
                    f.write(graph.create_png())
            with open(self.model_name + '_' + net.Name() + "_proto.txt",'wb') as f:
                f.write(str(net.Proto()))

    def plot_loss_trend(self):
        for loss in self.loss:
            loss = str(loss)
            plt.plot(
                self.reports['epoch'], 
                self.reports[loss], 
                label=loss
            )
        plt.legend()
        plt.show()


# Sanity test only
def main():
    SEQ_LEN = 12
    NUM_EXAMPLE = 10
    INPUT_DIM = 2
    CLASS_OUTPUT_DIM = 2
    REGRE_OUTPUT_DIM = 2
    model_path = 'model1/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    my_model = MaskRNN(
        model_path + 'MaskRNN_test',
        'test.minidb',
        seq_size=SEQ_LEN,
        batch_size=100,
        input_dim=INPUT_DIM,
        class_output_dim=CLASS_OUTPUT_DIM,
        regre_output_dim=REGRE_OUTPUT_DIM,
        hidden_size=6,
    )
    my_model.build_net(base_learning_rate=0.1)
    my_model.draw_nets()
    my_model.train(
        iters=10
    )
    my_model.plot_loss_trend()

if __name__ == '__main__':
    main()