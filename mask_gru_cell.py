from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools
from caffe2.python import brew, rnn_cell, scope
import numpy as np


class MaskGRUCell(rnn_cell.RNNCell):

    def __init__(
        self,
        input_size,   # input feature dims (i.e. D)
        hidden_size,
        forget_bias,  # Currently unused!  Values here will be ignored.
        memory_optimization,
        drop_states=False,
        **kwargs
    ):
        super(MaskGRUCell, self).__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forget_bias = float(forget_bias)
        self.memory_optimization = memory_optimization
        self.drop_states = drop_states

    # Unlike LSTMCell, GRUCell needs the output of one gate to feed into another.
    # (reset gate -> output_gate)
    # So, much of the logic to calculate the reset gate output and modified
    # output gate input is set here, in the graph definition.
    # The remaining logic lives in in gru_unit_op.{h,cc}.
    def _apply(
        self,
        model,
        input_t,
        seq_lengths,
        states,
        timestep,
        extra_inputs=None,
    ):
        hidden_t_prev = states[0]

        ## If we need the intervals to apply decay on hidden state
        # split_info = model.net.GivenTensorIntFill(
        #     [], self.scope("split_info"),
        #     values = np.array([[
        #         self.hidden_size, self.hidden_size, 
        #         self.hidden_size, self.input_size]])
        # )

        # Split input tensors to get inputs for each gate.
        (input_t_reset, input_t_update, 
            input_t_output) = model.net.Split(
            [
                input_t,
                # split_info
            ],
            [
                self.scope('input_t_reset'),
                self.scope('input_t_update'),
                self.scope('input_t_output'),
                # self.scope('intervals')
            ],
            axis=2,
        )
        # decay on hidden state
        # decays = self.build_decay(
        #     model, intervals, 
        #     self.input_size, self.hidden_size, 
        #     'hidden_decay'
        # )
        # hidden_t_prev = model.net.Mul(
        #     [hidden_t_prev, decays],
        #     self.scope('decayed_hidden_t_prev'),
        #     broadcast=0
        # )

        # Fully connected layers for reset and update gates.
        reset_gate_t = brew.fc(
            model,
            hidden_t_prev,
            self.scope('reset_gate_t'),
            dim_in=self.hidden_size,
            dim_out=self.hidden_size,
            axis=2,
        )
        update_gate_t = brew.fc(
            model,
            hidden_t_prev,
            self.scope('update_gate_t'),
            dim_in=self.hidden_size,
            dim_out=self.hidden_size,
            axis=2,
        )

        # Calculating the modified hidden state going into output gate.
        reset_gate_t = model.net.Sum(
            [reset_gate_t, input_t_reset],
            self.scope('reset_gate_t')
        )
        reset_gate_t_sigmoid = model.net.Sigmoid(
            reset_gate_t,
            self.scope('reset_gate_t_sigmoid')
        )
        modified_hidden_t_prev = model.net.Mul(
            [reset_gate_t_sigmoid, hidden_t_prev],
            self.scope('modified_hidden_t_prev')
        )
        output_gate_t = brew.fc(
            model,
            modified_hidden_t_prev,
            self.scope('output_gate_t'),
            dim_in=self.hidden_size,
            dim_out=self.hidden_size,
            axis=2,
        )

        # Add input contributions to update and output gate.
        # We already (in-place) added input contributions to the reset gate.
        update_gate_t = model.net.Sum(
            [update_gate_t, input_t_update],
            self.scope('update_gate_t'),
        )
        output_gate_t = model.net.Sum(
            [output_gate_t, input_t_output],
            self.scope('output_gate_t'),
        )

        # Join gate outputs and add input contributions
        gates_t, _gates_t_concat_dims = model.net.Concat(
            [
                reset_gate_t,
                update_gate_t,
                output_gate_t,
            ],
            [
                self.scope('gates_t'),
                self.scope('_gates_t_concat_dims'),
            ],
            axis=2,
        )

        hidden_t = model.net.GRUUnit(
            [
                hidden_t_prev,
                gates_t,
                seq_lengths,
                timestep,
            ],
            list(self.get_state_names()),
            forget_bias=self.forget_bias,
            drop_states=self.drop_states,
        )
        model.net.AddExternalOutputs(hidden_t)
        return (hidden_t,)

    def build_decay(
        self,
        model, 
        intervals, 
        input_size, 
        output_size, 
        namescope
    ):
        with scope.NameScope(namescope):
            decays = brew.fc(
                model,
                intervals,
                self.scope('intervals_fc'),
                dim_in=input_size,
                dim_out=output_size,
                axis=2,
            )
            ZEROS = model.net.ConstantFill(
                [decays], 
                self.scope("ZEROS"), value=0.0
            )
            # in-place update
            decays = model.net.Max(
                [decays, ZEROS],
                self.scope("max_intervals_fc")
            )
            decays = model.net.Negative(
                [decays],
                self.scope("neg_max_interval_fc")
            )
            decays = model.net.Exp(
                [decays],
                self.scope("decays")
            )
        return decays


    def prepare_input(self, model, input_blob):
        '''
           input_blob: the concat (axis = 2) of:
            - inputs: np.float32 T * N * D
            - inputs_last: np.float32 T * N * D
            - inputs_mean: np.float32 T * N * D
            - masks: np.float32 T * N * D (same size as the inputs)
            - interval: np.float32 T * N * D (same size as the inputs)
        '''
        # Split input blobs to get inputs for ...
        # equal-sized split
        (inputs, inputs_last, inputs_mean, 
            masks, intervals) = model.net.Split(
            [
                input_blob,
            ],
            [
                self.scope('inputs'),
                self.scope('inputs_last'),
                self.scope('inputs_mean'),
                self.scope('masks'),
                self.scope('intervals'),
            ],
            axis=2,
        )
        # Build the decay
        decays = self.build_decay(
            model, intervals, 
            self.input_size, self.input_size,
            'input_decay'
        )
        # Apply mask and decay to input_features
        ONES = model.net.ConstantFill(
            [masks], 
            self.scope("ONES"), value=1.0
        )
        one_minus_masks = model.net.Sub(
            [ONES, masks],
            self.scope("one_minus_masks"), broadcast=0
        )
        one_minus_decays = model.net.Sub(
            [ONES, decays],
            self.scope("one_minus_decays"), broadcast=0
        )
        masked_inputs = model.net.Mul(
            [masks, inputs],
            self.scope("masked_inputs_1"), broadcast=0
        )  
        masked_decayed_inputs_last = model.net.Mul(
            [one_minus_masks, 
                model.net.Mul(
                    [decays, inputs_last],
                    self.scope("decayed_inputs_last"), 
                    broadcast=0
                )
            ],
            self.scope("masked_decayed_inputs_last"), 
            broadcast=0
        )
        masked_decayed_inputs_mean = model.net.Mul(
            [one_minus_masks, 
                model.net.Mul(
                    [one_minus_decays, inputs_mean],
                    self.scope("decayed_inputs_mean"), 
                    broadcast=0
                )
            ],
            self.scope("masked_decayed_inputs_mean"), 
            broadcast=0
        )    
        masked_inputs = model.net.Add(
            [masked_inputs, 
                model.net.Add(
                    [masked_decayed_inputs_last, 
                     masked_decayed_inputs_mean],
                    self.scope("sum_input_last_mean"), 
                    broadcast=0
                )],
            self.scope("masked_inputs"),   
            broadcast=0
        )
        masked_inputs_hidden = brew.fc(
            model,
            masked_inputs,
            self.scope('masked_inputs_hidden'),
            dim_in=self.input_size,
            dim_out=3 * self.hidden_size,
            axis=2,
        )
        masks_hidden = brew.fc(
            model,
            masks,
            self.scope('masks_hidden'),
            dim_in=self.input_size,
            dim_out=3 * self.hidden_size,
            axis=2,
        )
        inputs_masks_blob = model.net.Add(
            [masked_inputs_hidden, masks_hidden],
            self.scope("inputs_masks_blob"), 
            broadcast=0            
        )
        # combined_inputs, _ = model.net.Concat(
        #     [inputs_masks_blob, intervals],
        #     [self.scope('combined_inputs'),
        #      self.scope('_combined_inputs_concat_dims')],
        #     axis=2
        # )
        # return combined_inputs
        return inputs_masks_blob

    def get_state_names(self):
        return (self.scope('hidden_t'),)

# The API of MaskGRU (use _LSTM in rnn_cell)
MaskGRU = functools.partial(rnn_cell._LSTM, MaskGRUCell)
    
