from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from functools import reduce

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import array_ops

from utils import *
from ops import *

class NTMCell(object):
    def __init__(self, input_dim, output_dim,
                 mem_size=128, mem_dim=20, controller_dim=100,
                 controller_layer_size=3, shift_range=1,
                 write_head_size=1, read_head_size=1):
        """Initialize the parameters for an NTM cell.
        Args:
            input_dim: int, The number of units in the LSTM cell
            output_dim: int, The dimensionality of the inputs into the LSTM cell
            mem_size: (optional) int, The size of memory [128]
            mem_dim: (optional) int, The dimensionality for memory [20]
            controller_dim: (optional) int, The dimensionality for controller [100]
            controller_layer_size: (optional) int, The size of controller layer [1]
        """
        # initialize configs
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mem_size = mem_size
        self.mem_dim = mem_dim
        self.controller_dim = controller_dim
        self.controller_layer_size = controller_layer_size
        self.shift_range = shift_range
        self.write_head_size = write_head_size
        self.read_head_size = read_head_size

        self.depth = 0
        self.states = []

    def __call__(self, input_, state, scope=None):
        """Run one step of NTM.

        Args:
            inputs: input Tensor, 2D, 1 x input_size.
            state: state Dictionary which contains M, read_w, write_w, read,
                output, hidden.
            scope: VariableScope for the created subgraph; defaults to class name.

        Returns:
            A tuple containing:
            - A 2D, batch x output_dim, Tensor representing the output of the LSTM
                after reading "input_" when previous state was "state".
                Here output_dim is:
                     num_proj if num_proj was set,
                     num_units otherwise.
            - A 2D, batch x state_size, Tensor representing the new state of LSTM
                after reading "input_" when previous state was "state".
        """
        M_prev = state['M']
        read_w_prev = state['read_w']
        write_w_prev = state['write_w']
        read_prev = state['read']
        output_prev = state['output']
        hidden_prev = state['hidden']

        # build a controller
        output, hidden = self.build_controller(input_, read_prev, output_prev,
                                               hidden_prev)

        # last output layer from LSTM controller
        last_output = gather(output, self.controller_layer_size - 1)

        # build a memory
        M, read_w, write_w, read = self.build_memory(M_prev, read_w_prev, write_w_prev,
                                                     last_output)

        # get a new output
        new_output = self.new_output(last_output)

        state = {
            'M': M,
            'read_w': read_w,
            'write_w': write_w,
            'read': read,
            'output': output,
            'hidden': hidden,
        }

        self.depth += 1
        self.states.append(state)

        return new_output, state

    # Logistic sigmoid output layers 
    def new_output(self, output):
        with tf.variable_scope('output'):
            return tf.sigmoid(Linear(output, self.output_dim, name='output'))

    # Build LSTM controller
    def build_controller(self, input_, read_prev, output_prev, hidden_prev):
        with tf.variable_scope("controller"):
            output_list = []
            hidden_list = []
            for layer_idx in xrange(self.controller_layer_size):
                o_prev = gather(output_prev, layer_idx)
                h_prev = gather(hidden_prev, layer_idx)

                if layer_idx == 0:
                    def new_gate(gate_name):
                        return linear([input_, o_prev] + \
                                      [gather(read_prev, read_idx) for read_idx in xrange(self.read_head_size)],
                                      output_size = self.controller_dim,
                                      bias = True,
                                      scope = "%s_gate_%s" % (gate_name, layer_idx))

                        in_modules = [
                            Linear(input_, self.controller_dim,
                                   name='%s_gate_1_%s' % (gate_name, layer_idx)),
                            Linear(o_prev, self.controller_dim,
                                   name='%s_gate_2_%s' % (gate_name, layer_idx)),
                        ]
                        if self.read_head_size == 1:
                            in_modules.append(
                                Linear(read_prev, self.controller_dim,
                                       squeeze=True, name='%s_gate_3_%s' % (gate_name, layer_idx))
                            )
                        else:
                            for read_idx in xrange(self.read_head_size):
                                vec = gather(read_prev, read_idx)
                                in_modules.append(
                                    Linear(vec, self.controller_dim,
                                           name='%s_gate_3_%s_%s' \
                                                    % (gate_name, layer_idx, read_idx))
                                )
                        return tf.add_n(in_modules)
                else:
                    def new_gate(gate_name):
                        return linear([output_list[-1], o_prev],
                                      output_size = self.controller_dim,
                                      bias = True,
                                      scope="%s_gate_%s" % (gate_name, layer_idx))

                        return tf.add_n([
                            Linear(output_list[-1], self.controller_dim,
                                   name='%s_gate_1_%s' % (gate_name, layer_idx)),
                            Linear(o_prev, self.controller_dim,
                                   name='%s_gate_2_%s' % (gate_name, layer_idx)),
                        ])

                # input, forget, and output gates for LSTM
                i = tf.sigmoid(new_gate('input'))
                f = tf.sigmoid(new_gate('forget'))
                o = tf.sigmoid(new_gate('output'))
                update = tf.tanh(new_gate('update'))

                # update the sate of the LSTM cell
                hidden_list.append(tf.add_n([f * h_prev, i * update]))
                output_list.append(o * tf.tanh(hidden_list[-1]))

            output = array_ops.pack(output_list)
            hidden = array_ops.pack(hidden_list)

            return output, hidden

    def build_read_head(self, M_prev, read_w_prev, last_output, idx):
        return self.build_head(M_prev, read_w_prev, last_output, True, idx)

    def build_write_head(self, M_prev, write_w_prev, last_output, idx):
        return self.build_head(M_prev, write_w_prev, last_output, False, idx)

    def build_head(self, M_prev, w_prev, last_output, is_read, idx):
        scope = "read" if is_read else "write"

        with tf.variable_scope(scope):
            # Figure 2.
            # Amplify or attenuate the precision
            with tf.variable_scope("k"):
                k = tf.tanh(Linear(last_output, self.mem_dim, name='k_%s' % idx))
            # Interpolation gate
            with tf.variable_scope("g"):
                g = tf.sigmoid(Linear(last_output, 1, name='g_%s' % idx))
            # shift weighting
            with tf.variable_scope("s_w"):
                w = Linear(last_output, 2 * self.shift_range + 1, name='s_w_%s' % idx)
                s_w = softmax(w)
            with tf.variable_scope("beta"):
                beta  = tf.nn.softplus(Linear(last_output, 1, name='beta_%s' % idx))
            with tf.variable_scope("gamma"):
                gamma = tf.add(tf.nn.softplus(Linear(last_output, 1, name='gamma_%s' % idx)),
                               tf.constant(1.0))

            # 3.3.1
            # Cosine similarity
            similarity = smooth_cosine_similarity(M_prev, k) # [mem_size x 1]
            # Focusing by content
            content_focused_w = softmax(scalar_mul(similarity, beta))

            # 3.3.2
            # Focusing by location
            gated_w = tf.add_n([
                scalar_mul(content_focused_w, g),
                scalar_mul(w_prev, (tf.constant(1.0) - g))
            ])

            # Convolutional shifts
            conv_w = circular_convolution(gated_w, s_w)

            # Sharpening
            powed_conv_w = tf.pow(conv_w, gamma)
            w = powed_conv_w / tf.reduce_sum(powed_conv_w)

            if is_read:
                # 3.1 Reading
                read = matmul(tf.transpose(M_prev), w)
                return w, read
            else:
                # 3.2 Writing
                erase = tf.sigmoid(Linear(last_output, self.mem_dim, name='erase_%s' % idx))
                add = tf.tanh(Linear(last_output, self.mem_dim, name='add_%s' % idx))
                return w, add, erase

    # build a memory to read & write
    def build_memory(self, M_prev, read_w_prev, write_w_prev, last_output):
        with tf.variable_scope("memory"):
            # 3.1 Reading
            if self.read_head_size == 1:
                read_w, read = self.build_read_head(M_prev, tf.squeeze(read_w_prev), last_output, 0)
            else:
                read_w_list = []
                read_list = []

                for idx in xrange(self.read_head_size):
                    read_w_prev_idx = gather(read_w_prev, idx)
                    read_w_idx, read_idx = self.build_read_head(M_prev, read_w_prev_idx,
                                                                last_output, idx)

                    read_w_list.append(read_w_idx)
                    read_list.append(read_idx)

                read_w = array_ops.pack(read_w_list)
                read = array_ops.pack(read_list)

            # 3.2 Writing
            if self.write_head_size == 1:
                write_w, write, erase = self.build_write_head(M_prev,
                                                              tf.squeeze(write_w_prev),
                                                              last_output, 0)

                M_erase = tf.ones([self.mem_size, self.mem_dim]) \
                              - outer_product(write_w, erase)
                M_write = outer_product(write_w, write)
            else:
                write_w_list = []
                write_list = []
                erase_list = []

                M_erases = []
                M_writes = []

                for idx in xrange(self.write_head_size):
                    write_w_prev_idx = gather(write_w_prev, idx)

                    write_w_idx, write_idx, erase_idx = \
                        self.build_write_head(M_prev, write_w_prev_idx, last_output, idx)

                    write_w_list.append(tf.transpose(write_w_idx))
                    write_list.append(write_idx)
                    erase_list.append(erase_idx)

                    M_erases.append(tf.ones([self.mem_size, self.mem_dim]) \
                                    * outer_product(write_w_idx, erase_idx))
                    M_writes.append(outer_product(write_w_idx, write_idx))

                write_w = array_ops.pack(write_w_list)
                write = array_ops.pack(write_list)
                erase = array_ops.pack(erase_list)

                M_erase = reduce(lambda x, y: x*y, M_erases)
                M_write = tf.add_n(M_writes)

            M = M_prev * M_erase + M_write

            return M, read_w, write_w, read

    def initial_state(self, dummy_value=0.0):
        self.depth = 0
        self.states = []
        with tf.variable_scope("init_cell"):
            # always zero
            dummy = tf.Variable(tf.constant([[dummy_value]], dtype=tf.float32))

            # memory
            M_init_linear = tf.tanh(Linear(dummy, self.mem_size * self.mem_dim,
                                    name='M_init_linear'))
            M_init = tf.reshape(M_init_linear, [self.mem_size, self.mem_dim])

            # read weights
            read_w_init_list = []
            read_init_list = []
            for idx in xrange(self.read_head_size):
                read_w_idx = Linear(dummy, self.mem_size, is_range=True, 
                                    name='read_w_%d' % idx)
                read_w_init_list.append(softmax(read_w_idx))

                read_init_idx = Linear(dummy, self.mem_dim,
                                       squeeze=True, name='read_init_%d' % idx)
                read_init_list.append(tf.tanh(read_init_idx))

            read_w_init = tf.reshape(array_ops.pack(read_w_init_list),
                                     [self.read_head_size, -1])
            read_init = array_ops.pack(read_init_list)

            # write weights
            write_w_init_list = []
            for idx in xrange(self.write_head_size):
                write_w_idx = Linear(dummy, self.mem_size, is_range=True,
                                     name='write_w_%s' % idx)
                write_w_init_list.append(softmax(write_w_idx))

            write_w_init = tf.reshape(array_ops.pack(write_w_init_list),
                                      [self.write_head_size, -1])

            # controller state
            output_init_list = []                     
            hidden_init_list = []                     
            for idx in xrange(self.controller_layer_size):
                output_init_idx = Linear(dummy, self.controller_dim,
                                         squeeze=True, name='output_init_%s' % idx)
                output_init_list.append(tf.tanh(output_init_idx))
                hidden_init_idx = Linear(dummy, self.controller_dim,
                                         squeeze=True, name='hidden_init_%s' % idx)
                hidden_init_list.append(tf.tanh(hidden_init_idx))

            output_init = array_ops.pack(output_init_list)
            hidden_init = array_ops.pack(hidden_init_list)

            output = tf.tanh(Linear(dummy, self.output_dim, name='new_output'))

            state = {
                'M': M_init,
                'read_w': read_w_init,
                'write_w': write_w_init,
                'read': tf.reshape(read_init, [self.read_head_size, self.mem_dim]),
                'output': output_init,
                'hidden': hidden_init
            }

            self.depth += 1
            self.states.append(state)

            return output, state

    def get_memory(self, depth=None):
        depth = depth if depth else self.depth
        return self.states[depth - 1]['M']

    def get_read_weights(self, depth=None):
        depth = depth if depth else self.depth
        return self.states[depth - 1]['read_w']

    def get_write_weights(self, depth=None):
        depth = depth if depth else self.depth
        return self.states[depth - 1]['write_w']

    def get_read_vector(self, depth=None):
        depth = depth if depth else self.depth
        return self.states[depth - 1]['read']

    def print_read_max(self):
        read_w = self.get_read_weights()

        fmt = "%-4d %.4f"
        if self.read_head_size == 1:
            print(fmt % (argmax(read_w[0])))
        else:
            for idx in xrange(self.read_head_size):
                print(fmt % np.argmax(read_w[idx]))

    def print_write_max(self):
        write_w = self.get_write_weights()

        fmt = "%-4d %.4f"
        if self.write_head_size == 1:
            print(fmt % (argmax(write_w[0])))
        else:
            for idx in xrange(self.write_head_size):
                print(fmt % argmax(gather(write_w, idx)))
