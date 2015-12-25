from __future__ import division
from __futere__ import print_function
from __future__ import absolute_import
from functools import reduce

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import array_ops

from utils import *
from layers import *

class NTMCell(object):
    def __init__(self, input_dim, output_dim,
                 mem_size=128, mem_dim=20, controller_dim=100,
                 controller_layer_size=1, shift_range=1,
                 write_head_size=1, read_head_size=1,
                 lr_rate=1e-4):
        """Initialize the parameters for an NTM cell.
        Args:
            input_dim: int, The number of units in the LSTM cell
            output_dim: int, The dimensionality of the inputs into the LSTM cell
            mem_dim: (optional) int, set True to enable diagonal/peephole connections.
            cell_clip: (optional) A float value, if provided the cell state is clipped
                by this value prior to the cell output activation.
            initializer: (optional) The initializer to use for the weight and
                projection matrices.
            num_proj: (optional) int, The output dimensionality for the projection
                matrices.  If None, no projection is performed.
            num_unit_shards: How to split the weight matrix.  If >1, the weight
                matrix is stored across num_unit_shards.
            num_proj_shards: How to split the projection matrix.  If >1, the
                projection matrix is stored across num_proj_shards.
        """
        # initialize configs
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.mem_size = config.mem_size
        self.mem_dim = config.mem_dim
        self.controller_dim = config.controller_dim
        self.controller_layer_size = config.controller_layer_size
        self.shift_range = config.shift_range
        self.write_head_size = config.write_head_size
        self.read_head_size = config.read_head_size
        self.current_lr = config.lr_rate

        self.depth = 0
        self.input_cells = []
        self.output_cells = []
        self.prev_outputs = []

        self.min_grad = -10
        self.max_grad = +10

        # training options
        self.lr = tf.Variable(self.current_lr)
        self.opt = tf.train.GradientDescentOptimizer(self.lr)

        self.global_step = tf.Variable(0, name="global_step")
        inc = self.global_step.assign_add(1)

        self.saver = tf.train.Saver()

    def build_init_cell(self):
        with tf.variable_scope("init_cell"):
            # always zero
            dummy = tf.placeholder(tf.float32, [1, 1], name='dummy')

            # memory
            M_init_linear = tf.tanh(Linear(dummy, self.mem_size * self.mem_dim, name='M_init_linear'))
            M_init = tf.reshape(M_init_linear, [self.mem_size, self.mem_dim])

            # read weights
            read_w_init = tf.Variable(tf.zeros([self.read_head_size, self.mem_size]))
            read_init = tf.Variable(tf.zeros([self.read_head_size, 1, self.mem_dim]))

            for idx in xrange(self.read_head_size):
                # initialize bias distribution with `tf.range(mem_size-2, 0, -1)`
                read_w_linear_idx = Linear(dummy, self.mem_size, is_range=True,
                                           name='read_w_linear_%s' % idx)
                read_w_init = tf.scatter_update(read_w_init, [idx], tf.nn.softmax(read_w_linear_idx))

                read_init_idx = tf.tanh(Linear(dummy, self.mem_dim, name='read_init_%s' % idx))
                read_init = tf.scatter_update(read_init, [idx], tf.reshape(read_init_idx, [1, 1, self.mem_dim]))

            # write weights
            write_w_init = tf.Variable(tf.zeros([self.write_head_size, self.mem_size]))
            for idx in xrange(self.write_head_size):
                write_w_linear_idx = Linear(dummy, self.mem_size, is_range=True,
                                            name='write_w_linear_%s' % idx)
                write_w_init = tf.scatter_update(write_w_init, [idx], tf.nn.softmax(write_w_linear_idx))

            # controller state
            output_init = tf.Variable(tf.zeros([self.controller_layer_size, self.controller_dim]))
            hidden_init = tf.Variable(tf.zeros([self.controller_layer_size, self.controller_dim]))

            for idx in xrange(self.controller_layer_size):
                output_init = tf.scatter_update(output_init, [idx], tf.reshape(
                        tf.tanh(Linear(dummy, self.controller_dim, name='output_init_%s' % idx)),
                        [1, self.controller_dim]
                    )
                )
                hidden_init = tf.scatter_update(hidden_init, [idx], tf.reshape(
                        tf.tanh(Linear(dummy, self.controller_dim, name='hidden_init_%s' % idx)),
                        [1, self.controller_dim]
                    )
                )

            new_output= tf.tanh(Linear(dummy, self.output_dim, name='new_output'))

            inputs = {
                'input': dummy,
            }
            outputs = {
                'new_output': new_output,
                'M': M_init,
                'read_w': read_w_init,
                'write_w': write_w_init,
                'read': tf.reshape(read_init, [self.read_head_size, self.mem_dim]),
                'output': output_init,
                'hidden': hidden_init
            }
            return inputs, outputs

    # Build a NTM cell which shares weights with "master" cell.
    def build_cell(self):
        input = tf.placeholder(tf.float32, [1, self.input_dim], name='build_cell_input')

        # previous memory state
        M_prev = tf.placeholder(tf.float32, [self.mem_size, self.mem_dim], name='build_cell_M_prev')

        # previous read/write weights
        read_w_prev = tf.placeholder(tf.float32, [self.read_head_size, self.mem_size], name='read_w_prev')
        write_w_prev = tf.placeholder(tf.float32, [self.write_head_size, self.mem_size], name='write_w_prev')

        # previous vector read from memory
        read_prev = tf.placeholder(tf.float32, [self.read_head_size, self.mem_dim], name='read_prev')

        # previous LSTM controller output
        output_prev = tf.placeholder(tf.float32, [self.controller_layer_size, self.controller_dim], name='output_prev')
        hidden_prev = tf.placeholder(tf.float32, [self.controller_layer_size, self.controller_dim], name='hidden_prev')

        # output and hidden states of controller module
        output, hidden = self.build_controller(input, read_prev, output_prev, hidden_prev)

        # last output layer from LSTM controller
        last_output = output if self.controller_layer_size == 1 \
                             else tf.reshape(tf.gather(output, self.controller_layer_size - 1), [1, -1])

        # build a memory
        M, read_w, write_w, read = self.build_memory(M_prev, read_w_prev, write_w_prev, last_output)
        new_output = self.build_output(last_output)

        # placeholder for true output
        true_output = tf.placeholder(tf.float32, [self.output_dim])
        loss = tf.nn.softmax_cross_entropy_with_logits(true_output, new_output)

        # gradient clipping
        grads_and_vars = self.opt.compute_gradients(loss)
        clipped_grads_and_vars = [(tf.clip_by_value(gv[0], self.min_grad, self.max_grad), gv[1]) \
                                   for gv in grads_and_vars]

        with tf.control_dependencies([inc]):
            optim = self.opt.apply_gradients(clipped_grads_and_vars)

        inputs = {
            'input': input,
            'M_prev': M_prev,
            'read_w_prev': read_w_prev,
            'write_w_prev': write_w_prev,
            'read_prev': read_prev,
            'output_prev': output_prev,
            'hidden_prev': hidden_prev,
            'true_output': true_output,
        }
        outputs = {
            'new_output': new_output,
            'M': M,
            'read_w': read_w,
            'write_w': write_w,
            'read': read,
            'output': output,
            'hidden': hidden,
            'loss': loss,
            'optim': optim,
        }
        return inputs, outputs

    def build_output(self, output):
        with tf.variable_scope('output'):
            return tf.sigmoid(Linear(output, self.output_dim, name='output'))

    def build_read_head(self, M_prev, read_w_prev, last_output, idx):
        return self.build_head(M_prev, read_w_prev, last_output, True, idx)

    def build_write_head(self, M_prev, write_w_prev, last_output, idx):
        return self.build_head(M_prev, write_w_prev, last_output, False, idx)

    def build_head(self, M_prev, w_prev, last_output, is_read, idx):
        scope = 'read' if is_read else 'write'

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
                s_w = tf.reshape(tf.nn.softmax(w), [-1, 1])
            with tf.variable_scope("beta"):
                beta  = tf.nn.softplus(Linear(last_output, 1, name='beta_%s' % idx))
            with tf.variable_scope("gamma"):
                gamma = tf.add(tf.nn.softplus(Linear(last_output, 1, name='gamma_%s' % idx)), tf.constant(1.0))

            # 3.3.1
            # Cosine similarity
            similarity = SmoothCosineSimilarity(M_prev, k) # [mem_size x 1]
            # Focusing by content
            content_focused_w_lin = tf.nn.softmax(tf.reshape(ScalarMul(similarity, beta), [1, self.mem_size]))
            content_focused_w = tf.reshape(content_focused_w_lin, [self.mem_size, 1])

            # 3.3.2
            # Focusing by location
            gated_w = tf.add_n([
                ScalarMul(content_focused_w, g),
                ScalarMul((tf.constant(1.0) - g), w_prev)
            ])

            # Convolutional shifts
            conv_w = CircularConvolution(vector=gated_w, kernel=s_w)

            # Sharpening
            powed_conv_w = tf.pow(conv_w, gamma)
            w = powed_conv_w / tf.reduce_sum(powed_conv_w)

            if is_read:
                # 3.1 Reading
                read = tf.batch_matmul(
                    tf.reshape(M_prev, [1, self.mem_size, self.mem_dim]),
                    tf.reshape(w, [1, self.mem_size, 1]) , adj_x=True)
                return w, tf.reshape(read, [self.mem_dim, 1])
            else:
                # 3.2 Writing
                erase = tf.sigmoid(Linear(last_output, self.mem_dim, name='erase_%s' % idx)) # [1 x mem_dim]
                add = tf.tanh(Linear(last_output, self.mem_dim, name='add_%s' % idx))
                return w, add, erase

    # build a memory to read & write
    def build_memory(self, M_prev, read_w_prev, write_w_prev, last_output):
        with tf.variable_scope("memory"):
            # 3.1 Reading
            if self.read_head_size == 1:
                read_w, read = self.build_read_head(M_prev, tf.reshape(read_w_prev, [-1, 1]), last_output, 0)
            else:
                read_w_list = []
                read_list = []

                for idx in xrange(self.read_head_size):
                    read_w_prev_idx = tf.reshape(tf.gather(read_w_prev, idx), [-1, 1])

                    read_w_idx, read_idx = self.build_read_head(M_prev, read_w_prev_idx, last_output, idx)

                    read_w_list.append(tf.transpose(read_w_idx))
                    read_list.append(tf.reshape(read_idx, [1, self.mem_size, self.mem_dim]))

                read_w = array_ops.pack(read_w_list)
                read = array_ops.pack(read_list)

            # 3.2 Writing
            if self.write_head_size == 1:
                write_w, write, erase = self.build_write_head(M_prev, tf.reshape(write_w_prev, [-1, 1]),
                                                              last_output, 0)

                M_erase = tf.ones([self.mem_size, self.mem_dim]) - OuterProd(write_w, erase)
                M_write = OuterProd(write_w, write)
            else:
                write_w_list = []
                write_list = []
                erase_list = []

                M_erases = []
                M_writes = []

                for idx in xrange(self.write_head_size):
                    write_w_prev_idx = tf.reshape(tf.gather(write_w_prev, idx), [-1, 1])

                    write_w_idx, write_idx, erase_idx = self.build_write_head(M_prev, write_w_prev_idx,
                                                                              last_output, idx)

                    write_w_list.append(tf.transpose(write_w_idx))
                    write_list.append(tf.reshape(write_idx, [1, self.mem_size, self.mem_dim]))
                    erase_list.append(tf.reshape(erase_idx, [1, 1, self.mem_dim]))

                    M_erases.append(tf.ones([self.mem_size, self.mem_dim]) * OuterProd(write_w_idx, erase_idx))
                    M_writes.append(OuterProd(write_w_idx, write_idx))

                write_w = array_ops.pack(write_w_list)
                write = array_ops.pack(write_list)
                erase = array_ops.pack(erase_list)

                M_erase = reduce(lambda x, y: x*y, M_erases)
                M_write = tf.add_n(M_writes)

            M = M_prev * M_erase + M_write

            return M, read_w, write_w, read

    # Build a LSTM controller
    def build_controller(self, input, read_prev, output_prev, hidden_prev):
        with tf.variable_scope("controller"):
            output = tf.Variable(tf.zeros([self.controller_layer_size, self.controller_dim]))
            hidden = tf.Variable(tf.zeros([self.controller_layer_size, self.controller_dim]))
            for layer_idx in xrange(self.controller_layer_size):
                if self.controller_layer_size == 1:
                    o_prev = output_prev
                    h_prev = hidden_prev
                else:
                    o_prev = tf.reshape(tf.gather(output_prev, layer_idx), [1, -1])
                    h_prev = tf.reshape(tf.gather(hidden_prev, layer_idx), [1, -1])

                if layer_idx == 0:
                    def new_gate(gate_name):
                        in_modules = [
                            Linear(input, self.controller_dim,
                                   name='%s_gate_1_%s' % (gate_name, layer_idx)),
                            Linear(o_prev, self.controller_dim,
                                   name='%s_gate_2_%s' % (gate_name, layer_idx)),
                        ]
                        if self.read_head_size == 1:
                            in_modules.append(
                                Linear(read_prev, self.controller_dim,
                                       name='%s_gate_3_%s' % (gate_name, layer_idx))
                            )
                        else:
                            for read_idx in xrange(self.read_head_size):
                                vec = tf.reshape(tf.gather(read_prev, read_idx), [1, -1])
                                in_modules.append(
                                    Linear(vec, self.controller_dim,
                                           name='%s_gate_3_%s_%s' % (gate_name, layer_idx, read_idx))
                                )
                        return tf.add_n(in_modules)
                else:
                    def new_gate(gate_name):
                        return tf.add_n([
                            Linear(tf.reshape(tf.gather(output, layer_idx-1), [1, -1]),
                                   self.controller_dim, name='%s_gate_1_%s' % (gate_name, layer_idx)),
                            Linear(o_prev, self.controller_dim,
                                   name='%s_gate_2_%s' % (gate_name, layer_idx)),
                        ])

                # input, forget, and output gates for LSTM
                i = tf.sigmoid(new_gate('input'))
                f = tf.sigmoid(new_gate('forget'))
                o = tf.sigmoid(new_gate('output'))
                update = tf.tanh(new_gate('update'))

                # update the sate of the LSTM cell
                hidden = tf.scatter_update(hidden, [layer_idx],
                                           tf.add_n([f * h_prev, i * update]))
                output = tf.scatter_update(output, [layer_idx],
                                           o * tf.tanh(tf.gather(hidden,layer_idx)))

            return output, hidden

    def build_model(self):
        self.init_input_cell, self.init_output_cell = self.build_init_cell()

        with tf.variable_scope("cell") as scope:
            self.cell_scope = scope
            self.master_input_cell, self.master_output_cell = self.build_cell()

        print(" [*] Initialization start...")
        self.sess.run(tf.initialize_all_variables())
        print(" [*] Initialization end")

    def add_new_cell(self):
        try:
            cur_input_cell = self.input_cells[self.depth]
            cur_output_cell = self.output_cells[self.depth]
        except:
            with tf.variable_scope(self.cell_scope, reuse=True):
                cur_input_cell, cur_output_cell = self.build_cell()
                self.sess.run(tf.initialize_all_variables())
                self.input_cells.append(cur_input_cell)
                self.output_cells.append(cur_output_cell)

        return cur_input_cell, cur_output_cell

    def backward(self, true_output):
        cur_input_cell = self.master_input_cell
        cur_output_cell = self.master_output_cell
        #cur_input_cell, cur_output_cell = self.add_new_cell()

        outputs = self.sess.run([
                cur_output_cell['new_output'],
                cur_output_cell['M'], cur_output_cell['read_w'], cur_output_cell['write_w'],
                cur_output_cell['read'], cur_output_cell['output'], cur_output_cell['hidden'],
            ], feed_dict = {
                cur_input_cell['input']: input,
                cur_input_cell['M_prev']: prev_outputs[1],
                cur_input_cell['read_w_prev']: prev_outputs[2],
                cur_input_cell['write_w_prev']: prev_outputs[3],
                cur_input_cell['read_prev']: prev_outputs[4],
                cur_input_cell['output_prev']: prev_outputs[5],
                cur_input_cell['hidden_prev']: prev_outputs[6],
            }
        )
        
        return loss

    def forward(self, input):
        if self.depth == 0:
            prev_outputs = self.sess.run([
                    self.init_output_cell['new_output'],
                    self.init_output_cell['M'], self.init_output_cell['read_w'],
                    self.init_output_cell['write_w'], self.init_output_cell['read'],
                    self.init_output_cell['output'], self.init_output_cell['hidden']
                ], feed_dict={
                    self.init_input_cell['input']: [[0.0]]
                }
            )
            self.initial_values = prev_outputs
        else:
            prev_outputs = self.prev_outputs[self.depth - 1]

        cur_input_cell = self.master_input_cell
        cur_output_cell = self.master_output_cell
        #cur_input_cell, cur_output_cell = self.add_new_cell()

        outputs = self.sess.run([
                cur_output_cell['new_output'],
                cur_output_cell['M'], cur_output_cell['read_w'], cur_output_cell['write_w'],
                cur_output_cell['read'], cur_output_cell['output'], cur_output_cell['hidden'],
            ], feed_dict = {
                cur_input_cell['input']: input,
                cur_input_cell['M_prev']: prev_outputs[1],
                cur_input_cell['read_w_prev']: prev_outputs[2],
                cur_input_cell['write_w_prev']: prev_outputs[3],
                cur_input_cell['read_prev']: prev_outputs[4],
                cur_input_cell['output_prev']: prev_outputs[5],
                cur_input_cell['hidden_prev']: prev_outputs[6],
            }
        )
        outputs[2] = outputs[2].T
        outputs[3] = outputs[3].T
        outputs[4] = outputs[4].T
        self.output = outputs[0]
        self.prev_outputs.append(outputs)

        self.depth += 1
        
        return self.output

    def get_memory(self, depth=None):
        if self.depth == 0:
            return prev_outputs[1]
        depth = depth if depth else self.depth
        return self.prev_outputs[depth - 1][1]

    def get_read_weights(self, depth=None):
        if self.depth == 0:
            return prev_outputs[2]
        depth = depth if depth else self.depth
        return self.prev_outputs[depth - 1][2]

    def get_write_weights(self, depth=None):
        if self.depth == 0:
            return prev_outputs[3]
        depth = depth if depth else self.depth
        return self.prev_outputs[depth - 1][3]

    def get_read_vector(self, depth=None):
        if self.depth == 0:
            return prev_outputs[4]
        depth = depth if depth else self.depth
        return self.prev_outputs[depth - 1][4]

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
                print(fmt % np.argmax(write_w[idx]))
