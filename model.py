from __future__ import division, print_function, absolute_import
from functools import reduce

import numpy as np
import tensorflow as tf

from utils import *
from layers import *

class NTM(object):
    def __init__(self, config=None):
        self.input_dim = 128
        self.output_dim = 128
        self.mem_size = 128
        self.mem_dim = 20
        self.controller_dim = 100
        self.controller_layer_size = 2
        self.shift_range = 1
        self.write_head_size = 1
        self.read_head_size = 1

        #self.input_dim = config.input_dim or 128
        #self.output_dim = config.output_dim or 128
        #self.mem_size = config.mem_size or 128
        #self.mem_dim = config.mem_dim or 20
        #self.controller_dim = config.controller_dim    or 100
        #self.controller_layer_size = config.controller_layer_size or 1
        #self.shift_range = config.shift_range or 1
        #self.write_head_size = config.write_head_size or 1
        #self.read_heads_size = config.read_heads_size  or 1

        self.depth = 0
        self.cells = {}
        self.master_cell = self.build_cell() # [input_sets, output_sets]
        self.init_module = self.new_init_module()

    def build_init_module(self):
        dummy = tf.placeholder(tf.float32, [None, 1])
        output_init = tf.tanh(Linear(dummy, output_dim, bias=True, bias_init=1))

        # memory
        m_init = tf.reshape(tf.tanh(Linear(dummy, mem_rows * mem_cols, bias=True)),
                            [mem_rows, mem_cols])

        # read weights
        write_init, read_init = [], []
        for idx in xrange(read_head_size):
            write_w = tf.Variable(tf.random_normal([1, mem_rows]))
            write_b = tf.Variable(tf.cast(tf.range(mem_rows-2, 0, -1), dtype=tf.float32))
            write_init_lin = tf.nn.bias_add(tf.matmul(dummy, write_w), write_b)

            write_init.append(tf.nn.softmax(write_init_lin))
            #write_init.append(tf.nn.softmax(Linear(dummy, mem_rows, name='write_lin')))

            read_init.append(tf.nn.softmax(Linear(dummy, mem_cols, name='read_lin')))

        # write weights
        ww_init = []
        for idx in xrange(write_head_size):
            ww_w = tf.Variable(tf.random_normal([1, mem_rows]))
            ww_b = tf.Variable(tf.cast(tf.range(mem_rows-2, 0, -1), dtype=tf.float32))
            ww_init_lin = tf.nn.bias_add(tf.matmul(dummy, ww_w), ww_b)

            ww_init.append(tf.nn.softmax(ww_init_lin))

        # controller state
        m_init, c_init = [], []
        for idx in xrange(controller_layer_size):
            m_init.append(tf.tanh(Linear(dummy, controller_dim)))
            c_init.append(tf.tanh(Linear(dummy, controller_dim)))

        ww = tf.placeholder(tf.float32, [])
        wr = tf.placeholder(tf.float32, [])
        r = tf.placeholder(tf.float32, [])
        m = tf.placeholder(tf.float32, [])
        c = tf.placeholder(tf.float32, [])

    # Build a NTM cell which shares weights with "master" cell.
    def build_cell(self):
        input = tf.placeholder(tf.float32, [1, self.input_dim])

        # previous memory state
        M_prev = tf.placeholder(tf.float32, [self.mem_size, self.mem_dim])

        # previous read/write weights
        read_w_prev = tf.placeholder(tf.float32, [self.read_head_size, self.mem_size])
        write_w_prev = tf.placeholder(tf.float32, [self.write_head_size, self.mem_size])

        # previous vector read from memory
        read_prev = tf.placeholder(tf.float32, [1, self.mem_dim])

        # previous LSTM controller output
        output_prev = tf.placeholder(tf.float32, [self.controller_layer_size, self.output_dim])
        hidden_prev = tf.placeholder(tf.float32, [self.controller_layer_size, self.controller_dim])

        # output and hidden states of controller module
        output, hidden = self.build_controller(input, read_prev, output_prev, hidden_prev)

        # last output layer from LSTM controller
        last_output = output if self.controller_layer_size == 1 else output[-1]

        # Build a memory
        M, read_w, write_w, read = self.build_memory(M_prev, read_w_prev, write_w_prev, last_output)
        output = self.build_output(last_output)

        self.inputs = [input, M_prev, read_w_prev, write_w_prev, read_prev, output_prev, hidden_prev]
        self.outputs = [output, M, read_w, write_w, read, output, hidden]

        return [self.inputs, self.outputs]

    def build_read_head(self, M_prev, read_w_prev, last_output):
        return self.build_head(M_prev, read_w_prev, last_output, True)

    def build_write_head(self, M_prev, write_w_prev, last_output):
        return self.build_head(M_prev, write_w_prev, last_output, False)

    def build_output(self, output):
        return tf.sigmoid(Linear(output, self.output_dim))

    def build_head(self, M_prev, w_prev, last_output, is_read):
        # Figure 2.
        # Amplify or attenuate the precision
        k = tf.tanh(Linear(last_output, self.mem_dim))
        # Interpolation gate
        g = tf.sigmoid(Linear(last_output, 1))
        # shift weighting
        s_w = tf.reshape(tf.nn.softmax(Linear(last_output, 2 * self.shift_range + 1)), [-1, 1])
        beta  = tf.nn.softplus(Linear(last_output, 1))
        gamma = tf.add(tf.nn.softplus(Linear(last_output, 1)), tf.constant(1.0))

        # 3.3.1
        # Cosine similarity
        similarity = SmoothCosineSimilarity(M_prev, k) # [mem_size x 1]
        # Focusing by content
        content_focused_w = tf.nn.softmax(ScalarMul(similarity, beta))

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
            read = M_prev * w
            return w, read
        else:
            # 3.2 Writing
            erase = tf.sigmoid(Linear(last_output, self.mem_dim)) # [1 x mem_dim]
            add = tf.tanh(Linear(last_output, self.mem_dim))
            return w, add, erase

    # build a memory to read & write
    def build_memory(self, M_prev, read_w_prev, write_w_prev, last_output):
        # 3.1 Reading
        if self.read_head_size == 1:
            read_w, read = self.build_read_head(M_prev, tf.reshape(read_w_prev, [-1, 1]), last_output)
        else:
            read_w = tf.Variable(tf.zeros([self.read_head_size, self.mem_size]))
            read = tf.Variable(tf.zeros([self.read_head_size, self.mem_size, self.mem_dim]))

            for idx in xrange(self.read_head_size):
                read_w_prev_idx = tf.reshape(tf.gather(read_w_prev, idx), [-1, 1])

                read_w_idx, read_idx = self.build_read_head(M_prev, read_w_prev_idx, last_output)

                read_w = tf.scatter_update(read_w, [idx], tf.transpose(read_w_idx))
                read = tf.scatter_update(read, [idx], tf.reshape(read_idx, [1, self.mem_size, self.mem_dim]))

        # 3.2 Writing
        if self.write_head_size == 1:
            write_w, write, erase = self.build_write_head(M_prev, tf.reshape(write_w_prev, [-1, 1]), last_output)

            M_erase = tf.ones([self.mem_size, self.mem_dim]) - OuterProd(write_w, erase)
            M_write = OuterProd(write_w, write)
        else:
            write_w = tf.Variable(tf.zeros([self.write_head_size, self.mem_size]))
            write = tf.Variable(tf.zeros([self.write_head_size, self.mem_size, self.mem_dim]))
            erase = tf.Variable(tf.zeros([self.write_head_size, 1, self.mem_dim]))

            M_erases = []
            M_writes = []

            for idx in xrange(self.write_head_size):
                write_w_prev_idx = tf.reshape(tf.gather(write_w_prev, idx), [-1, 1])

                write_w_idx, write_idx, erase_idx = self.build_write_head(M_prev, write_w_prev_idx, last_output)

                write_w = tf.scatter_update(write_w, [idx], tf.transpose(write_w_idx))
                write = tf.scatter_update(write, [idx], tf.reshape(write_idx, [1, self.mem_size, self.mem_dim]))
                erase = tf.scatter_update(erase, [idx], tf.reshape(erase_idx, [1, 1, self.mem_dim]))

                M_erases.append(tf.ones([self.mem_size, self.mem_dim]) * OuterProd(write_w_idx, erase_idx))
                M_writes.append(OuterProd(write_w_idx, write_idx))

            M_erase = reduce(lambda x, y: x*y, M_erases)
            M_write = tf.add_n(M_writes)

        M = M_prev * M_erase + M_write

        return M, read_w, write_w, read

    # Build a LSTM controller
    def build_controller(self, input, read_prev, output_prev, hidden_prev):
        output, hidden = [], []
        for layer_idx in xrange(self.controller_layer_size):
            if self.controller_layer_size == 1:
                o_prev = output_prev
                h_prev = hidden_prev
            else:
                o_prev = tf.reshape(tf.gather(output_prev, layer_idx), [1, -1])
                h_prev = tf.reshape(tf.gather(hidden_prev, layer_idx), [1, -1])

            if layer_idx == 0:
                def new_gate():
                    in_modules = [
                        Linear(input, self.controller_dim),
                        Linear(o_prev, self.controller_dim),
                    ]
                    if self.read_head_size == 1:
                        in_modules.append(
                            Linear(read_prev, self.controller_dim)
                        )
                    else:
                        for read_idx in xrange(self.read_head_size):
                            vec = tf.reshape(tf.gather(read_prev, read_idx), [1, -1])
                            in_modules.append(
                                Linear(vec, self.controller_dim)
                            )
                    return tf.add_n(in_modules)
            else:
                def new_gate():
                    return tf.add_n([
                        Linear(output[layer_idx-1], self.controller_dim),
                        Linear(o_prev, self.controller_dim),
                    ])

            # input, forget, and output gates for LSTM
            i = tf.sigmoid(new_gate())
            f = tf.sigmoid(new_gate())
            o = tf.sigmoid(new_gate())
            update = tf.tanh(new_gate())

            # update the sate of the LSTM cell
            hidden.append(tf.add_n([f * h_prev, i * update]))
            output.append(o * tf.tanh(hidden[layer_idx]))

        return output, hidden

    def forward(self, input):
        self.depth += 1

        try:
            cell = self.cells[self.depth]
        except:
            cell = self.new_cell()
            self.cells.append(cell)
        
        if self.depth == 1:
            prev_outputs = self.init_module(tf.constant(0.0))
        else:
            prev_outputs = self.cells[self.depth - 1]#.output

        inputs = [input]
        for idx in xrange(1, len(prev_outputs)):
            inputs.append(prev_outputs[idx])

        outputs = cell.forward(inputs)
        self.output = outputs[0]

        return self.output
