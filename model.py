from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

def Linear(input, weight_size, name):
    return tf.matmul(input, tf.Variable(tf.truncated_normal(weight_size, name=name)))

class NTM(config):
    def __init__(self):
        self.input_dim   = config.input_dim   or error('config.input_dim must be specified')
        self.output_dim  = config.output_dim  or error('config.output_dim must be specified')
        self.mem_rows    = config.mem_rows    or 128
        self.mem_cols    = config.mem_cols    or 20
        self.cont_dim    = config.cont_dim    or 100
        self.cont_layers = config.cont_layers or 1
        self.shift_range = config.shift_range or 1
        self.write_heads = config.write_heads or 1
        self.read_heads  = config.read_heads  or 1

        self.depth = 0
        self.cells = {}
        self.master_cell = self:new_cell()
        self.init_module = self:new_init_module()

        self:init_grad_inputs()

    def build_model(self):
        dummy = tf.placeholder(tf.float32, [1])
        output_init = tf.nn.tanh(linear(dummy, [1, self.output_dim], 'output_w'))

        # memorny
        m_init_lin = linear(dummy, [1, self.mem_rows * self.mem_cols], 'm_init_w')
