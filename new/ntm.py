from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from functools import reduce

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import array_ops

import ntm_cell
from utils import *
from ops import *

class NTM(object):
    def __init__(self, cell, scope=None):
        if not isinstance(cell, ntm_cell.NTMCell):
            raise TypeError("cell must be an instance of NTMCell")
        self.cell = cell
        self.scope = scope

    def forward(self, inputs):
        if not inputs:
            raise ValueError("inputs must not be empty")

        self.outputs = []
        self.states = []
        with tf.variable_scope(self.scope or "NTM"):
            output, state = self.cell.initial_state()

            for time, input_ in enumerate(inputs):
                if time > 0:
                    tf.get_variable_scope().reuse_variables()
                output, state = self.cell(input_, state)

                self.outputs.append(output)
                self.states.append(state)

        return self.outputs
