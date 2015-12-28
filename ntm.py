from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.seq2seq import sequence_loss

import ntm_cell
from ops import binary_cross_entropy_with_logits

class NTM(object):
    def __init__(self, cell, 
                 min_grad=-10, max_grad=+10, 
                 lr=1e-4, scope="NTM"):
        """Create a neural turing machine specified by NTMCell "cell".

        Args:
            cell: An instantce of NTMCell.
            min_grad: (optional) Minimum gradient for gradient clipping [-10].
            max_grad: (optional) Maximum gradient for gradient clipping [+10].
            lr: (optional) Learning rate [1e-4].
            scope: VariableScope for the created subgraph ["NTM"].
        """
        if not isinstance(cell, ntm_cell.NTMCell):
            raise TypeError("cell must be an instance of NTMCell")
        self.cell = cell
        self.scope = scope
        self.reuse = False

        self.min_grad = min_grad
        self.max_grad = min_grad
        self.current_lr = lr
        self.lr = tf.Variable(self.current_lr, trainable=False)
        
        self.losses = {}
        self.optims = {}
	self.saver = tf.train.Saver()

    def forward(self, inputs, logging=False):
        if not inputs:
            raise ValueError("inputs must not be empty")

        with tf.variable_scope(self.scope):
            if self.reuse:
                tf.get_variable_scope().reuse_variables()

            output, state = self.cell.initial_state()

            # write head max
            for time, input_ in enumerate(inputs):
                if not self.reuse and time > 0:
                    tf.get_variable_scope().reuse_variables()
                _, state = self.cell(input_, state)

            # read head max
            loss = 0
            input_dim = inputs[0].shape[0]
            zeros = np.zeros(input_dim, dtype=np.float32)

            self.outputs = []
            for _ in inputs:
                output, state = self.cell(zeros, state)
                self.outputs.append(output)

            if not self.losses.has_key(self.cell.depth):
                loss = sequence_loss(logits = self.outputs,
                                     targets = inputs,
                                     weights = [tf.ones([input_dim])] * len(inputs),
                                     num_decoder_symbols = -1, # trash
                                     average_across_timesteps = True,
                                     average_across_batch = False,
                                     softmax_loss_function = \
                                         binary_cross_entropy_with_logits)

                tvars = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
                #grads = [tf.clip_by_value(grad, self.min_grad, self.max_grad) \
                #             for grad in tf.gradients(loss, tvars)]
                optimizer = tf.train.GradientDescentOptimizer(self.lr)

                self.losses[self.cell.depth] = loss 
                self.optims[self.cell.depth] = optimizer.apply_gradients(zip(grads, tvars))

            self.reuse = True

        return self.outputs, loss

    def print_write_max(self):
        cell.print_write_max()

    @property
    def loss(self):
        return self.losses[self.cell.depth]

    @property
    def optim(self):
        return self.optims[self.cell.depth]
