"""An identity layer that prints its input."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class Print(object):
    def __init__(self, label):
        self.label = label

    @property
    def input_size(self):
        """Integer: size of inputs accepted by this cell."""
        return self._num_units

    @property
    def output_size(self):
        """Integer: size of outputs produced by this cell."""
        return self._num_units

    def __call__(self, inputs, scope=None):
        """Most basic RNN: output = new_state = tanh(W * input + U * state + B)."""
        with tf.variable_scope(scope or type(self).__name__):
            if self.label:
                print(self.label)

            print(inputs)
            self.outputs = inputs

            return self.outputs

    def zero_state(self, batch_size, dtype):
        """Return state tensor (shape [batch_size x state_size]) filled with 0.

        Args:
        batch_size: int, float, or unit Tensor representing the batch size.
        dtype: the data type to use for the state.

        Returns:
        A 2D Tensor of shape [batch_size x state_size] filled with zeros.
        """
        zeros = tf.zeros(tf.pack([batch_size, self.state_size]), dtype=dtype)
        # The reshape below is a no-op, but it allows shape inference of shape[1].
        return tf.reshape(zeros, [-1, self.state_size])
