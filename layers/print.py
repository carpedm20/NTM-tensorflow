"""An identity layer that prints its input."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.models.rnn import linear


class RNNCell(object):
  def __call__(self, inputs, state, scope=None):
    """Run this RNN cell on inputs, starting from the given state.

    Args:
      inputs: 2D Tensor with shape [batch_size x self.input_size].
      state: 2D Tensor with shape [batch_size x self.state_size].
      scope: VariableScope for the created subgraph; defaults to class name.

    Returns:
      A pair containing:
      - Output: A 2D Tensor with shape [batch_size x self.output_size]
      - New state: A 2D Tensor with shape [batch_size x self.state_size].
    """
    raise NotImplementedError("Abstract method")

  @property
  def input_size(self):
    """Integer: size of inputs accepted by this cell."""
    raise NotImplementedError("Abstract method")

  @property
  def output_size(self):
    """Integer: size of outputs produced by this cell."""
    raise NotImplementedError("Abstract method")

  @property
  def state_size(self):
    """Integer: size of state used by this cell."""
    raise NotImplementedError("Abstract method")

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
