"""Module for constructing Nueral Turing Machine Cells."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from six.moves import xrange

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh


class NTMCell(object):
    """Neural Turing Machine (NTM) Cell.

    The implementation is based on: http://arxiv.org/pdf/1410.5401.pdf.
    """

    def __init__(self, num_units, input_size, output_size):
        """Initialize the parameters for an NTM cell.

        Args:
            num_units: int, The number of units in the NTM cell
            input_size: int, The dimensionality of the inputs into the NTM cell
            output_size: int, The dimensionality of the outputs from the NTM cell
        """
        self._num_units = num_units
        self._input_size = input_size
        self._output_size = output_size

        self._controller = build_controller()
        self._memory = build_controller()

    def __call__(self, inputs, state, scope=None):
        """Run one step of NTM.

        Args:
            inputs: 2D Tensor with shape [batch_size x self.input_size].
            state: 2D Tensor with shape [batch_size x self.state_size].
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
        zeros = array_ops.zeros(
                array_ops.pack([batch_size, self.state_size]), dtype=dtype)
        zeros.set_shape([None, self.state_size])
        return zeros


