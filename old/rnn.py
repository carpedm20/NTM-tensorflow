"""RNN helpers for TensorFlow models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.models.rnn import rnn_cell
from tensorflow.python.ops import control_flow_ops


def rnn(cell, inputs, initial_state=None, dtype=None,
        sequence_length=None, scope=None):
  if not isinstance(cell, rnn_cell.RNNCell):
    raise TypeError("cell must be an instance of RNNCell")
  if not isinstance(inputs, list):
    raise TypeError("inputs must be a list")
  if not inputs:
    raise ValueError("inputs must not be empty")

  outputs = []
  states = []
  with tf.variable_scope(scope or "RNN"):
    batch_size = tf.shape(inputs[0])[0]
    if initial_state is not None:
      state = initial_state
    else:
      if not dtype:
        raise ValueError("If no initial_state is provided, dtype must be.")
      state = cell.zero_state(batch_size, dtype)

    if sequence_length:  # Prepare variables
      zero_output_state = (
          tf.zeros(tf.pack([batch_size, cell.output_size]),
                   inputs[0].dtype),
          tf.zeros(tf.pack([batch_size, cell.state_size]),
                   state.dtype))
      max_sequence_length = tf.reduce_max(sequence_length)

    output_state = (None, None)
    for time, input_ in enumerate(inputs):
      if time > 0:
        tf.get_variable_scope().reuse_variables()
      output_state = cell(input_, state)
      if sequence_length:
        (output, state) = control_flow_ops.cond(
            time >= max_sequence_length,
            lambda: zero_output_state, lambda: output_state)
      else:
        (output, state) = output_state

      outputs.append(output)
      states.append(state)

    return (outputs, states)


def state_saving_rnn(cell, inputs, state_saver, state_name,
                     sequence_length=None, scope=None):
  initial_state = state_saver.State(state_name)
  (outputs, states) = rnn(cell, inputs, initial_state=initial_state,
                          sequence_length=sequence_length, scope=scope)
  save_state = state_saver.SaveState(state_name, states[-1])
  with tf.control_dependencies([save_state]):
    outputs[-1] = tf.identity(outputs[-1])

  return (outputs, states)
