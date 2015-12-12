import numpy as np
import tensorflow as tf

init_std = 1
input_dim = 10
output_dim = 10
mem_rows = 128
mem_cols = 20
cont_dim = 100

read_head_size = 1

start_symbol = np.zeros(input_dim)
start_symbol[1] = 1
end_symbol = np.zeros(input_dim)
end_symbol[2] = 1

def generate_sequence(length, bits):
    seq = np.zeros(length, bits + 2)
    for idx in xrange(length):
        seq[idx, 2:bits+2] = np.random.rand(bits).round()
    return seq

def Linear(input, shape, stddev=0.5, name=None):
    if name:
        w_name, b_name = "%s_w" % name, "%s_b" % name
    else:
        w_name, b_name = None, None
    w = tf.Variable(tf.random_normal(shape, stddev=stddev, name=w_name))
    b = tf.Variable(tf.constant(0.0, shape=[shape[1]], dtype=tf.float32, name=b_name),
                                trainable=True)
    return tf.nn.bias_add(tf.matmul(input, w), b)

# always zero?
# [batch_size x 1]
dummy = tf.placeholder(tf.float32, [None, 1])

# [batch_size x output_dim]
output_init = tf.tanh(Linear(dummy, [1, output_dim], name='output_lin'))


##########
# memory
##########

# [mem_rows x mem_cols]
m_init = tf.reshape(tf.tanh(Linear(dummy, [1, mem_rows * mem_cols])), \
                    [mem_rows, mem_cols])

# read weights
write_init, read_init = [], []
for idx in xrange(read_head_size):
    write_init.append(tf.nn.softmax(Linear(dummy, [1, mem_rows], name='write_lin')))
    read_init.append(tf.nn.softmax(Linear(dummy, [1, mem_cols], name='read_lin')))

