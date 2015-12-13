import numpy as np
import tensorflow as tf

# Fix this in TensorFlow v0.6.
try:
    from tensorflow.models.rnn import linear
except AttributeError:
    from tensorflow.python.ops.rnn_cell import linear

init_std = 1
input_dim = 10
output_dim = 10
mem_rows = 128
mem_cols = 20
count_dim = 100

read_head_size = 1 # reader header size
write_head_size = 1 # writer header size
cont_layer_size = 1 # controller layer size

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
    w_name = "%s_w" % name if name else None
    b_name = "%s_b" % name if name else None

    w = tf.Variable(tf.random_normal(shape, stddev=stddev, name=w_name))
    b = tf.Variable(tf.constant(0.0, shape=[shape[1]], dtype=tf.float32, name=b_name),
                                trainable=True)
    return tf.nn.bias_add(tf.matmul(input, w), b)

# always zero?
# [batch_size x 1]
dummy = tf.placeholder(tf.float32, [None, 1])

# [batch_size x output_dim]
output_init = tf.tanh(linear.linear(dummy, output_dim, bias=True, bias_start=1))

# memory
m_init = tf.reshape(tf.tanh(linear.linear(dummy, mem_rows * mem_cols, bias=True)),
                    [mem_rows, mem_cols])

# read weights
write_init, read_init = [], []
for idx in xrange(read_head_size):
    write_w = tf.Variable(tf.random_normal([1, mem_rows]))
    write_b = tf.Variable(tf.cast(tf.range(mem_rows-2, 0, -1), dtype=tf.float32))
    write_init_lin = tf.nn.bias_add(tf.matmul(dummy, write_w), write_b)

    write_init.append(tf.nn.softmax(write_init_lin))
    #write_init.append(tf.nn.softmax(Linear(dummy, [1, mem_rows], name='write_lin')))

    read_init.append(tf.nn.softmax(Linear(dummy, [1, mem_cols], name='read_lin')))

# write weights
ww_init = []
for idx in xrange(write_head_size):
    ww_w = tf.Variable(tf.random_normal([1, mem_rows]))
    ww_b = tf.Variable(tf.cast(tf.range(mem_rows-2, 0, -1), dtype=tf.float32))
    ww_init_lin = tf.nn.bias_add(tf.matmul(dummy, ww_w), ww_b)

    ww_init.append(tf.nn.softmax(ww_init_lin))

# controller state
m_init, c_init = [], []
for idx in xrange(cont_layer_size):
    m_init.append(tf.tanh(Linear(dummy, [1, count_dim])))
    c_init.append(tf.tanh(Linear(dummy, [1, count_dim])))


#####################
# Combine memories
#####################

ww = tf.placeholder(tf.float32, [])
wr = tf.placeholder(tf.float32, [])
r = tf.placeholder(tf.float32, [])
m = tf.placeholder(tf.float32, [])
c = tf.placeholder(tf.float32, [])


######################
# Cotfect NTM cells
######################

ww_p = tf.placeholder(tf.float32, [])
wr_p = tf.placeholder(tf.float32, [])
r_p = tf.placeholder(tf.float32, [])
m_p = tf.placeholder(tf.float32, [])
c_p = tf.placeholder(tf.float32, [])

#############
# new cell
#############

input = tf.placeholder(tf.float32, [])

# previous memory state and read/write weights
memory_prev = tf.placeholder(tf.float32, [])
read_weight_prev = tf.placeholder(tf.float32, [])
write_weight_prev = tf.placeholder(tf.float32, [])

# vecter read from emory
read_prev = tf.placeohlder(tf.float32, [])

# ?????
# LSTM controller output
mtable_prev = tf.placeohlder(tf.float32, [])
ctable_prev = tf.placeohlder(tf.float32, [])

#mtable, ctable = new_controller_module()

#########################
# new_controller_module
#########################

input = input

read_prev = read_prev
mtable_prev = mtable_prev
ctable_prev = ctable_prev

mtable = []
ctable = []

for layer_idx in xrange(controller_layer_size):
    if controller_layer_size == 1:
        m_p = mtable_prev
        c_p = ctable_prev
    else:
        m_p = tf.gather(mtable_prev, layer_idx)
        c_p = tf.gather(ctable_prev, layer_idx)

    if layer_idx == 1:
        def new_gate():
            in_modules = [
                linear.linear(input, controller_dim, bias=True),
                linear.linear(m_p, controller_dim, bias=True),
            ]
            if read_heads == 1:
                in_modules.append(linear.linear(r_p, controller_dim, bias=True))
            else:
                for head_idx in xrange(read_heads):
                    vec = tf.gather(r_p, head_idx)
                    in_modules.append(linear.linear(r_p, controller_dim, bias=True))
            return tf.reduced_sum(in_modules, 0) 
    else:
        def new_gate():
            return tf.reduced_sum([
                linear.linear(input, controller_dim, bias=True),
                linear.linear(m_p, controller_dim, bias=True),
            ], 0)

    i = tf.sigmoid(new_gate())
    f = tf.sigmoid(new_gate())
    o = tf.sigmoid(new_gate())
    update = tf.tanh(new_gate())
