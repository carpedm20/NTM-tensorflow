import numpy as np
import tensorflow as tf

from layers import *

init_std = 1
input_dim = 10
output_dim = 10
mem_rows = 128
mem_cols = 20
controller_dim = 100

read_head_size = 1 # reader header size
write_head_size = 1 # writer header size
controller_layer_size = 1 # controller layer size

start_symbol = np.zeros(input_dim)
start_symbol[1] = 1
end_symbol = np.zeros(input_dim)
end_symbol[2] = 1

def generate_sequence(length, bits):
    seq = np.zeros(length, bits + 2)
    for idx in xrange(length):
        seq[idx, 2:bits+2] = np.random.rand(bits).round()
    return seq

# always zero?
# [batch_size x 1]
dummy = tf.placeholder(tf.float32, [None, 1])

# [batch_size x output_dim]
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

input = tf.placeholder(tf.float32, [None, input_dim])

# previous memory state and read/write weights
memory_prev = tf.placeholder(tf.float32, [None, mem_cols * mem_rows])
read_weight_prev = tf.placeholder(tf.float32, [None, mem_cols])
write_weight_prev = tf.placeholder(tf.float32, [None, mem_cols])

# vecter read from emory
read_prev = tf.placeholder(tf.float32, [])

# ?????
# LSTM controller output
mtable_prev = tf.placeholder(tf.float32, [None, mem_cols * mem_rows])
ctable_prev = tf.placeholder(tf.float32, [None, mem_cols * mem_rows])

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
                Linear(input, controller_dim, bias=True),
                Linear(m_p, controller_dim, bias=True),
            ]
            if read_heads == 1:
                in_modules.append(Linear(r_p, controller_dim, bias=True))
            else:
                for head_idx in xrange(read_heads):
                    vec = tf.gather(r_p, head_idx)
                    in_modules.append(Linear(r_p, controller_dim, bias=True))
            return tf.reduced_sum(in_modules, 0) 
    else:
        def new_gate():
            return tf.add(
                Linear(input, controller_dim, bias=True),
                Linear(m_p, controller_dim, bias=True),
            )

    i = tf.sigmoid(new_gate())
    f = tf.sigmoid(new_gate())
    o = tf.sigmoid(new_gate())
    update = tf.tanh(new_gate())

    ctable[layer_idx] = tf.add(
        tf.mul(f, c_p),
        tf.mul(i, update)
    )
    mtable[layer_idx] = tf.mul(o, tf.tanh(ctable[layer_idx]))

mtable = tf.identity(ctable)
ctable = tf.identity(ctable)
