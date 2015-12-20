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
    seq = np.zeros([length, 1, bits + 2])
    for idx in xrange(length):
        seq[idx, :, 2:bits+2] = np.random.rand(bits).round()
    return seq

def forward(model, seq, print_flag):
    seq_len = len(seq)
    loss = 0

    model.forward(start_symbol)
