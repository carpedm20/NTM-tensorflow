import numpy as np
import tensorflow as tf

input_dim = 10
output_dim = 10
mem_rows = 128
mem_cols = 20
cont_dim = 100

start_symbol = np.zeros(input_dim)
start_symbol[1] = 1
end_symbol = np.zeros(input_dim)
end_symbol[2] = 1

def generate_sequence(length, bits):
    seq = np.zeros(length, bits + 2)
    for idx in xrange(length):
        seq[idx, 2:bits+2] = np.random.rand(bits).round()
    return seq

def forward(model, seq, print_flag):
    pass
