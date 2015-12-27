from random import randint

import numpy as np
import tensorflow as tf

from ntm import NTM
from ntm_cell import NTMCell

def main(_):
    copy()

def copy():
    epoch = 10
    print_interval = 1

    input_dim = 150
    output_dim = 75
    min_length = 1
    max_length = 20

    with tf.Session() as sess:
        start_symbol = np.zeros([1, input_dim])
        start_symbol[0][1] = 1
        end_symbol = np.zeros([1, input_dim])
        end_symbol[0][2] = 1

        cell = NTMCell(input_dim=input_dim, output_dim=output_dim)
        ntm = NTM(cell)

        for idx in xrange(epoch):
            print_flag = (idx % print_interval == 0)
            loss = 0

            seq_length = randint(min_length, max_length)
            seq = generate_sequence(seq_length, input_dim - 2)

            outputs = ntm.forward(seq)

def generate_sequence(length, bits):
    seq = np.zeros([length, bits + 2], dtype=np.float32)
    for idx in xrange(length):
        seq[idx, 2:bits+2] = np.random.rand(bits).round()
    return list(seq)

if __name__ == '__main__':
    tf.app.run()
