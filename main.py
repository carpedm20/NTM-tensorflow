from random import randint

import numpy as np
import tensorflow as tf

from ntm import NTM
from ntm_cell import NTMCell

def main(_):
    copy()

def copy():
    epoch = 1
    print_interval = 1

    input_dim = 10
    output_dim = 10
    min_length = 1
    max_length = 5
    #max_length = 20

    with tf.device('/cpu:0'), tf.Session() as sess:
        # delimiter flag for start and end
        start_symbol = np.zeros([input_dim], dtype=np.float32)
        start_symbol[0] = 1
        end_symbol = np.zeros([input_dim], dtype=np.float32)
        end_symbol[1] = 1

        cell = NTMCell(input_dim=input_dim, output_dim=output_dim)
        ntm = NTM(cell)

        is_initialized = False

        for idx in xrange(epoch):
            print_flag = (idx % print_interval == 0)
            loss = 0

            seq_length = randint(min_length, max_length)
            seq = generate_sequence(seq_length, input_dim - 2)

            outputs, loss = ntm.forward([start_symbol] + seq + [end_symbol], logging=print_flag)

            if not is_initialized:
                tf.initialize_all_variables().run()
                is_initialized = True

            cost, state = sess.run([ntm.loss, ntm.optim])
            ntm.print_write_max()

def generate_sequence(length, bits):
    seq = np.zeros([length, bits + 2], dtype=np.float32)
    for idx in xrange(length):
        seq[idx, 2:bits+2] = np.random.rand(bits).round()
    return list(seq)

if __name__ == '__main__':
    tf.app.run()
