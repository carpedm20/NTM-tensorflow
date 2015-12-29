import os
import time
import numpy as np
import tensorflow as tf
from random import randint

from ntm import NTM
from ntm_cell import NTMCell

def copy(ntm, seq_length):
    with tf.device('/cpu:0'), tf.Session() as sess:
        start_symbol = np.zeros([ntm.cell.input_dim], dtype=np.float32)
        start_symbol[0] = 1
        end_symbol = np.zeros([ntm.cell.input_dim], dtype=np.float32)
        end_symbol[1] = 1

        seq = generate_copy_sequence(seq_length, input_dim - 2)

        feed_dict = {input_:vec for vec, input_ in zip(seq, ntm.inputs)}
        feed_dict.update(
            {true_output:vec for vec, true_output in zip(seq, ntm.true_outputs)}
        )
        feed_dict.update({
            ntm.start_symbol: start_symbol,
            ntm.end_symbol: end_symbol
        })

        outputs = sess.run(ntm.outputs + ntm.losses[seq_length], feed_dict=feed_dict)

        outputs = outputs[:-1]
        loss = outputs[-1]

        print(" true output : %s" % seq)
        print(" predicted output : %s" % outputs)
        print(" Loss : %f" % loss)

def copy_train():
    epoch = 10000
    print_interval = 10

    input_dim = 10
    output_dim = 10
    min_length = 5
    max_length = 5
    #max_length = 20

    checkpoint_dir = './checkpoint'
    if not os.path.isdir(checkpoint_dir):
        raise Exception(" [!] Directory %s not found" % checkpoint_dir)

    with tf.device('/cpu:0'), tf.Session() as sess:
        # delimiter flag for start and end
        start_symbol = np.zeros([input_dim], dtype=np.float32)
        start_symbol[0] = 1
        end_symbol = np.zeros([input_dim], dtype=np.float32)
        end_symbol[1] = 1

        with tf.variable_scope("NTM") as scope:
            cell = NTMCell(input_dim=input_dim, output_dim=output_dim)
            ntm = NTM(cell, sess, min_length, max_length, scope=scope)

        tf.initialize_all_variables().run()

        start_time = time.time()
        for idx in xrange(epoch):
            seq_length = randint(min_length, max_length)
            seq = generate_copy_sequence(seq_length, input_dim - 2)

            feed_dict = {input_:vec for vec, input_ in zip(seq, ntm.inputs)}
            feed_dict.update(
                {true_output:vec for vec, true_output in zip(seq, ntm.true_outputs)}
            )
            feed_dict.update({
                ntm.start_symbol: start_symbol,
                ntm.end_symbol: end_symbol
            })

            _, cost, step = sess.run([ntm.optims[seq_length],
                                      ntm.losses[seq_length],
                                      ntm.global_step], feed_dict=feed_dict)

            if idx % 100 == 0:
                ntm.saver.save(self.sess,
                               os.path.join(checkpoint_dir, "NTM.model"),
                               global_step = ntm.step.astype(int))

            if idx % print_interval == 0:
                print("[%4d] %d: %.2f (%.1fs)" % (idx, seq_length, cost, time.time() - start_time))

def generate_copy_sequence(length, bits):
    seq = np.zeros([length, bits + 2], dtype=np.float32)
    for idx in xrange(length):
        seq[idx, 2:bits+2] = np.random.rand(bits).round()
    return list(seq)
