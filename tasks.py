import os
import time
import numpy as np
import tensorflow as tf
from random import randint

from ntm import NTM
from ntm_cell import NTMCell

epoch = 10000
print_interval = 10

input_dim = 10
output_dim = 10

min_length = 1
max_length = 2

checkpoint_dir = './checkpoint'

def recall(seq_length):
    pass

def copy(seq_length):
    with tf.device('/cpu:0'), tf.Session() as sess:
        with tf.variable_scope("NTM") as scope:
            cell = NTMCell(input_dim=input_dim, output_dim=output_dim)
            ntm = NTM(cell, sess, min_length, max_length,
                      scope=scope, forward_only=True)

        ntm.load(checkpoint_dir)

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

        result = sess.run(ntm.outputs[seq_length] + [ntm.losses[seq_length]], feed_dict=feed_dict)

        outputs = result[0]
        loss = result[1]

        print(" true output : %s" % seq)
        print(" predicted output : %s" % outputs)
        print(" Loss : %f" % loss)

def copy_train():
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

        print(" [*] Initialize all variables")
        tf.initialize_all_variables().run()
        print(" [*] Initialization finished")

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
                ntm.saver.save(sess,
                               os.path.join(checkpoint_dir, "NTM.model"),
                               global_step = step.astype(int))

            if idx % print_interval == 0:
                print("[%5d] %2d: %.2f (%.1fs)" % (idx, seq_length, cost, time.time() - start_time))

def generate_copy_sequence(length, bits):
    seq = np.zeros([length, bits + 2], dtype=np.float32)
    for idx in xrange(length):
        seq[idx, 2:bits+2] = np.random.rand(bits).round()
    return list(seq)
