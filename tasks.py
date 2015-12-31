import os
import time
import numpy as np
import tensorflow as tf
from random import randint

from ntm import NTM
from utils import pprint
from ntm_cell import NTMCell

print_interval = 5

def recall(seq_length):
    pass

def copy(ntm, seq_length, sess, print_=True):
    start_symbol = np.zeros([ntm.cell.input_dim], dtype=np.float32)
    start_symbol[0] = 1
    end_symbol = np.zeros([ntm.cell.input_dim], dtype=np.float32)
    end_symbol[1] = 1

    seq = generate_copy_sequence(seq_length, ntm.cell.input_dim - 2)

    feed_dict = {input_:vec for vec, input_ in zip(seq, ntm.inputs)}
    feed_dict.update(
        {true_output:vec for vec, true_output in zip(seq, ntm.true_outputs)}
    )
    feed_dict.update({
        ntm.start_symbol: start_symbol,
        ntm.end_symbol: end_symbol
    })

    result = sess.run(ntm.get_outputs(seq_length) + [ntm.get_loss(seq_length)], feed_dict=feed_dict)

    outputs = result[:-1]
    loss = result[-1]

    if print_:
        np.set_printoptions(suppress=True)
        print(" true output : ")
        pprint(seq)
        print(" predicted output :")
        pprint(np.round(outputs))
        print(" Loss : %f" % loss)
        np.set_printoptions(suppress=False)
    else:
        return seq, outputs, loss

def copy_train(config):
    sess = config.sess

    if not os.path.isdir(config.checkpoint_dir):
        raise Exception(" [!] Directory %s not found" % config.checkpoint_dir)

    # delimiter flag for start and end
    start_symbol = np.zeros([config.input_dim], dtype=np.float32)
    start_symbol[0] = 1
    end_symbol = np.zeros([config.input_dim], dtype=np.float32)
    end_symbol[1] = 1

    cell = NTMCell(input_dim=config.input_dim, output_dim=config.output_dim)
    ntm = NTM(cell, sess, config.min_length, config.max_length)

    print(" [*] Initialize all variables")
    tf.initialize_all_variables().run()
    print(" [*] Initialization finished")

    start_time = time.time()
    for idx in xrange(config.epoch):
        seq_length = randint(config.min_length, config.max_length)
        seq = generate_copy_sequence(seq_length, config.input_dim - 2)

        feed_dict = {input_:vec for vec, input_ in zip(seq, ntm.inputs)}
        feed_dict.update(
            {true_output:vec for vec, true_output in zip(seq, ntm.true_outputs)}
        )
        feed_dict.update({
            ntm.start_symbol: start_symbol,
            ntm.end_symbol: end_symbol
        })

        _, cost, step = sess.run([ntm.optims[seq_length],
                                  ntm.get_loss(seq_length),
                                  ntm.global_step], feed_dict=feed_dict)

        if idx % 100 == 0:
            ntm.saver.save(sess,
                           os.path.join(config.checkpoint_dir, "copy/NTM_copy.model"),
                           global_step = step.astype(int),
                           latest_filename = 'copy_checkpoint')

        if idx % print_interval == 0:
            print("[%5d] %2d: %.2f (%.1fs)" % (idx, seq_length, cost, time.time() - start_time))

    print("Training Copy task finished")

def generate_copy_sequence(length, bits):
    seq = np.zeros([length, bits + 2], dtype=np.float32)
    for idx in xrange(length):
        seq[idx, 2:bits+2] = np.random.rand(bits).round()
    return list(seq)

def generate_recall_sequence(length, num_items):
    seq = np.zeros([length, bits + 2], dtype=np.float32)
    for idx in xrange(length):
        seq[idx, 2:bits+2] = np.random.rand(bits).round()
    return list(seq)
