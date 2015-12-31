import tensorflow as tf

from tasks import *
from utils import pp

flags = tf.app.flags
flags.DEFINE_string("task", "copy", "Task to run [copy]")
flags.DEFINE_integer("epoch", 100000, "Epoch to train [100000]")
flags.DEFINE_integer("input_dim", 10, "Dimension of input [10]")
flags.DEFINE_integer("output_dim", 10, "Dimension of output [10]")
flags.DEFINE_integer("min_length", 1, "Minimum length of input sequence [1]")
flags.DEFINE_integer("max_length", 20, "Maximum length of output sequence [20]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    with tf.device('/cpu:0'), tf.Session() as sess:
        FLAGS.sess = sess
    
        if FLAGS.task == 'copy':
            if FLAGS.is_train:
                copy_train(FLAGS)

            cell = NTMCell(input_dim=FLAGS.input_dim, output_dim=FLAGS.output_dim)
            ntm = NTM(cell, sess, 1, 40, forward_only=True)

            ntm.load(FLAGS.checkpoint_dir)
            import ipdb; ipdb.set_trace() 
            copy(ntm, 20, sess)
        elif FLAGS.task == 'recall':
            pass

if __name__ == '__main__':
    tf.app.run()
