import tensorflow as tf

from tasks import *

flags = tf.app.flags
flags.DEFINE_string("task", "copy", "task to run [copy or recall]")
FLAGS = flags.FLAGS

def main(_):
    if FLAGS.task == 'copy':
        copy_train()
        copy(5)
    elif FLAGS.task == 'recall':
        pass

if __name__ == '__main__':
    tf.app.run()
