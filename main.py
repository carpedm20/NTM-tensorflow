import pprint
import tensorflow as tf

from model import NTM
from tasks.copy import task as copy_task
from tasks.recall import task as recall_task

pp = pprint.PrettyPrinter()

flags = tf.app.flags

flags.DEFINE_integer("input_dim", 150, "dimension of input vectors [150]")
flags.DEFINE_integer("output_dim", 75, "dimension of output vectors [75]")
flags.DEFINE_integer("mem_size", 128, "szie of  memory [128]")
flags.DEFINE_integer("mem_dim", 20, "dimension of memory [20]")
flags.DEFINE_integer("controller_dim", 100, "dimension of controller [100]")
flags.DEFINE_integer("controller_layer_size", 1, "number of LSTM layer in controller [1]")
flags.DEFINE_integer("shift_range", 1, "allowed range of shifting read & write weights [1]")
flags.DEFINE_integer("write_head_size", 1, "number of write heads [1]")
flags.DEFINE_integer("read_head_size", 1, "number of read heads [1]")
flags.DEFINE_integer("epoch", 10000, "number of epoch to train [10000]")
flags.DEFINE_integer("lr_rate", 1e-4, "learning rate [1e-4]")
flags.DEFINE_integer("momentum",0.9, "momentum [0.9]")
flags.DEFINE_integer("decay", 0.95, "learning_rate decay rate [0.95]")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "checkpoint directory [checkpoints]")
flags.DEFINE_string("task", "copy", "task to run [copy]")
flags.DEFINE_boolean("is_test", False, "True for testing, False for Training [False]")
flags.DEFINE_boolean("show", False, "print progress [False]")

FLAGS = flags.FLAGS

def main(_):
    pp.pprint(FLAGS.__flags)

    with tf.Session() as sess:
        model = NTM(FLAGS, sess)
        model.build_model()

        task = FLAGS.task
        if task == 'copy':
            copy_task(model, FLAGS)
        elif task == 'recall':
            recall_task(model, FLAGS)
        else:
            raise Exception(" [!] Unkown task: %s" % task)

if __name__ == '__main__':
    tf.app.run()
