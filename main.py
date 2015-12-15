import pprint
import tensorflow as tf

from model import NTM

pp = pprint.PrettyPrinter()

flags = tf.app.flags

flags.DEFINE_integer("input_dim", 150, "dimension of input vectors [150]")
flags.DEFINE_integer("output_dim", 75, "dimension of output vectors [75]")
flags.DEFINE_integer("memory_size", 128, "szie of  memory [128]")
flags.DEFINE_integer("memory_dim", 20, "dimension of memory [20]")
flags.DEFINE_integer("controller_dim", 100, "dimension of controller [100]")
flags.DEFINE_integer("controller_layer_size", 1, "number of LSTM layer in controller [1]")
flags.DEFINE_integer("shift_range", 1, "allowed range of shifting read & write weights [1]")
flags.DEFINE_integer("write_head_size", 1, "number of write heads [1]")
flags.DEFINE_integer("read_head_size", 1, "number of read heads [1]")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "checkpoint directory [checkpoints]")
flags.DEFINE_boolean("is_test", False, "True for testing, False for Training [False]")
flags.DEFINE_boolean("show", False, "print progress [False]")

FLAGS = flags.FLAGS

def main(_):
    pp.pprint(FLAGS.__flags)

    with tf.Session() as sess:
        model = NTM(FLAGS)

if __name__ == '__main__':
    tf.app.run()
