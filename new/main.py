import pprint
import tensorflow as tf

from model import NTMCell

def main(_):
    with tf.Session() as sess:
        model = NTMCell(input_dim=150, output_dim=75)
        model.build_model()

if __name__ == '__main__':
    tf.app.run()
