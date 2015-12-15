import pprint
import tensorflow as tf

from model import NTM

with tf.Session() as sess:
    model = NTM()
