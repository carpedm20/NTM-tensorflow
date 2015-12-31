import numpy as np

def recall(seq_length):
    pass

def generate_recall_sequence(length, num_items):
    seq = np.zeros([length, bits + 2], dtype=np.float32)
    for idx in xrange(length):
        seq[idx, 2:bits+2] = np.random.rand(bits).round()
    return list(seq)
