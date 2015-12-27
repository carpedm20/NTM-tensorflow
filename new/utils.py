import numpy as np

try:
    xrange
except NameError:
    xrange = range

def argmax(x):
    index = 0
    max_num = x[index]
    for idx in xrange(1, len(x)-1):
        if x[idx] > max_num:
            index = idx
            max_num = x[idx]
    return index, max_num
