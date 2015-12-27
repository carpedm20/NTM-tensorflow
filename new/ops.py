import math
import tensorflow as tf

from utils import *

def Linear(inputs, output_size, stddev=0.5,
           is_range=False, squeeze=False,
           name=None, reuse=None):
    with tf.variable_scope("Linear", reuse=reuse):
        total_input_size = 0
        if type(inputs) != list:
            inputs = [inputs]
        try:
            shapes = [input_.get_shape().as_list() for input_ in inputs]
        except:
            shapes = [input_.shape for input_ in inputs]

        for shape in shapes:
            if len(shape) != 2:
                raise ValueError("Linear is expecting 2D inputuments: %s" % str(shapes))
            if not shape[1]:
                raise ValueError("Linear expects shape[1] of inputuments: %s" % str(shapes))
            else:
                total_input_size += shape[1]

        w_name = "%s_w" % name if name else None
        b_name = "%s_b" % name if name else None

        w = tf.get_variable(w_name, [total_input_size, output_size], tf.float32,
                            tf.random_normal_initializer(stddev=stddev))
        if len(inputs) == 1:
            mul = tf.matmul(inputs[0], w)
        else:
            mul = tf.matmul(tf.concat(1, inputs), w)

        if is_range:
            def identity_initializer(tensor):
                def _initializer(shape, dtype=tf.float32):
                    return tf.identity(tensor)
                return _initializer

            range_ = tf.cast(tf.reverse(tf.range(1, output_size+1, 1), [True]), tf.float32)
            b = tf.get_variable(b_name, [output_size], tf.float32, identity_initializer(range_))
        else:
            b = tf.get_variable(b_name, [output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))

        if squeeze:
            return tf.squeeze(tf.nn.bias_add(mul, b))
        else:
            return tf.nn.bias_add(mul, b)

def smooth_cosine_similarity(m, v):
    """Compute smooth cosine similarity.

    Args:
        m: a 2-D `Tensor` (matrix)
        v: a 1-D `Tensor` (vector)
    """
    shape_x = m.get_shape().as_list()
    shape_y = m.get_shape().as_list()
    if shape_x[1] != shape_y[0]:
        raise ValueError("Smooth cosine similarity is expecting same dimemsnion")

    m_norm = tf.sqrt(tf.reduce_sum(tf.pow(m, 2),1))
    v_norm = tf.sqrt(tf.matmul(v, v, transpose_b=True))
    m_dot_v = tf.matmul(m, v, transpose_b=True)

    similarity = tf.div(m_dot_v, (m_norm * k_norm + 1e-3))
    return similarity

def circular_convolution(m, v):
    """Compute circular convolution

    Args:
        m: a 2-D `Tensor` (matrix)
        v: a 1-D `Tensor` (vector)
    """
    size = vector.get_shape()[0]
    kernel_size = kernel.get_shape()[0]
    kernel_shift = math.floor(kernel_size/2.0)
    output = tf.zeros_like(vector)

    def loop(idx):
        if idx < 0: return size + idx
        if idx >= size : return idx - size
        else: return idx

    kernels = []
    for i in xrange(size):
        indices = [loop(i+j) for j in xrange(kernel_shift, -kernel_shift-1, -1)]
        v = tf.gather(vector, indices)
        kernels.append(tf.reduce_sum(v * kernel, 0, keep_dims=True))

    output = tf.dynamic_stitch([[i] for i in xrange(size)], kernels)

    # # code with double loop
    # for i in xrange(size):
    #     for j in xrange(kernel_size):
    #         idx = i + kernel_shift - j + 1
    #         if idx < 0: idx = idx + size
    #         if idx >= size: idx = idx - size
    #         w = tf.gather(vector, int(idx)) * tf.gather(kernel, j)
    #         output = tf.scatter_add(output, [i], tf.reshape(w, [1, -1]))

    return output

def outer_product(*inputs):
    """Compute outer product

    Args:
        m: a 2-D `Tensor` (matrix)
        v: a 1-D `Tensor` (vector)
    """
    order = len(inputs)
    if order == 2:
        output = tf.mul(inputs[0], inputs[1])
    elif order == 3:
        size = []
        idx = 1
        for i in xrange(order):
            size.append(inputs[i].get_shape()[0])
        output = tf.zeros(size)

        u, v, w = inputs[0], inputs[1], inputs[2]
        uv = tf.mul(inputs[0], inputs[1])
        for i in xrange(self.size[-1]):
            output = tf.scatter_add(output, [0,0,i], uv)

    return output

def scalar_mul(x, beta, name=None):
    return x * beta

def scalar_div(x, beta, name=None):
    return x / beta
