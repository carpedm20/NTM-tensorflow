import tensorflow as tf

def Linear(inputs, output_size, stddev=0.5, bias=True, bias_init=0.0, name=None):
    total_input_size = 0
    if type(inputs) != list:
        inputs = [inputs]
    shapes = [a.get_shape().as_list() for a in inputs]

    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D inputuments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of inputuments: %s" % str(shapes))
        else:
            total_input_size += shape[1]

    w_name = "%s_w" % name if name else None
    b_name = "%s_b" % name if name else None

    w = tf.Variable(tf.random_normal([total_input_size, output_size], \
                                      stddev=stddev, name=w_name))
    if len(inputs) == 1:
      mul = tf.matmul(inputs[0], w)
    else:
      mul = tf.matmul(tf.concat(1, inputs), w)

    if bias:
        b = tf.Variable(tf.constant(bias_init, shape=[output_size], dtype=tf.float32))
        return tf.nn.bias_add(mul, b)
    else:
        return mul
