import tensorflow as tf


def is_tensor_or_variable(x):
    return tf.is_tensor(x) or isinstance(x, tf.Variable)