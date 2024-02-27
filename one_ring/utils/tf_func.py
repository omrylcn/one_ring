import tensorflow as tf


def is_tensor_or_variable(x):
    return tf.is_tensor(x) or isinstance(x, tf.Variable)


def set_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
        