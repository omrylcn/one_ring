"""
This is experimental data augmentation modıle. It contains tensorflow and jax functions.

"""


import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

# from tensorflow.python.framework import dtypes
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.random_ops import random_uniform
from tensorflow.python.ops import gen_math_ops

import jax
from jax import random


key = random.PRNGKey(0)


@tf.function
def tf_random_crop(image, mask, channels=(3, 3), name=None, seed=48, crop_percent=0.8):
    """Randomly crops the given image and mask.

    Parameters
    ----------
    image : tf.Tensor
        Image to be cropped.
    mask : tf.Tensor
        Mask to be cropped.
    channels : tuple of ints
        Number of channels in the image and mask.
    name : str
        Name of the op.
    seed : int
        Random seed.
    crop_percent : float
        Percentage of the image to be cropped.

    Notes
    --------
    This function recode the original code from tf.image.random_crop for segmnetation data

    Returns
    -------
    i : tf.Tensor
        Cropped image.
    m : tf.Tensor
        Cropped mask.

    """
    with ops.name_scope(name, "random_crop", [image]) as name:

        shape = array_ops.shape(image)
        f_shape = gen_math_ops.cast(shape, tf.float32)
        crop_percent = ops.convert_to_tensor(crop_percent, name="percent")

        h = gen_math_ops.cast(f_shape[0] * crop_percent, tf.int32)
        w = gen_math_ops.cast(f_shape[1] * crop_percent, tf.int32)

        i_size = gen_math_ops.cast((h, w, channels[0]), tf.int32)
        m_size = gen_math_ops.cast((h, w, channels[1]), tf.int32)

        check = control_flow_ops.Assert(math_ops.reduce_all(shape >= i_size), ["Need value.shape >= size, got ", shape, i_size], summarize=1000)

        shape = control_flow_ops.with_dependencies([check], shape)
        limit = shape - i_size + 1
        offset = random_uniform(array_ops.shape(shape), dtype=i_size.dtype, maxval=i_size.dtype.max, seed=seed) % limit

        m = array_ops.slice(mask, offset, m_size, name="image_crop")
        i = array_ops.slice(image, offset, i_size, name="mask_crop")

    return i, m


@jax.jit
def jax_random_crop(image, mask, image_crop_sizes=[512, 512, 3], mask_crop_sizes=[512, 512, 3]):

    image_shape = image.shape
    crop_sizes = image_crop_sizes
    random_keys = jax.random.split(key, len(crop_sizes))

    # print(image_crop_size,mask_crop_size)
    slice_starts = [jax.random.randint(k, (), 0, img_size - crop_size + 1) for k, img_size, crop_size in zip(random_keys, image_shape, crop_sizes)]
    i_out = jax.lax.dynamic_slice(image, slice_starts, image_crop_sizes)
    m_out = jax.lax.dynamic_slice(mask, slice_starts, mask_crop_sizes)

    return i_out, m_out
