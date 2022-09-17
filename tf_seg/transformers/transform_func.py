import numpy as np
from typing import Union
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops


def normalize(
    img: np.ndarray,
    mean: Union[list, tuple] = (0.485, 0.456, 0.406),
    std: Union[list, tuple] = (0.229, 0.224, 0.225),
    max_pixel_value: float = 255.0,
) -> np.ndarray:
    """
    Normalize the image with mean and std,
    it is used imagenet mean and std values by default.

    This function takes from : https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/functional.py


    Parameters
    ----------
    img : np.ndarray
        Input image
    mean : list
        Mean of the image,default = [0.485, 0.456, 0.406]
    std : list
        Standard deviation of the image
    max_pixel_value : float, optional
        Maximum pixel value, by default 255.0

    Returns
    -------
    image : np.ndarray
        Normalized image

    """

    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img


# @tf.function(input_signature=[tf.TensorSpec(None, tf.float32)])
# def tf_normalize(image):
#     """
#     Normalize a tensor with imagenet mean and std.
#     """
#     image = tf.numpy_function(normalize, [image], tf.float32)
#     return image


def tf_normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255):
    """
    Normalize a tensor with imagenet mean and std.
    """

    with ops.name_scope(None, "per_image_standardization", [image]) as scope:
        image = ops.convert_to_tensor(image, name="image")
        # image = _AssertAtLeast3DImage(image)

        image = math_ops.cast(image, dtype=tf.dtypes.float32)
        mean = tf.constant(mean, dtype=tf.float32)
        std = tf.constant(std, dtype=tf.float32)

        adjusted_stddev = std * max_pixel_value

        image -= mean
        image = math_ops.divide(image, adjusted_stddev, name=scope)
        return image
