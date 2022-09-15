import numpy as np
from typing import Union


def normalize(img: np.ndarray, mean: Union[list, tuple], std: Union[list, tuple], max_pixel_value: float = 255.0)->np.ndarray:
    """
    Normalize the image with mean and std

    This function takes from : https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/functional.py


    Parameters
    ----------
    img : np.ndarray
        Input image
    mean : list
        Mean of the image
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
