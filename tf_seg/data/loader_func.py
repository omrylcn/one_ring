import os
from glob import glob
from typing import Union, Dict, Tuple, List
from omegaconf import DictConfig, ListConfig
import tensorflow as tf

from tf_seg.data import DataLoader


def get_camvid_data_loader(
    data_config: Union[Dict, DictConfig, ListConfig],
    train_data: bool = True,
    val_data: bool = True,
    test_data: bool = True,
) -> Tuple[tf.data.Dataset]:

    """
    A data loader function for camvid dataset. A data loader is a class to load tf.data.Dataset objects from given data path

    """
    output_type = (tf.float32, tf.float32)
    # image_output_type = getattr(tf, data_config["output_type"][0].split(".")[-1])  # objectinput must be like tf.float32 string
    # mask_output_type = getattr(tf, data_config["output_type"][1].split(".")[-1])

    data_loader_list = []
    if train_data:
        train_image_paths = sorted(glob(os.path.join(data_config["path"], "train/*.png")))
        train_mask_paths = sorted(glob(os.path.join(data_config["path"], "train_labels/*.png")))

        train_data_loader = DataLoader(
            train_image_paths,
            train_mask_paths,
            name=("train_" + data_config["name"]),
            image_size=data_config["image_size"],
            normalizing=data_config["normalizing"],
            batch_size=data_config["batch_size"],
            output_type=output_type,  # (image_output_type, mask_output_type),
            one_hot_encoding=data_config["one_hot_encoding"],
            channels=data_config["channels"],
            palette=data_config["palette"],
            background_adding=data_config["background_adding"],
        )

        data_loader_list.append(train_data_loader)

    if val_data:
        val_image_paths = sorted(glob(os.path.join(data_config["path"], "val/*.png")))
        val_mask_paths = sorted(glob(os.path.join(data_config["path"], "val_labels/*.png")))

        val_data_loader = DataLoader(
            val_image_paths,
            val_mask_paths,
            name=("val_" + data_config["name"]),
            image_size=data_config["image_size"],
            normalizing=data_config["normalizing"],
            batch_size=data_config["batch_size"],
            output_type=output_type,  # (image_output_type, mask_output_type),
            one_hot_encoding=data_config["one_hot_encoding"],
            channels=data_config["channels"],
            palette=data_config["palette"],
            background_adding=data_config["background_adding"],
        )
        data_loader_list.append(val_data_loader)

    if test_data:
        test_image_paths = sorted(glob(os.path.join(data_config["path"], "test/*.png")))
        test_mask_paths = sorted(glob(os.path.join(data_config["path"], "test_labels/*.png")))

        test_data_loader = DataLoader(
            test_image_paths,
            test_mask_paths,
            name=("test_" + data_config["name"]),
            image_size=data_config["image_size"],
            normalizing=data_config["normalizing"],
            batch_size=data_config["batch_size"],
            output_type=output_type,  # (image_output_type, mask_output_type),
            one_hot_encoding=data_config["one_hot_encoding"],
            channels=data_config["channels"],
            palette=data_config["palette"],
            background_adding=data_config["background_adding"],
        )
        data_loader_list.append(test_data_loader)

    return tuple(data_loader_list)


def get_custom_data_loader(
    data_config: Union[dict, DictConfig, ListConfig],
    train_data: bool = True,
    val_data: bool = True,
    test_data: bool = True,
) -> Tuple[tf.data.Dataset]:

    """
    A loader function for custom dataset. A data loader is a class to load tf.data.Dataset objects from given data path

    Parameters
    ----------
    data_config : Union[dict, DictConfig, ListConfig]
        A configuration for data loader.
    train_data : bool, optional
        A flag to load train data. The default is True.
    val_data : bool, optional
        A flag to load validation data. The default is True.
    test_data : bool, optional
        A flag to load test data. The default is True.
    paths : Dict[str, str], optional
        A dictionary of paths to load data. The default is {}.

    Returns
    -------
        pass

    """

    image_output_type = getattr(tf, data_config["output_type"][0].split(".")[-1])  # objectinput must be like tf.float32 string
    mask_output_type = getattr(tf, data_config["output_type"][1].split(".")[-1])

    data_loader_list = []
    if train_data:
        train_image_paths = sorted(glob(os.path.join(data_config["path"], "train_images/*")))
        train_mask_paths = sorted(glob(os.path.join(data_config["path"], "train_masks/*")))
        _check_paths(train_image_paths, train_mask_paths, "train data")

        train_data_loader = DataLoader(
            train_image_paths,
            train_mask_paths,
            name=("train_" + data_config["name"]),
            image_size=data_config["image_size"],
            normalizing=data_config["normalizing"],
            batch_size=data_config["batch_size"],
            output_type=(image_output_type, mask_output_type),
            one_hot_encoding=data_config["one_hot_encoding"],
            channels=data_config["channels"],
            palette=data_config["palette"],
            background_adding=data_config["background_adding"],
        )

        data_loader_list.append(train_data_loader)

    if val_data:
        val_image_paths = sorted(glob(os.path.join(data_config["path"], "val_images/*")))
        val_mask_paths = sorted(glob(os.path.join(data_config["path"], "val_masks/*")))
        _check_paths(val_image_paths, val_mask_paths, "val data")

        val_data_loader = DataLoader(
            val_image_paths,
            val_mask_paths,
            name=("val_" + data_config["name"]),
            image_size=data_config["image_size"],
            normalizing=data_config["normalizing"],
            batch_size=data_config["batch_size"],
            output_type=(image_output_type, mask_output_type),
            one_hot_encoding=data_config["one_hot_encoding"],
            channels=data_config["channels"],
            palette=data_config["palette"],
            background_adding=data_config["background_adding"],
        )
        data_loader_list.append(val_data_loader)

    if test_data:
        test_image_paths = sorted(glob(os.path.join(data_config["path"], "test_images/*")))
        test_mask_paths = sorted(glob(os.path.join(data_config["path"], "test_masks/*")))
        _check_paths(test_image_paths, test_mask_paths, "test data")

        test_data_loader = DataLoader(
            test_image_paths,
            test_mask_paths,
            name=("test_" + data_config["name"]),
            image_size=data_config["image_size"],
            normalizing=data_config["normalizing"],
            batch_size=data_config["batch_size"],
            output_type=(image_output_type, mask_output_type),
            one_hot_encoding=data_config["one_hot_encoding"],
            channels=data_config["channels"],
            palette=data_config["palette"],
            background_adding=data_config["background_adding"],
        )
        data_loader_list.append(test_data_loader)

    return data_loader_list


def _check_paths(image_paths: List[str], mask_paths: List[str], info: str) -> None:
    """
    Check if the number of image and mask paths are the same.

    Parameters
    ----------
    image_paths : List[str]
        A list of image paths.
    mask_paths : List[str]
        A list of mask paths.
    info : str
        A string to describe the data.

    Returns
    -------
        None

    """

    if len(image_paths) == 0:
        raise FileNotFoundError(f"{info} image paths {image_paths} is empty")
    if len(mask_paths) == 0:
        raise FileNotFoundError(f"{info} mask paths {mask_paths} is empty")

    assert len(image_paths) == len(mask_paths), f"Number of {info} images and masks are not equal {len(image_paths)} != {len(mask_paths)}"
    