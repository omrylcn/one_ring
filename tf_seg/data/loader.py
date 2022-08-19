import os
from glob import glob
from typing import Union
from omegaconf import DictConfig, ListConfig
import tensorflow as tf

from tf_seg.data import DataLoader


def get_camvid_data_loader(data_config: Union[dict, DictConfig, ListConfig], train_data: bool = True, val_data: bool = True, test_data: bool = True) -> tf.data.Dataset:

    """
    Custom data loader function for camvid dataset. A data loader is a class to load tf.data.Dataset objects from given data path

    """

    image_output_type = getattr(tf, data_config["output_type"][0].split(".")[-1])  # objectinput must be like tf.float32 string
    mask_output_type = getattr(tf, data_config["output_type"][1].split(".")[-1])

    data_loader_list = []
    if train_data:
        train_image_paths = sorted(glob(os.path.join(data_config["data_path"], "train/*.png")))
        train_mask_paths = sorted(glob(os.path.join(data_config["data_path"], "train_labels/*.png")))

        train_data_loader = DataLoader(
            train_image_paths,
            train_mask_paths,
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
        val_image_paths = sorted(glob(os.path.join(data_config["data_path"], "val/*.png")))
        val_mask_paths = sorted(glob(os.path.join(data_config["data_path"], "val_labels/*.png")))

        val_data_loader = DataLoader(
            val_image_paths,
            val_mask_paths,
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
        test_image_paths = sorted(glob(os.path.join(data_config["data_path"], "test/*.png")))
        test_mask_paths = sorted(glob(os.path.join(data_config["data_path"], "test_labels/*.png")))

        test_data_loader = DataLoader(
            test_image_paths,
            test_mask_paths,
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

    return tuple(data_loader_list)
