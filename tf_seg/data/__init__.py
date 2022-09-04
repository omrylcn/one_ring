from typing import Union, Tuple
from omegaconf import DictConfig, ListConfig
from tf_seg.data.data import DataLoader
from tf_seg.data.loader_func import get_camvid_data_loader, get_custom_data_loader


# data loader function library
loader_lib = {"camvid": get_camvid_data_loader, "custom": get_custom_data_loader}


def get_data_loader(config: Union[dict, DictConfig, ListConfig], train_data: bool, val_data: bool, test_data: bool) -> Tuple[DataLoader]:
    """Get data loader from config file"""
    function_name = config.function_name
    assert function_name in loader_lib.keys(), f"{function_name} is not in loader_lib"
    loader_function = loader_lib[function_name]

    data_config = config.copy()
    data_config.pop("function_name")

    loader_tuple = loader_function(data_config, train_data, val_data, test_data)

    return loader_tuple
