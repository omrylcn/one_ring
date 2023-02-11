import os
import logging
from pathlib import Path
from typing import Optional, Union
from omegaconf import DictConfig, ListConfig, OmegaConf

# from warnings import warn
from tf_seg.config import CONFIG_FILE_EXTENSION, CONFIG_STORE_PATH
from tf_seg.config import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.INFO)


def get_config(
    config_filename: str,
    config_path: Optional[Union[Path, str]] = None,
    config_file_extension: Optional[str] = CONFIG_FILE_EXTENSION,
    config_store_path: Union[Path, str] = CONFIG_STORE_PATH,
) -> Union[DictConfig, ListConfig]:

    """
    Get configurable parameters from config file.

    Parameters
    ----------
    config_filename : str, optional
        Name of the config file.
    config_path : Union[Path, str], optional
        Path of the config file. The default is None.
    config_file_extension : str, optional
        File extension of the config file. The default is ".yaml".
    config_store_path : str
        Path of the config store.

    Returns
    -------
    config : Union[DictConfig, ListConfig]
        Configurable parameters.

    """

    assert os.path.isdir(config_store_path), f"{config_store_path} is not a directory"

    if config_path is None:
        config_path = Path(
            f"{config_store_path}/{config_filename}{config_file_extension}"
        )

    logger.info(f"Loading config from {config_path}")
    config = OmegaConf.load(config_path)
    try:
        config = extact_config(config)
    except Exception as e:
        raise Exception(f"{e} running extact_config() failed")

    return config


def extact_config(
    config: Union[DictConfig, ListConfig]
) -> Union[DictConfig, ListConfig]:
    """ "Extract parent base config and merge with child configs"""

    first_config_keys = list(config.keys())
    f_config = config.copy()

    for first_k in first_config_keys:

        sub_config = f_config[first_k]

        if check_base_config_exist(sub_config):
            buffer_config = load_base_config(sub_config)
            sub_config.pop("base")
            buffer_config.merge_with(sub_config)
            sub_config = buffer_config

        else:
            if "base" in sub_config.keys():
                sub_config.pop("base")

        f_config[first_k] = sub_config

    return f_config


def check_base_config_exist(config: Union[DictConfig, ListConfig]) -> bool:
    """check if base config exist in config file"""

    if "base" in config.keys():
        if config["base"] is None:
            return False
        else:
            return True
    else:
        return False


def load_yaml_style_config(yaml_path: str) -> Union[DictConfig, ListConfig]:
    """load config from yaml file"""
    assert os.path.isfile(yaml_path), f"{yaml_path} is not a file"
    config = OmegaConf.load(yaml_path)
    return config


def load_module_style_config(module_path: str) -> Union[DictConfig, ListConfig]:
    """load config from python module"""
    raise NotImplementedError("load_module_style_config is not implemented")


def load_base_config(config):
    """load base config from base parameter"""

    # find file extension
    load_style = os.path.splitext(config["base"])[1]

    if load_style == ".py":  # module style
        raise NotImplementedError("load_style is not implemented")

    elif load_style == ".yaml" or load_style == ".yml":
        return load_yaml_style_config(config["base"])

    else:
        raise ValueError("load_style is not valid")
