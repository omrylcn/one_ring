from pathlib import Path
from typing import List, Optional, Union
from warnings import warn
from omegaconf import DictConfig, ListConfig, OmegaConf


def get_config(
    config_filename: Optional[str] = None, config_path: Optional[Union[Path, str]] = None, config_file_extension: Optional[str] = CONFIG_FILE_EXTENSION, config_store_path: "str" = CONFIG_STORE_PATH
) -> Union[DictConfig, ListConfig]:

    """
    Get configurable parameters from config file.

    Parameters
    ----------
    config_filename : str, optional
        Name of the config file. The default is None.
    config_path : Union[Path, str], optional
        Path of the config file. The default is None.
    config_file_extension : str, optional
        File extension of the config file. The default is ".yaml".
    config_store_path : str
        Path of the config store. The default is "./config".

    Returns
    -------
    config : Union[DictConfig, ListConfig]
        Configurable parameters.

    """
    assert os.path.isdir(config_store_path), f"{config_store_path} is not a directory"

    if config_path is None:
        config_path = Path(f"{config_store_path}/{config_name}{config_file_extension}")

    config = OmegaConf.load(config_path)

    return config
