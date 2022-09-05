import os
import importlib
import tensorflow as tf
from typing import Union
from omegaconf import DictConfig, ListConfig





def load_module_style_transformer(config: Union[DictConfig, ListConfig]):
    """Load python module style transformer"""

    transformer_lib = {}

    for p in ["train", "val", "test"]:

        string = config[p].path

        if string:

            parameters = config[p].parameters
            if parameters is None:
                parameters = {}

            module = string.split(":")[0]
            function_name = string.split(":")[1]
            module_file_path = module.replace(".", "/") + ".py"
            assert os.path.isfile(module_file_path), f"{module_file_path} module file not found"

            # load module
            m = importlib.import_module(module)
            transformer = getattr(m, function_name)(**parameters)
            transformer_lib[p] = transformer

        else:
            transformer_lib[p] = None

    return transformer_lib


def load_file_style_transformer():
    raise NotImplementedError()
