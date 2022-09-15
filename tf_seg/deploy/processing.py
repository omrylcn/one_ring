import importlib
import numpy as np
import tensorflow as tf
from typing import Dict, Union
from omegaconf import DictConfig, ListConfig
from tf_seg.utils import TensorLike


# def load_preprocessor(config: Union[Dict, DictConfig, ListConfig]):
#     pass


# def load_postprocessor(config: Union[Dict, DictConfig, ListConfig]):
#     pass


def get_test_transform_from_string(string: str):
    module = string.split(":")[0]
    function_name = string.split(":")[1]
    #module_file_path = module.replace(".", "/") + ".py"
    m = importlib.import_module(module)
    transformer = getattr(m, function_name)

    return transformer


# Preprocessors
class AlbumentationsPreprocessor:
    def __init__(self, config: Union[Dict, DictConfig, ListConfig])->None:
        #self.config = config
        #print("pre",config)
        if config["preprocessor_load_style"] == "module":
            self.transform = get_test_transform_from_string(config["preprocessor_path"])(image_size=config["image_size"])
        else:
            raise ValueError(f"preprocessor_load_style not supported {config['preprocessor_load_style']}")

    def __call__(self, image: TensorLike)->np.ndarray:
        
        if isinstance(image, np.ndarray):
            pass
            
        elif isinstance(image, tf.Tensor):
            image = image.numpy()
        
        else:
            raise ValueError(f"image type is not supported , {type(image)} ")
          
        return self.transform(image=image)["image"]
        
class TensorFlowPreprocessor:
    def __init__(self, config: Union[Dict, DictConfig, ListConfig]):
        self.config = config

    def __call__(self, image):
        pass


# Postprocessor
class VanillaPostprocessor:
    def __init__(self, config: Union[Dict, DictConfig, ListConfig]):
        self.config = config

    def __call__(self, image):
        pass


preprocessor_lib = {"albumentations": AlbumentationsPreprocessor, "tensorflow": TensorFlowPreprocessor}

postprocessor_lib = {"vanilla": VanillaPostprocessor}
