import importlib
import numpy as np
import tensorflow as tf
from typing import Dict, Union
from omegaconf import DictConfig, ListConfig
from one_ring.utils import TensorLike


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
        self.transform = get_test_transform_from_string(config["preprocessor_path"])(image_size=config["image_size"])
        
    def __call__(self, image: TensorLike)->np.ndarray:
        
        if isinstance(image, np.ndarray):
            pass
            
        elif isinstance(image, tf.Tensor):
            image = image.numpy()
        
        else:
            raise ValueError(f"image type is not supported , {type(image)} ")
        
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
          
        return self.transform(image=image)["image"]


class TensorFlowPreprocessor:
    """
    Not implemented yet. This is for preprocessing images using tensorflow functions.

    """
    def __init__(self, config: Union[Dict, DictConfig, ListConfig]):
        self.config = config

    def __call__(self, image):
        
        return image


# Postprocessor
class VanillaPostprocessor:
    """
    Standard postprocessing for segmentation models. It return binary mask and  pred image
    """
    def __init__(self, config: Union[Dict, DictConfig, ListConfig]):
        self.config = config
        self.threshold = config["threshold"]

    def __call__(self, image: TensorLike)->tuple:

        return image > self.threshold, image


preprocessor_lib = {"albumentations": AlbumentationsPreprocessor, "tensorflow": TensorFlowPreprocessor}

postprocessor_lib = {"vanilla": VanillaPostprocessor}
