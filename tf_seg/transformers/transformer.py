from omegaconf import DictConfig
from typing import List, Dict
import numpy as np
import tensorflow as tf
from tf_seg.transformers.wrappers import AlbumentatiosWrapper
from tf_seg.utils import TensorLike

wrapper_class_lib = {"albumentations": AlbumentatiosWrapper}


# TODO : design transformers type
class Transformer:
    """Transformer class for data augmentation"""

    def __init__(self, config: DictConfig, name: str, transformers, **kwargs):
        """
        It is a adapter class for data augmentation to choose a wrapper class

        Parameters
        ----------
        config : DictConfig
            config file
        transformers : dict
            a dict with transformers
        """
        self.config = config
        self.transformer = transformers
        self.name = name
        self.transformer_object = wrapper_class_lib[config["aug_type"]](
            self.transformer, **kwargs
        )

    @tf.function
    def __call__(self, image: tf.Tensor, mask: tf.Tensor) -> List[tf.Tensor]:
        """Apply augmentation to image and mask"""
        return self.transformer_object(image, mask)

    def transform(self, image: TensorLike, mask: TensorLike) -> Dict[str, np.ndarray]:
        """Apply augmentation to image and mask"""
        return self.transformer_object.transform(image=image, mask=mask)
