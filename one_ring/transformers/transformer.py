from omegaconf import DictConfig
from typing import List, Dict, Any
import numpy as np
import tensorflow as tf
from one_ring.transformers.wrappers import AlbumentatiosWrapper
from one_ring.utils import TensorLike

wrapper_class_lib = {"albumentations": AlbumentatiosWrapper}


# TODO : design transformers type
class Transformer:
    """Transformer class for data augmentation using tensorflow dataset"""

    def __init__(self, config: DictConfig, name: str, transformers: Any = None, **kwargs):
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
        self.transformer_object = wrapper_class_lib[config["aug_type"]](self.transformer, **kwargs)

    @tf.function
    def __call__(self, image: tf.Tensor, mask: tf.Tensor) -> List[tf.Tensor]:
        """Apply augmentation to image and mask"""
        return self.transformer_object(image, mask)

    def transform(self, image: TensorLike, mask: TensorLike) -> Dict[str, np.ndarray]:
        """Apply augmentation to image and mask"""
        return self.transformer_object.transform(image=image, mask=mask)

    def save(self, path: str) -> None:
        """Save transformers"""
        self.transformer_object.save(path)

    def load(self, path: str) -> None:
        """Load transformers"""
        self.transformer_object.load(path)

        return self

    def to_dict(self) -> Dict:
        """Return transformers as dict"""
        return self.transformer_object.to_dict()

    def from_dict(self, data: Dict = None) -> None:
        """Load transformers from dict"""
        data = data if data else self.config[self.name]

        self.transformer_object.from_dict(data)

        return self

    def extract_params(self):
        """Return transformer object for only mlflow logging"""

        params = {}
        if self.config["aug_type"] == "albumentations":
            dict_params = self.to_dict()
            params = {}
            param = {}
            for t in dict_params["transform"]["transforms"]:
                for k, v in t.items():
                    if k == "__class_fullname__":
                        continue
                    param[k] = v
                params[t["__class_fullname__"]] = param

        return params
