from typing import Union
from tensorflow.keras.models import Model
from omegaconf import DictConfig, ListConfig

from tf_seg.models.unet import Unet
from tf_seg.models.resunet import ResUnet
from tf_seg.models.resunet_pp import ResUnetPlusPlus
from tf_seg.models.deeplabv3plus import DeepLabV3Plus


model_lib = {"unet": Unet, "resunet": ResUnet, "resunet_pp": ResUnetPlusPlus, "deeplabv3plus": DeepLabV3Plus}


def pascal_case_to_snake_case(s: str) -> str:
    """Convert class name to snake case name."""
    return "".join(["_" + i.lower() if i.isupper() else i for i in s]).lstrip("_")


def get_model_builder(config: Union[DictConfig, ListConfig]) -> Model:
    """Get model builder  from config file"""
    class_name = pascal_case_to_snake_case(config.class_name)
    model = model_lib[class_name]

    model_config = config.copy()
    model_config.pop("class_name")

    return model(**model_config)
