from typing import Union
from omegaconf import DictConfig, ListConfig
from tf_seg.utils import load_module_style_transformer, load_file_style_transformer

#from tf_seg.transformers.wrappers import AlbumentatiosWrapper
from tf_seg.transformers.transformer import Transformer


# def load_file_style_transformer():
#     pass

# def load_module_style_transformer():
#     pass

load_style_lib = {"module": load_module_style_transformer, "file": load_file_style_transformer}


# TODO : design transformers type
def get_transformer(config: Union[DictConfig, ListConfig]) -> dict:
    """
    Get data augmentation transformers from config file.

    Returns
    -------
    transformer_lib : dict
        a dict with transformers

    """

    load_style = config.augmentation.load_style
    aug_config = config.augmentation.copy()

    if load_style == "module":
        transformer_lib = load_module_style_transformer(aug_config)
    elif load_style == "file":
        raise NotImplementedError(f"Not implemented  : {load_style}")
        # transformer_lib = load_file_style_transformer(aug_config)
    else:
        raise NotImplementedError(f"Unvalid style : {load_style}")

    transformers_object_lib = {}
    for k,v in transformer_lib.items():
        if v is not None:
            transformers_object_lib[k] = Transformer(aug_config, k, v)   
        else:
            transformers_object_lib[k] = None
    
    return transformers_object_lib
  