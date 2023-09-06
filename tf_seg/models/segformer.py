from typing import Tuple, List, Dict, Any, Union, Optional
from transformers import TFSegformerForSemanticSegmentation

from tensorflow.keras import backend
from tensorflow.keras.models import Model
from tf_seg.base import ModelBuilder

class SegFormer(ModelBuilder):
    def __init__(
        self,
        output_size: int,
        input_shape: Tuple = (512, 512,3),
        pretrained: str = "nvidia/mit-b0",
    ) -> None:
    """Initializes the model builder.
    
    Parameters
    ----------
    input_shape : Tuple
        Shape of the input image.
    output_size : int
        The size of the output, typically this would be the number of classes for segmentation.
    pretrained : str
        Which pretrained model to use, defaults to "nvidia/mit-b0".

    Notes
    -----
    SegFormer Article : https://arxiv.org/abs/2105.15203

    """
    self.output_size = output_size
    self.input_shape = input_shape
    self.pretrained = pretrained

    def build_model(self):
        pass

    
