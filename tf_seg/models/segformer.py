import os
from typing import Tuple, List, Dict, Any, Union, Optional

import tensorflow as tf
from tensorflow.keras.models import Model

from tf_seg.base import ModelBuilder


class SegFormer(ModelBuilder):
    """A model builder for SegFormer semantic segmentation model."""

    def __init__(
        self,
        backbone: str,
        input_shape: Tuple[int],
        output_size: int,
        softmax_output: bool = True,
    ):
        """Initializes the model builder.
        
        Parameters
        ----------
        input_shape : Tuple
            Shape of the input image.
        output_size : int
            The size of the output, typically this would be the number of classes for segmentation.
        backbone : str
            Which model variant to use , defaults to "nvidia/mit-b0".


        Methods
        -------
        build_model(self)
            Builds and returns the model.

        Notes
        -----
        SegFormer Article: https://arxiv.org/abs/2105.15203



        """
        self.backbone = backbone
        self.input_shape = input_shape
        self.output_size = output_size
        self.softmax_output = softmax_output

    def build_model():
        """Builds the model.

        Returns
        -------
        Model : tf.keras.model.Model
            The model.

        """
        pass
