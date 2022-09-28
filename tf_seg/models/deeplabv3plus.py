from typing import Tuple, List
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tf_seg.backbones import get_backbone

import pandas as pd

pd.DataFrame()


class DeepLabV3Plus:
    """
    Model builder for DeepLabV3+ Model, originally is used xception backbone, but here is used different backbones.

    Parameters
    ----------
    output_size : int
        The output size of the model.
    final_activation : str
        Activation function to use for the final layer.
    name : str, optional
        The name of the model. The default is "deeplabv3+".
    input_shape : Tuple
        Shape of the input image.
    atrous_rates : List[int]
        Atrous rates for ASPP module.
    filters : int
        Number of filters in ASPP module.
    activation : str
        Activation function to use.
    backbone : str, default "ResNet50"
        Backbone to use.
    pretrained : str, default "imagenet"
        Whether to use pretrained weights.
    bakcbone_type : str, default: "deeplab", {"deeplab", classic"}.
        Type of backbone to use. if "deeplab"  bakcbone layers as activation layers.

    Notes
    -----
    DeepLabV3+ Article : https://arxiv.org/abs/1802.02611

    """

    def __init__(
        self,
        output_size: int,
        final_activation: str,
        name: str = "deeplabv3+",
        input_shape: Tuple = (512, 512, 3),
        atrous_rates: List[int] = [6, 12, 18],
        filters: int = 256,
        activation: str = "relu",
        backbone: str = "ResNet50",
        pretrained: str = "imagenet",
        bakcbone_type: str = "deeplab",
    ) -> None:

        self.output_size = output_size
        self.final_activation = final_activation
        self.name = name
        self.input_shape = input_shape
        self.atrous_rates = atrous_rates
        self.filters = filters
        self.activation_name = activation
        self.backbone = backbone
        self.pretrained = pretrained
        self.backbone_type = bakcbone_type

    def build_model(self) -> Model:
        inputs = Input(shape=self.input_shape)

        if self.backbone is not None:
            backbone = get_backbone(
                self.backbone,
                input_tensor=inputs,
                weights=self.pretrained,
                freeze_backbone=False,
                freeze_batch_norm=False,
                backbone_type=self.backbone_type,
                depth=5)
            x = backbone.output
        else:
            raise ValueError("Backbone is not defined. Please define a backbone.")

        return inputs,backbone #Model(inputs=inputs, outputs=x, name=self.name)
