from typing import Tuple, List
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, UpSampling2D, Concatenate, Conv2D,Activation
from one_ring.backbones import get_backbone
from one_ring.base.model_builder import ModelBuilder
from one_ring.layers import AtrousSpatialPyramidPooling, DeepLabConv



class DeepLabV3Plus(ModelBuilder):
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
    backbone_outputs_order : List[int], default: [1,-2]
        Order of backbone outputs to use. first value is low level features, last value is last layer of backbone.

    Notes
    -----
    DeepLabV3+ Article : https://arxiv.org/abs/1802.02611

    """

    def __init__(
        self,
        output_size: int,
        final_activation: str = "sigmoid",
        name: str = "deeplabv3plus",
        input_shape: Tuple = (224, 224, 3),
        atrous_rates: List[int] = [6, 12, 18],
        n_filters: int = 256,
        activation: str = "relu",
        backbone_name: str = "ResNet50",
        pretrained: str = "imagenet",
        backbone_outputs_order: List[int] = [1, -2],
        freeze_backbone: bool = True,
        freeze_batch_norm: bool = False
        # bakcbone_type: str = "deeplab",
    ) -> None:
        super().__init__()

        self.output_size = output_size
        self.final_activation = final_activation
        self.name = name
        self.input_shape = input_shape
        self.atrous_rates = atrous_rates
        self.filters = n_filters
        self.activation_name = activation
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.backbone_outputs_order = backbone_outputs_order
        self.freeze_backbone = freeze_backbone
        self.freeze_batch_norm = freeze_batch_norm
        # self.backbone_type = bakcbone_type
        print(self.filters)
        if type(self.filters) == list:
            self.filters = self.filters[-1]

        self.conv_low_level_features = DeepLabConv(n_filter=48, kernel_size=1, name="conv_low_level_features")
        self.conv1 = DeepLabConv(n_filter=self.filters, kernel_size=1, name="conv1")
        self.aspp = AtrousSpatialPyramidPooling(atrous_rates, n_filter=self.filters)
        self.up_sampling1 = UpSampling2D(size=(4, 4), interpolation="bilinear")
        self.concat = Concatenate(axis=-1)
        self.conv2 = DeepLabConv(n_filter=self.filters, kernel_size=3, name="conv2")
        self.conv3 = DeepLabConv(n_filter=self.filters, kernel_size=3, name="conv3")
        self.up_sampling2 = UpSampling2D(size=(4, 4), interpolation="bilinear")
        self.final_layer = Conv2D(output_size, kernel_size=1, activation=final_activation, padding="same", name="output")
        
    def build_model(self) -> Model:
        inputs = Input(shape=self.input_shape, name="input")
        if self.backbone_name is not None:
            self.backbone = get_backbone(
                self.backbone_name,
                input_tensor=inputs,
                weights=self.pretrained,
                freeze_backbone=self.freeze_backbone,
                freeze_batch_norm=self.freeze_batch_norm,
                depth=5,
                backbone_type="deeplabv3_plus",
                outputs_order=self.backbone_outputs_order,
            )
            backbone_outputs = self.backbone(inputs)
        else:
            raise ValueError("Backbone is not defined. Please define a backbone.")

        low_level_features = backbone_outputs[0]  # self.bakcbone_outputs_order[0]]
        low_level_features = self.conv_low_level_features(low_level_features)

        # x = self.backbone_conv(x) #depthwise conv
        x = backbone_outputs[1]  # [self.bakcbone_outputs_order[1]]
       # print(x.shape,low_level_features.shape)
        x = self.aspp(x)
        x = self.conv1(x)
        x = self.up_sampling1(x)

        x = self.concat([x, low_level_features])
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.up_sampling2(x)
        x = self.final_layer(x)

        return Model(inputs=inputs, outputs=x, name=self.name)

        # return Model(inputs=inputs, outputs=x, name=self.name)
