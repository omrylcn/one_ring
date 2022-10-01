from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, Concatenate, Layer, DepthwiseConv2D, GlobalAveragePooling2D, Reshape, UpSampling2D
from typing import List, Iterable
import numpy as np
import tensorflow as tf


class DeepLabConv(Layer):
    """
    DeepLabV3+ Convolution Layer with Batch Normalization and ReLU Activation

    """

    def __init__(self, n_filter: int, kernel_size: int, use_bias: bool = False, activation_name: str = "relu", name="deeplab_conv", **kwargs):
        super().__init__(name=name)
        self.n_filter = n_filter
        self.kernel_size = kernel_size
        self.activation_name = activation_name
        self.use_bias = use_bias

        self.conv = Conv2D(
            filters=self.n_filter,
            kernel_size=self.kernel_size,
            padding="same",
            use_bias=self.use_bias,
        )
        self.bn = BatchNormalization()
        self.activation = Activation(self.activation_name)

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.activation(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "n_filter": self.n_filter,
                "kernel_size": self.kernel_size,
                "activation_name": self.activation_name,
                "use_bias": self.use_bias,
            }
        )
        return config


class AtrousSepConv(Layer):
    """
    Atrous Separable Convolution Block, it consists of a Depthwise Convolution, Batch Normalization, ReLU Activation and a Pointwise Convolution
    """

    def __init__(self, n_filter: int, kernel_size: int, rate: int, use_bias: bool, activation_name: str = "relu", name: str = "atrous_conv", **kwargs):
        super().__init__(name=name)
        self.n_filter = n_filter
        self.kernel_size = kernel_size
        self.rate = rate
        self.activation_name = activation_name

        self.depth_conv = DepthwiseConv2D(kernel_size=self.kernel_size, padding="same", dilation_rate=self.rate)
        self.bn = BatchNormalization()
        self.activation = Activation(self.activation_name)
        self.point_conv = DeepLabConv(n_filter, kernel_size, use_bias, activation_name)

    def call(self, inputs, training=None):
        x = self.depth_conv(inputs)
        x = self.bn(x, training=training)
        x = self.activation(x)
        x = self.point_conv(x, training=training)

        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "n_filter": self.n_filter,
                "kernel_size": self.kernel_size,
                "rate": self.rate,
                "activation_name": self.activation_name,
            }
        )

        return config


class DeepLabPooling(Layer):
    """
    DeepLabV3+ Pooling Layer, it consists of a Global Average Pooling, Pointwise Convolution, and Upsampling
    """

    def __init__(self, n_filter: int, kernel_size: int = 1, upsampling_size: int = 4, use_bias: bool = False, activation_name: str = "relu", name="deeplab_conv", **kwargs):
        super().__init__(name=name)
        self.n_filter = n_filter
        self.kernel_size = kernel_size
        self.activation_name = activation_name
        self.use_bias = use_bias

        self.pooling = GlobalAveragePooling2D()
        self.point_conv = DeepLabConv(self.n_filter, self.kernel_size, self.use_bias, self.activation_name)

    def call(self, inputs, training=None):
        x = self.pooling(inputs)
        x = Reshape(target_shape=(1, 1, x.shape[-1]))(x)
        x = self.point_conv(x, training=training)
        x = UpSampling2D(size=inputs.shape[1:3], interpolation="bilinear")(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "n_filter": self.n_filter,
                "kernel_size": self.kernel_size,
                "activation_name": self.activation_name,
                "use_bias": self.use_bias,
            }
        )
        return config


class AtrousSpatialPyramidPooling(Layer):
    """
    Atrous Spatial Pyramid Pooling Layer, it consists of a 1x1 Convolution, 3 Atrous Separable Convolution Blocks, and a Pooling Layer
    """

    def __init__(self, atrous_rates: List[int] = [6, 12, 18], n_filter: int = 256, kernel_size: int = 3, activation_nam: str = "relu", use_bias: bool = False, name: str = "aspp", **kwargs):
        super().__init__(name=name)
        self.atrous_rates = atrous_rates
        self.n_filter = n_filter
        self.kernel_size = kernel_size
        self.activation_name = activation_nam
        self.use_bias = use_bias

        self.conv_k_one = DeepLabConv(n_filter, kernel_size=1, activation_name=activation_nam)
        self.atrous_sep_convs = []
        for rate in self.atrous_rates:
            self.atrous_sep_convs.append(
                AtrousSepConv(
                    n_filter=n_filter,
                    kernel_size=3,
                    rate=rate,
                    use_bias=False,
                    activation_name="relu",
                )
            )
        self.pooling = DeepLabPooling(n_filter, kernel_size=1, activation_name=activation_nam)

    def call(self, inputs, training=None):
        outputs_k_one = self.conv_k_one(inputs, training=training)
        atrous_sep_outputs = []
        for conv in self.atrous_sep_convs:
            atrous_sep_outputs.append(conv(inputs, training=training))
        outputs_pooling = self.pooling(inputs, training=training)

        # print(atrous_sep_outputs)
        return Concatenate()([outputs_k_one, *atrous_sep_outputs, outputs_pooling])

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "atrous_rates": self.atrous_rates,
                "n_filter": self.n_filter,
                "kernel_size": self.kernel_size,
                "activation_name": self.activation_name,
                "use_bias": self.use_bias,
            }
        )
        return config
