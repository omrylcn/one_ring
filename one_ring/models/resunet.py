"""
ResUNet architecture in Keras TensorFlow
"""

import os
import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    Activation,
    Add,
    MaxPooling2D,
    Conv2DTranspose,
    Concatenate,
)
from tensorflow.keras.models import Model
from typing import Tuple, List, Dict, Any, Union, Optional
from one_ring.base import ModelBuilder


class ResUnet(ModelBuilder):
    """A model builder for Deep Residual U-Net Model."""

    def __init__(
        self,
        output_size: int,
        input_shape: Tuple = (224, 224, 3),
        n_filters: List[int] = [16, 32, 64, 96, 128],
        activation: str = "relu",
        final_activation: str = "sigmoid",
        backbone: str = None,
        pretrained: str = "imagenet",
    ) -> None:
        """Initializes the model builder.

        Parameters
        ----------
        input_shape : Tuple
            Shape of the input image.
        n_filters : List[int]
            Number of filters in each convolutional layer.
        activation : str
            Activation function to use.
        final_activation : str
            Activation function to use for the final layer.
        backbone : str
            Backbone to use.
        pretrained : str
            Which pretrained model to use, default "imagenet".

        Notes
        -----
        Deep Residual Unet Article : https://arxiv.org/pdf/1711.10684.pdf

        """

        self.input_shape = input_shape
        self.final_activation = final_activation
        self.n_filters = n_filters
        self.activation_name = activation
        self.backbone = backbone
        self.pretrained = pretrained
        self.output_size = output_size

    def build_model(self):
        """Builds the model.

        Returns
        -------
        Model : tf.keras.model.Model
            The model.

        """

        n_filters = self.n_filters
        inputs = Input(self.input_shape)

        c0 = inputs
        # Encoder
        c1, p1 = self._resnet_block(c0, n_filters[0])

        c2, p2 = self._resnet_block(p1, n_filters[1])
        c3, p3 = self._resnet_block(p2, n_filters[2])
        c4, p4 = self._resnet_block(p3, n_filters[3])

        # Bridge
        b1 = self._resnet_block(p4, n_filters[4], pool=False)
        b2 = self._resnet_block(b1, n_filters[4], pool=False)

        # Decoder
        d1 = Conv2DTranspose(n_filters[3], (3, 3), padding="same", strides=(2, 2))(b2)
        # d1 = UpSampling2D((2, 2))(b2)
        d1 = Concatenate()([d1, c4])
        d1 = self._resnet_block(d1, n_filters[3], pool=False)

        d2 = Conv2DTranspose(n_filters[3], (3, 3), padding="same", strides=(2, 2))(d1)
        # # d2 = UpSampling2D((2, 2))(d1)
        d2 = Concatenate()([d2, c3])
        d2 = self._resnet_block(d2, n_filters[2], pool=False)

        d3 = Conv2DTranspose(n_filters[3], (3, 3), padding="same", strides=(2, 2))(d2)
        # d3 = UpSampling2D((2, 2))(d2)
        d3 = Concatenate()([d3, c2])
        d3 = self._resnet_block(d3, n_filters[1], pool=False)

        d4 = Conv2DTranspose(n_filters[3], (3, 3), padding="same", strides=(2, 2))(d3)
        # d4 = UpSampling2D((2, 2))(d3)
        d4 = Concatenate()([d4, c1])
        d4 = self._resnet_block(d4, n_filters[0], pool=False)

        # output
        outputs = Conv2D(self.output_size, (1, 1), padding="same")(d4)
        outputs = BatchNormalization()(outputs)
        outputs = Activation("sigmoid")(outputs)

        # Model
        model = Model(inputs, outputs)
        return model

    def _conv_block(self, x, n_filter):
        x_init = x

        # Conv 1
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(n_filter, (1, 1), padding="same")(x)
        # Conv 2
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(n_filter, (3, 3), padding="same")(x)
        # Conv 3
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(n_filter, (1, 1), padding="same")(x)

        # Shortcut
        s = Conv2D(n_filter, (1, 1), padding="same")(x_init)
        s = BatchNormalization()(s)

        # Add
        x = Add()([x, s])
        return x

    def _resnet_block(self, x, n_filter, pool=True):
        x = self._conv_block(x, n_filter)
        c = x

        # Pooling
        if pool is True:
            x = MaxPooling2D((2, 2), (2, 2))(x)
            return c, x
        else:
            return c
