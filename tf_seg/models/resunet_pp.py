# import tensorflow as tf
from typing import Tuple, List
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Reshape,
    Dense,
    Multiply,
    BatchNormalization,
    Conv2D,
    Activation,
    Add,
    MaxPooling2D,
    Input,
    Concatenate,
    UpSampling2D,
)
from tf_seg.base import ModelBuilder


class ResUnetPlusPlus(ModelBuilder):
    def __init__(
        self,
        input_shape: Tuple = (256, 256, 3),
        n_filters: List[int] = [16, 32, 64, 128, 256],
        attn_activation_name: str = "linear",
        name: str = "ResUnet++",
    ) -> None:
        self.input_shape = input_shape
        self.n_filters = n_filters
        self.name = name
        self.attn_activation_name = attn_activation_name

    def build_model(self) -> Model:
        """ """

        n_filters = self.n_filters
        input_shape = self.input_shape
        inputs = Input(input_shape)

        c0 = inputs
        c1 = self._stem_block(c0, n_filters[0], strides=1)

        # encoder
        c2 = self._resnet_block(c1, n_filters[1], strides=2)
        c3 = self._resnet_block(c2, n_filters[2], strides=2)
        c4 = self._resnet_block(c3, n_filters[3], strides=2)

        # bridge
        b1 = self._assp_block(c4, n_filter=n_filters[4])

        # decoder
        d1 = self._attention_block(g=c3, x=b1, activation=self.attn_activation_name)
        d1 = UpSampling2D(size=(2, 2))(d1)
        d1 = Concatenate()([d1, c3])
        d1 = self._resnet_block(d1, n_filters[3], strides=1)

        d2 = self._attention_block(g=c2, x=d1, activation=self.attn_activation_name)
        d2 = UpSampling2D(size=(2, 2))(d2)
        d2 = Concatenate()([d2, c2])
        d2 = self._resnet_block(d2, n_filters[2], strides=1)

        d3 = self._attention_block(g=c1, x=d2, activation=self.attn_activation_name)
        d3 = UpSampling2D(size=(2, 2))(d3)
        d3 = Concatenate()([d3, c1])
        d3 = self._resnet_block(d3, n_filters[1], strides=1)

        # output
        outputs = self._assp_block(d3, n_filters[0])
        outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(outputs)

        # model
        model = Model(inputs, outputs, name=self.name)
        return model

    def _resnet_block(self, x, n_filter, activation="relu", strides=1):

        x_init = x

        # Conv 1
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = Conv2D(n_filter, (3, 3), padding="same", strides=strides)(x)

        # Con 2
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = Conv2D(n_filter, (3, 3), padding="same")(x)

        # Shortcut
        s = Conv2D(n_filter, (1, 1), padding="same", strides=strides)(x_init)
        s = BatchNormalization()(s)

        # Add
        x = Add()([x, s])
        x = self._squeeze_excite_block(x)
        return x

    def _squeeze_excite_block(self, inputs, ratio=8):
        init = inputs
        channel_axis = -1
        filters = init.shape[channel_axis]
        se_shape = (1, 1, filters)

        se = GlobalAveragePooling2D()(init)
        se = Reshape(se_shape)(se)
        se = Dense(
            filters // ratio,
            activation="relu",
            kernel_initializer="he_normal",
            use_bias=False,
        )(se)
        se = Dense(
            filters,
            activation="sigmoid",
            kernel_initializer="he_normal",
            use_bias=False,
        )(se)
        x = Multiply()([init, se])
        return x

    def _stem_block(self, x, n_filter, strides):
        x_init = x

        # Conv 1
        x = Conv2D(n_filter, (3, 3), padding="same", strides=strides)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(n_filter, (3, 3), padding="same")(x)

        # Shortcut
        s = Conv2D(n_filter, (1, 1), padding="same", strides=strides)(x_init)
        s = BatchNormalization()(s)

        # Add
        x = Add()([x, s])
        x = self._squeeze_excite_block(x)
        return x

    def _assp_block(self, x, n_filter, rate_scale=1):
        x1 = Conv2D(
            n_filter,
            (3, 3),
            dilation_rate=(6 * rate_scale, 6 * rate_scale),
            padding="same",
        )(x)
        x1 = BatchNormalization()(x1)

        x2 = Conv2D(
            n_filter,
            (3, 3),
            dilation_rate=(12 * rate_scale, 12 * rate_scale),
            padding="same",
        )(x)
        x2 = BatchNormalization()(x2)

        x3 = Conv2D(
            n_filter,
            (3, 3),
            dilation_rate=(18 * rate_scale, 18 * rate_scale),
            padding="same",
        )(x)
        x3 = BatchNormalization()(x3)

        x4 = Conv2D(n_filter, (3, 3), padding="same")(x)
        x4 = BatchNormalization()(x4)

        y = Add()([x1, x2, x3, x4])
        y = Conv2D(n_filter, (1, 1), padding="same")(y)

        return y

    def _attention_block(self, g, x, activation="linear"):
        """
        g: Output of Parallel Encoder block
        x: Output of Previous Decoder block
        activation = last activation  layer type
        """

        filters = x.shape[-1]

        g_conv = BatchNormalization()(g)
        g_conv = Activation("relu")(g_conv)
        g_conv = Conv2D(filters, (3, 3), padding="same")(g_conv)

        g_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(g_conv)

        x_conv = BatchNormalization()(x)
        x_conv = Activation("relu")(x_conv)
        x_conv = Conv2D(filters, (3, 3), padding="same")(x_conv)

        gc_sum = Add()([g_pool, x_conv])

        # print("gc_sum shape:", gc_sum.shape)

        gc_conv = BatchNormalization()(gc_sum)
        gc_conv = Activation("relu")(gc_conv)
        gc_conv = Conv2D(filters, (3, 3), padding="same")(gc_conv)

        # last activation
        gc_conv = Activation(activation)(gc_conv)
        gc_mul = Multiply()([gc_conv, x])

        return gc_mul
