from typing import Tuple, List
from tensorflow.keras.layers import Conv2DTranspose, Concatenate, Conv2D, BatchNormalization, Activation, MaxPooling2D, Input
from tensorflow.keras import Model


class Unet:
    def __init__(self, input_shape: Tuple = (512, 512, 3), n_filters: List[int] = [16, 32, 64, 128, 256], final_activation: str = "sigmoid"):
        self.input_shape = input_shape
        self.final_activation = final_activation
        self.n_filters = n_filters

    def build_model(self) -> Model:

        n_filters = self.n_filters
        inputs = Input(self.input_shape)

        c0 = inputs
        # Encoder
        c1, p1 = self._conv_block(c0, n_filters[0])
        c2, p2 = self._conv_block(p1, n_filters[1])
        c3, p3 = self._conv_block(p2, n_filters[2])
        c4, p4 = self._conv_block(p3, n_filters[3])

        # Bridge
        b1 = self._conv_block(p4, n_filters[4], pool=False)
        b2 = self._conv_block(b1, n_filters[4], pool=False)

        # Decoder
        d1 = Conv2DTranspose(n_filters[3], (3, 3), padding="same", strides=(2, 2))(b2)
        d1 = Concatenate()([d1, c4])
        d1 = self._conv_block(d1, n_filters[3], pool=False)

        d2 = Conv2DTranspose(n_filters[3], (3, 3), padding="same", strides=(2, 2))
        d2 = Concatenate()([d2, c3])
        d2 = self._conv_block(d2, n_filters[2], pool=False)

        d3 = Conv2DTranspose(n_filters[3], (3, 3), padding="same", strides=(2, 2))(d2)
        d3 = Concatenate()([d3, c2])
        d3 = self._conv_block(d3, n_filters[1], pool=False)

        d4 = Conv2DTranspose(n_filters[1], (3, 3), padding="same", strides=(2, 2))(d3)
        d4 = Concatenate()([d4, c1])
        d4 = self._conv_block(d4, n_filters[0], pool=False)

        # output
        outputs = Conv2D(1, (1, 1), activation=self.final_activation)(d4)

        # Model
        model = Model(inputs, outputs)
        return model

    def _conv_block(self, x, n_filter, pool=True):
        x = Conv2D(n_filter, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(n_filter, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        c = x

        if pool == True:
            x = MaxPooling2D((2, 2), (2, 2))(x)
            return c, x
        else:
            return c
