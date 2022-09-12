from typing import Tuple, List
from tensorflow.keras.layers import (
    Layer,
    Conv2DTranspose,
    Concatenate,
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    Input,
)
from tensorflow.keras import Model
from tf_seg.backbones import get_backbone
from tf_seg.layers import ConvUnet as ConvBlock
from tf_seg.base import ModelBuilder

# TODO: add backbone and pretrained weights
class Unet(ModelBuilder):
    """Model builder for Vanialla UNet Model"""

    def __init__(
        self,
        output_size: int,
        name: str = "unet",
        input_shape: Tuple = (512, 512, 3),
        n_filters: List[int] = [16, 32, 64, 128, 256],
        activation: str = "relu",
        final_activation: str = "sigmoid",
        backbone: str = None,
        pretrained: str = "imagenet",
    ) -> None:
        """Unet constructor.

        Parameters
        ----------
        output_size : int
            The output size of the model.
        name : str, optional
            The name of the model. The default is "unet".
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
        pretrained : bool
            Whether to use pretrained weights.

        Notes
        -----
        Unet Article : https://arxiv.org/abs/1505.04597

        """

        self.input_shape = input_shape
        self.final_activation = final_activation
        self.n_filters = n_filters
        self.activation_name = activation
        self.backbone = backbone
        self.pretrained = pretrained
        self.name = name
        self.output_size = output_size

    def build_model(self) -> Model:
        """Builds the model.

        Returns
        -------
        Model : tf.keras.model.Model
            The model.

        """

        n_filters = self.n_filters

        if self.backbone is not None:
            if len(n_filters) >= 7:
                raise ValueError("If select backbone, n_filters must be lesser than 7")

            elif len(n_filters) < 3:
                raise ValueError("If select backbone, n_filters must be greater than 2")
            else:
                pass

        encoder_output = self._build_encoder()
        inputs = encoder_output[0]
        bridge = encoder_output[-1]
        connection_list = encoder_output[1:-1][::-1]
        decoder_n_filters = n_filters[:-1][::-1]

        # Decoder
        d = bridge
        for n, c in enumerate(connection_list):
            d = Conv2DTranspose(
                decoder_n_filters[n], (3, 3), padding="same", strides=(2, 2)
            )(d)
            d = Concatenate()([d, c])
            d = ConvBlock(
                decoder_n_filters[n],
                self.activation_name,
                name=f"decode_{decoder_n_filters[n]}",
            )(d, pool=False)

        # if connection count is lesser decoder filter count , we make convolution without concatenation
        depth = len(connection_list)
        if len(decoder_n_filters) > depth:
            n_remain_block = len(decoder_n_filters) - depth
            remain_decoder_n_filter = decoder_n_filters[(-1 * n_remain_block) :]
            for fltr in remain_decoder_n_filter:
                d = Conv2DTranspose(fltr, (3, 3), padding="same", strides=(2, 2))(d)
                d = ConvBlock(fltr, self.activation_name, name=f"decode_{fltr}")(
                    d, pool=False
                )

        # output
        outputs = Conv2D(self.output_size, (1, 1), activation=self.final_activation)(d)
        # Model
        model = Model(inputs, outputs, name=self.name)

        return model

    def _build_encoder(self):

        inputs = Input(self.input_shape, name="input")
        c0 = inputs

        if self.backbone is None:
            n_filters = self.n_filters
            connection_list = []
            p = c0
            for fltr in n_filters[:-1]:
                # c, p = self._conv_block(p, fltr)
                c, p = ConvBlock(
                    fltr, self.activation_name, name="conv_block_{}".format(fltr)
                )(p)
                connection_list.append(c)

            # Bridge
            b1 = ConvBlock(
                n_filters[-1],
                self.activation_name,
                name="bridge_1_{}".format(n_filters[-1]),
            )(p, pool=False)
            b2 = ConvBlock(
                n_filters[-1],
                self.activation_name,
                name="bridge_2_{}".format(n_filters[-1]),
            )(b1, pool=False)

            return [c0, *connection_list, b2]

        else:
            backbone_ = get_backbone(
                self.backbone, self.pretrained, inputs, len(self.n_filters) - 1
            )

            return [
                c0,
                *backbone_(c0),
            ]  # backbone_#backbone_(c0)#[c0,*backbone_(c0)]#*outputs.outputs]

    # def _conv_block(self, x, n_filter, pool=True):
    #     x = Conv2D(n_filter, (3, 3), padding="same")(x)
    #     x = BatchNormalization()(x)
    #     x = Activation(self.activation_name)(x)

    #     x = Conv2D(n_filter, (3, 3), padding="same")(x)
    #     x = BatchNormalization()(x)
    #     x = Activation(self.activation_name)(x)
    #     c = x

    #     if pool == True:
    #         x = MaxPooling2D((2, 2), (2, 2))(x)
    #         return c, x
    #     else:
    #         return c
