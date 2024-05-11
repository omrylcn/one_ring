from typing import Tuple, List
from tensorflow.keras.layers import (
    Conv2DTranspose,
    Concatenate,
    Conv2D,
    Input,
)
from tensorflow.keras import Model

from one_ring.backbones import get_backbone
from one_ring.layers import ConvUnet as ConvBlock, AttentionGate
from one_ring.base import ModelBuilder


class AttUnet(ModelBuilder):
    """
    Model builder for Attention U-Net Model

    Parameters
    ----------
    output_size : int
        The output size of the model. Typically, this would be the number of classes for segmentation.
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
    backbone_name : str
        The name of the backbone architecture to use. This backbone will be a pre-trained model and is optional.
    pretrained : str
        Whether to use pretrained weights for the backbone. Can be either "imagenet" for pretrained imagenet weights, or None for random initialization.
    freeze_backbone : bool
        Whether to freeze the weights of the backbone model. If True, the weights of the backbone model will not be updated during training.
        
    Notes
    -----
    Unet with Attention Mechanism combines the powerful U-Net architecture with attention gates, 
    allowing the model to focus on salient features and ignore irrelevant regions, 
    potentially improving performance on tasks where certain areas of the image are more important than others.

    This can be especially useful in applications such as medical imaging, where the area of interest 
    might be a small part of the overall image. The attention mechanism can focus more on pathological 
    regions and less on normal tissue.

    Reference U-Net Article : https://arxiv.org/abs/1505.04597
    Reference to Attention Gate Mechanism : https://arxiv.org/abs/1804.03999
    
    """

    def __init__(
        self,
        output_size: int,
        name: str = "unet",
        input_shape: Tuple = (224, 224, 3),
        n_filters: List[int] = [16, 32, 64, 128, 256],
        activation: str = "relu",
        final_activation: str = "sigmoid",
        backbone_name: str = None,
        pretrained: str = "imagenet",
        freeze_backbone: bool = False,
        **kwargs,
    ) -> None:
        """Unet constructor."""

        self.input_shape = input_shape
        self.final_activation = final_activation
        self.n_filters = n_filters
        self.activation_name = activation
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.name = name
        self.output_size = output_size
        self.freeze_backbone = freeze_backbone

    def build_model(self) -> Model:
        """Builds the model.

        Returns
        -------
        Model : tf.keras.model.Model
            The model.

        """

        n_filters = self.n_filters

        if self.backbone_name is not None:
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
        decoder_n_filters = n_filters[:][::-1]

        # [print(i.shape) for i in encoder_output]

        # Decoder
        d = bridge
        for n, c in enumerate(connection_list):
            d = Conv2DTranspose(decoder_n_filters[n], (3, 3), padding="same", strides=(2, 2))(d)
            # print(decoder_n_filters[n])
            # print(d.shape,c.shape)

            c = AttentionGate(c.shape[-1], name=f"attention_{decoder_n_filters[n]}")([d, c])

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
                d = ConvBlock(fltr, self.activation_name, name=f"decode_{fltr}")(d, pool=False)

        # output
        outputs = Conv2D(self.output_size, (1, 1), activation=self.final_activation)(d)
        # Model
        model = Model(inputs, outputs, name=self.name)

        return model

    def _build_encoder(self):

        inputs = Input(self.input_shape, name="input")
        c0 = inputs

        if self.backbone_name is None:
            n_filters = self.n_filters
            connection_list = []
            p = c0
            for fltr in n_filters[:-1]:
                # c, p = self._conv_block(p, fltr)
                c, p = ConvBlock(fltr, self.activation_name, name="conv_block_{}".format(fltr))(p)
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
            self.backbone = get_backbone(self.backbone_name, self.pretrained, inputs, depth=len(self.n_filters),freeze_backbone=self.freeze_backbone)

            return [
                c0,
                *self.backbone(c0),
            ]  #
