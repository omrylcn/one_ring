"""
TO check reference:
https://github.com/yingkaisha/keras-unet-collection/blob/main/keras_unet_collection/_backbone_zoo.py

"""

from tensorflow.keras.applications import *
from tensorflow.keras.models import Model
# from keras_unet_collection.utils import freeze_model
from tensorflow.python.keras.engine import keras_tensor


layer_cadidates = {
    "ResNet50": ("conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"),
    "ResNet101": ("conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block23_out", "conv5_block3_out"),
    "ResNet152": ("conv1_relu", "conv2_block3_out", "conv3_block8_out", "conv4_block36_out", "conv5_block3_out"),
    "ResNet50V2": ("conv1_conv", "conv2_block3_1_relu", "conv3_block4_1_relu", "conv4_block6_1_relu", "post_relu"),
    "ResNet101V2": ("conv1_conv", "conv2_block3_1_relu", "conv3_block4_1_relu", "conv4_block23_1_relu", "post_relu"),
    "ResNet152V2": ("conv1_conv", "conv2_block3_1_relu", "conv3_block8_1_relu", "conv4_block36_1_relu", "post_relu"),
    "EfficientNetB0": ("block2a_expand_activation", "block3a_expand_activation", "block4a_expand_activation", "block6a_expand_activation", "top_activation"),
    "EfficientNetB1": ("block2a_expand_activation", "block3a_expand_activation", "block4a_expand_activation", "block6a_expand_activation", "top_activation"),
    "EfficientNetB2": ("block2a_expand_activation", "block3a_expand_activation", "block4a_expand_activation", "block6a_expand_activation", "top_activation"),
    "EfficientNetB3": ("block2a_expand_activation", "block3a_expand_activation", "block4a_expand_activation", "block6a_expand_activation", "top_activation"),
    "EfficientNetB4": ("block2a_expand_activation", "block3a_expand_activation", "block4a_expand_activation", "block6a_expand_activation", "top_activation"),
    "EfficientNetB5": ("block2a_expand_activation", "block3a_expand_activation", "block4a_expand_activation", "block6a_expand_activation", "top_activation"),
    "EfficientNetB6": ("block2a_expand_activation", "block3a_expand_activation", "block4a_expand_activation", "block6a_expand_activation", "top_activation"),
    "EfficientNetB7": ("block2a_expand_activation", "block3a_expand_activation", "block4a_expand_activation", "block6a_expand_activation", "top_activation"),
}


def get_backbone(backbone_name: str, weights: str, input_tensor: keras_tensor.KerasTensor, depth: int, freeze_backbone: bool = True, freeze_batch_norm: bool = False):
    """
    Configuring a user specified encoder model based on the `tensorflow.keras.applications`

    Parameters
    ----------
    backbone_name : str
        the bakcbone model name. Expected as one of the `tensorflow.keras.applications` class.
        Currently supported backbones are:
        (1) VGG16, VGG19
        (2) ResNet50, ResNet101, ResNet152
        (3) ResNet50V2, ResNet101V2, ResNet152V2
        (4) DenseNet121, DenseNet169, DenseNet201
        (5) EfficientNetB[0,7]
    weights : str
        one of None (random initialization), 'imagenet' (pre-training on ImageNet),or the path to the weights file to be loaded.
    input_tensor : KerasTensor
        The input tensor
    depth : int,
        Number of encoded feature maps. If four dwonsampling levels are needed, then depth=4.
    freeze_backbone : bool, default: True
        For a frozen backbone
    freeze_batch_norm : bool, default: False
        For not freezing batch normalization layers.

    Returns
    -------
    model : keras.model.Model
        a keras backbone model.

    """

    cadidate = layer_cadidates[backbone_name]

    # ----- #
    # depth checking
    depth_max = len(cadidate)
    if depth > depth_max:
        depth = depth_max
    # ----- #

    backbone_func = eval(backbone_name)
    backbone_ = backbone_func(
        include_top=False,
        weights=weights,
        input_tensor=input_tensor,
        pooling=None,
    )

    X_skip = []

    for i in range(depth):
        X_skip.append(backbone_.get_layer(cadidate[i]).output)

    model = Model(
        inputs=[
            input_tensor,
        ],
        outputs=X_skip,
        name="{}_backbone".format(backbone_name),
    )

    if freeze_backbone:
        model = freeze_model(model, freeze_batch_norm=freeze_batch_norm)

    return model


def freeze_model(model, freeze_batch_norm=False):
    """
    freeze a keras model

    Input
    ----------
        model: a keras model
        freeze_batch_norm: False for not freezing batch notmalization layers
    """
    if freeze_batch_norm:
        for layer in model.layers:
            layer.trainable = False
    else:
        from tensorflow.keras.layers import BatchNormalization

        for layer in model.layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False
    return model
