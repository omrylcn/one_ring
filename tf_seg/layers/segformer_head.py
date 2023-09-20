# Copyright 2023 David Landup
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import Layer


class SegformerSegmentationHead(Layer):
    """
    Segformer Segmentation Head Layer.

    """

    def __init__(
        self,
        in_dims: Tuple(int),
        num_classes: int,
        embed_dim=256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.linear_layers = []

        for i in in_dims:
            self.linear_layers.append(
                tf.keras.layers.Dense(embed_dim, name=f"linear_{i}")
            )

        # To fuse multiple layer outputs into a single feature map using a Conv2d
        self.linear_fuse = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=embed_dim, kernel_size=1, use_bias=False
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("relu"),
            ]
        )
        self.dropout = tf.keras.layers.Dropout(0.1)
        # Final segmentation output
        self.seg_out = tf.keras.layers.Conv2D(filters=num_classes, kernel_size=1)

    def call(
        self,
        features,
        training=None,
    ):
        B, H, W, _ = features[0].shape
        outs = []

        for feature, layer in zip(features, self.linear_layers):
            feature = layer(feature)
            feature = tf.image.resize(feature, size=(H, W), method="bilinear")
            outs.append(feature)

        seg = self.linear_fuse(tf.concat(outs[::-1], axis=3))
        seg = self.dropout(seg)
        seg = self.seg_out(seg)

        return seg

    def get_config(
        self,
    ):
        config = super().get_config().copy()
        config.update(
            {
                "in_dims": self.in_dims,
                "num_classes": self.num_classes,
                "embed_dim": self.embed_dim,
            }
        )

        return config
