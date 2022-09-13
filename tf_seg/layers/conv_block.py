from tensorflow.keras.layers import (
    Layer,
    Activation,
    BatchNormalization,
    Conv2D,
    MaxPooling2D,
)


class ConvUnet(Layer):
    """Convolutional Black fıor Unet and Variants"""

    def __init__(self, n_filter: int, activation: str, name: str = None):
        super(ConvUnet, self).__init__(name=name)
        self.n_filter = n_filter
        self.activation = activation
        self.layer_name = name

        self.conv1 = Conv2D(n_filter, (3, 3), padding="same")
        self.bn1 = BatchNormalization()
        self.act1 = Activation(activation)

        self.conv2 = Conv2D(n_filter, (3, 3), padding="same")
        self.bn2 = BatchNormalization()
        self.act2 = Activation(activation)

        self.pool = MaxPooling2D((2, 2), (2, 2))

    def call(self, x, pool=True):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        c = x

        if pool == True:
            x = self.pool(x)
            return c, x
        return c
        
    def get_config(self):
        config = super(ConvUnet,self).get_config()
        config.update({"n_filter": self.n_filter, "activation": self.activation})
        return config