from tensorflow.keras.layers import Layer, Activation, BatchNormalization, Conv2D, Add


class ResidualLayer(Layer):
    def __init__(self, n_filter, kernel_size: int = 3, activation: str = "relu", padding: str = "same"):
        super().__init__()
        assert n_filter is not None, "n_filter must be specified"

        self.activation = Activation(activation)

        self.conv1 = Conv2D(filters=n_filter, kernel_size=kernel_size, padding=padding)
        self.batch_norm1 = BatchNormalization()

        self.conv2 = Conv2D(filters=n_filter, kernel_size=kernel_size, padding=padding)
        self.batch_norm2 = BatchNormalization()

        self.conv_skip = Conv2D(filters=n_filter, kernel_size=(1, 1), padding=padding)
        self.batch_norm_skip = BatchNormalization()

        self.add = Add()

    def __call__(self, inputs):

        x = self.batch_norm1(inputs)
        x = self.activation(x)
        x = self.conv1(x)

        x = self.batch_norm2(x)
        x = self.activation(x)
        x = self.conv2(x)

        x_i = self.conv_skip(inputs)
        x_i = self.batch_norm_skip(x_i)
        x = self.add([x, x_i])

        return x
