from tensorflow.keras.layers import Layer, Activation, BatchNormalization, Conv2D, Add


class ResidualLayer(Layer):
    def __init__(self, n_filter, kernel_size=3, activation="relu", padding="same", name="residual"):
        super().__init__(name=name)
        assert n_filter is not None, "n_filter must be specified"

        self.activation_name = activation
        self.n_filter = n_filter
        self.kernel_size = kernel_size
        self.padding = padding

    def build(self, input_shape):
        self.activation = Activation(self.activation_name)

        self.conv1 = Conv2D(filters=self.n_filter, kernel_size=self.kernel_size, padding=self.padding)
        self.batch_norm1 = BatchNormalization()

        self.conv2 = Conv2D(filters=self.n_filter, kernel_size=self.kernel_size, padding=self.padding)
        self.batch_norm2 = BatchNormalization()

        self.conv_skip = Conv2D(filters=self.n_filter, kernel_size=(1, 1), padding=self.padding)
        self.batch_norm_skip = BatchNormalization()

        self.add = Add()

    def call(self, inputs):

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
