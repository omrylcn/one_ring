from tensorflow.keras.layers import add, multiply, Multiply
from tensorflow.keras.layers import Layer, ReLU, LeakyReLU, PReLU, ELU, Conv2D, BatchNormalization, Activation


class AttentionGate(Layer):
    """
    Self-attention gate modified from Oktay et al. 2018.

    Parameters
    ----------
    n_filter : int
        Number of filters in the convolutional layer.
    attention_type : str, optional
        Type of attention operation to apply (default is "add"). Can be "add", "multiply" or "subtract".
    activation_type : str, optional
        Type of activation function to apply after attention operation (default is "relu").
    name : str, optional
        Name of the layer (default is "attention_gate").

    """

    def __init__(self, n_filter, attention_type="add", activation_type="relu", name="attention_gate", **kwargs):
        super().__init__(name=name)

        self.n_filter = n_filter
        self.attention_type = attention_type
        self.activation_type = activation_type

        self.attention_func = eval(self.attention_type)
        self.wg_conv = Conv2D(self.n_filter, (1, 1), padding="same")
        self.ws_conv = Conv2D(self.n_filter, (1, 1), padding="same")
        self.activation = Activation(self.activation_type)
        self.conv_final = Conv2D(self.n_filter, (1, 1), padding="same")
        self.activation_final = Activation("sigmoid")

    def call(self, inputs, training=None):
        """
        Applies the attention gate, generating a weighted version of the skip connection.

        Parameters
        ----------
        gate : Tensor
            Gating signal tensor, typically from a decoder layer.
        skip : Tensor
            Skip connection tensor, typically from a corresponding encoder layer.
        training : bool, optional
            Whether the layer is in training mode or inference mode.

        Returns
        -------
        Tensor
            Weighted skip connection after applying the attention coefficients.

        """
        #print(len(inputs))
        gate = inputs[0]
        skip = inputs[1]
        #print(gate)
        Wg = self.w_conv(gate)
        # Wg = BatchNormalization()(Wg)

        Ws = self.ws_conv(skip)

        #print(Wg.shape, Ws.shape)
        # Ws = BatchNormalization()(Ws)

        query = self.attention_func([Wg, Ws])

        f = self.activation(query)
        f = self.conv_final(f)

        coef_att = self.activation_final(f)

        return multiply([skip, coef_att])

    def get_config(self):
        config = super().get_config().copy()

        config.update(
            {
                "n_filter": self.n_filter,
                "attention_type": self.attention_type,
            }
        )

        return config
