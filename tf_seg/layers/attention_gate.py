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
        gate= inputs[0]
        skip = inputs[1]

        Wg = Conv2D(self.n_filter, (1, 1), padding="same")(gate)
        # Wg = BatchNormalization()(Wg)

        Ws = Conv2D(self.n_filter, (1, 1), padding="same")(skip)

        #print(Wg.shape, Ws.shape)
        # Ws = BatchNormalization()(Ws)

        query = self.attention_func([Wg, Ws])

        f = Activation(self.activation_type)(query)
        f = Conv2D(self.n_filter, (1, 1), padding="same")(f)

        coef_att = Activation("sigmoid")(f)

        return Multiply()([skip, coef_att])

    def get_config(self):
        config = super().get_config().copy()

        config.update(
            {
                "n_filter": self.n_filter,
                "attention_type": self.attention_type,
            }
        )

        return config
