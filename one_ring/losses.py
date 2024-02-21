"""


# check more information :
- https://github.com/yingkaisha/keras-unet-collection
- https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions
- https://github.com/tensorflow/addons

"""


import tensorflow as tf
import tensorflow.keras.backend as K
from one_ring.utils.types import FloatTensorLike, TensorLike, Tensor
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from one_ring.utils import is_tensor_or_variable
from tensorflow.python.util.tf_export import keras_export

# import tensorflow as tf
# from tensorflow.keras.losses import LossFunctionWrapper
# from tensorflow.types.experimental import TensorLike, FloatTensorLike
# from tensorflow.keras.backend import epsilon as K_epsilon


class LossFunctionWrapper(tf.keras.losses.Loss):
    """Wraps a loss function in the `Loss` class."""

    one_ring_tpye = "loss"

    def __init__(self, fn, reduction=tf.keras.losses.Reduction.AUTO, name=None, **kwargs):
        # convert numpy tensorflow doc style to numpy doc style

        """Initializes `LossFunctionWrapper` instance.

        Parameters
        ----------
        fn : callable
            The loss function to wrap, with signature `fn(y_true, y_pred,
            **kwargs)`.
        reduction : `tf.keras.losses.Reduction`, optional (default=AUTO)
            The reduction to apply to the loss.
        name : str, optional (default=None)
            The name for the loss.
        **kwargs : dict
            Keyword arguments to pass to `fn`.


        """

        super().__init__(reduction=reduction, name=name)
        self.fn = fn
        self._fn_kwargs = kwargs

    def call(self, y_true: TensorLike, y_pred: TensorLike):

        """Invokes the `LossFunctionWrapper` instance.

        Parameters
        ----------
        y_true : tensor-like
            The ground truth values.
        y_pred : tenso-like
            The predicted values.


        Returns
        -------
        loss : tensor
          Loss values per sample.
        """
        return self.fn(y_true, y_pred, **self._fn_kwargs)

    def get_config(self):
        config = {}
        for k, v in self._fn_kwargs.items():
            config[k] = tf.keras.backend.eval(v) if is_tensor_or_variable(v) else v
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # def get_config(self):
    #     config = {}
    #     for k, v in iter(self._fn_kwargs.items()):
    #         config[k] = tf.keras.backend.eval(v) if is_tensor_or_variable(v) else v
    #     base_config = super().get_config()
    #     return {**base_config, **config}


@tf.function()
def dice_coef(y_true: TensorLike, y_pred: TensorLike, const: FloatTensorLike = K.epsilon()) -> Tensor:
    """
    Sørensen–Dice coefficient for 2-d samples.

    Parameters
    ----------
    y_true : tensor-like
        The ground truth values.
    y_pred : tenso-like
        The predicted values.
    const : float-tensor-like, optional (default=K.epsilon())
        a constant that smooths the loss gradient and reduces numerical instabilities.

    Returns
    -------
    loss : tensor
        Loss values per sample.

    """

    # flatten 2-d tensors
    y_true_pos = tf.reshape(y_true, [-1])
    y_pred_pos = tf.reshape(y_pred, [-1])

    # get true pos (TP), false neg (FN), false pos (FP).
    true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
    false_neg = tf.reduce_sum(y_true_pos * (1 - y_pred_pos))
    false_pos = tf.reduce_sum((1 - y_true_pos) * y_pred_pos)

    # 2TP/(2TP+FP+FN) == 2TP/()
    coef_val = (2.0 * true_pos + const) / (2.0 * true_pos + false_pos + false_neg)

    return coef_val


@tf.function()
def dice_loss(y_true: TensorLike, y_pred: TensorLike, const: FloatTensorLike = K.epsilon()) -> Tensor:
    """Sørensen–Dice Loss function for 2-d samples."""

    loss = 1 - dice_coef(y_true, y_pred)
    return loss


class DiceLoss(LossFunctionWrapper):
    """Implements the Sørensen–Dice loss.

    The Sørensen–Dice loss is designed to balance the importance of true positives
    and false positives, without accounting for class imbalance.
    It is defined as the following::
        dice_loss = 1 - dice_coef(y_true, y_pred)

    See Also
    --------
    https://arxiv.org/pdf/1707.03237.pdf

    Examples
    --------
    >>> y_true = K.constant([[0, 1, 0], [0, 0, 1]])
    >>> y_pred = K.constant([[0.1, 0.9, 0.1], [0, 0.5, 0.5]])
    >>> dice_loss = one_ring.losses.DiceLoss()
    >>> dice_loss(y_true, y_pred)
    <tf.Tensor: shape=(), dtype=float32, numpy=0.8>

    """

    def __init__(self, const: FloatTensorLike = K.epsilon(), name="dice_loss", **kwargs):
        super().__init__(fn=dice_loss, name=name, const=const, **kwargs)

    # def get_config(self):
    #     config = {"const": self._fn_kwargs["const"]}
    #     base_config = super().get_config()
    #     return {**base_config, **config}


@tf.function()
def log_cosh_dice_loss(y_true: TensorLike, y_pred: TensorLike, const: FloatTensorLike = K.epsilon()) -> Tensor:
    """
    Computes the log-cosh of the Dice loss.

    Parameters
    ----------
    y_true : tensor-like
        The ground truth values.
    y_pred : tensor-like
        The predicted values.
    const : float-tensor-like, optional (default=K.epsilon())
        A constant that smooths the loss gradient and reduces numerical instabilities.

    Returns
    -------
    loss : tensor
        Log-cosh Dice loss values per sample.
    """

    dice_loss_value = dice_loss(y_true, y_pred, const)
    log_cosh_dice = tf.math.log((tf.exp(dice_loss_value) + tf.exp(-dice_loss_value)) / 2.0)
    return log_cosh_dice

class LogCoshDiceLoss(LossFunctionWrapper):
    """
    Implements the Log-Cosh Dice loss.

    This loss combines the log-cosh function with the Sørensen–Dice coefficient to smooth
    the loss landscape and potentially provide more stable and effective training outcomes.

    See Also
    --------
    ttps://arxiv.org/pdf/2006.14822.pdf

    Examples
    --------
    >>> y_true = tf.constant([[0, 1, 0], [0, 0, 1]])
    >>> y_pred = tf.constant([[0.1, 0.9, 0.1], [0, 0.5, 0.5]])
    >>> log_cosh_dice_loss = LogCoshDiceLoss()
    >>> log_cosh_dice_loss(y_true, y_pred)
    <tf.Tensor: shape=(), dtype=float32, numpy=value>
    """

    def __init__(self, const: FloatTensorLike = K.epsilon(), name="log_cosh_dice_loss", **kwargs):
        super().__init__(fn=log_cosh_dice_loss, name=name, const=const, **kwargs)


# ========================= #
# Tversky loss and variants


@tf.function
def tversky_coef(y_true: TensorLike, y_pred: TensorLike, alpha: FloatTensorLike = 0.5, gamma: FloatTensorLike = 4 / 3, const: FloatTensorLike = K.epsilon(),) -> Tensor:
    """
    Tversky coefficient for 2-d samples.

    Parameters
    ----------
    y_true : tensor-like
        The ground truth values.
    y_pred : tensor-like
        The predicted values.
    alpha : float-tensor-like, optional (default=0.5)
        The weight of `true positives` in the weighted average.
    const : float-tensor-like, optional (default=K.epsilon())
        A constant that smooths the loss gradient and reduces numerical instabilities.

    See Also
    --------
    Tversky index, https://en.wikipedia.org/wiki/Tversky_index


    Returns
    -------
    loss : tensor
        Loss values per sample.

    """

    # flatten 2-d tensors
    y_true_pos = tf.reshape(y_true, [-1])
    y_pred_pos = tf.reshape(y_pred, [-1])

    # get true pos (TP), false neg (FN), false pos (FP).
    true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
    false_neg = tf.reduce_sum(y_true_pos * (1 - y_pred_pos))
    false_pos = tf.reduce_sum((1 - y_true_pos) * y_pred_pos)

    # TP/(TP + a*FN + b*FP); a+b = 1
    coef_val = (true_pos + const) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + const)

    return coef_val


@tf.function
def tversky(y_true: TensorLike, y_pred: TensorLike, alpha: FloatTensorLike = 0.5, gamma: FloatTensorLike = 4 / 3, const: FloatTensorLike = K.epsilon(),) -> Tensor:
    """
    Tversky Loss.

    Parameters
    ----------
    y_true : tensor-like
        The ground truth values.
    y_pred : tensor-like
        The predicted values.
    alpha : float-tensor-like, optional (default=0.5)
        The weight of `true positives` in the weighted average.
    const : float-tensor-like, optional (default=K.epsilon())
        A constant that smooths the loss gradient and reduces numerical instabilities.

    See Also
    --------
    Tversky loss function for image segmentation using 3D fully convolutional deep networks.
    https://arxiv.org/abs/1706.05721

    Tversky as a loss function for highly unbalanced image segmentation using 3d fully convolutional deep networks.
    https://arxiv.org/abs/1803.11078.

    Returns
    -------
    loss : tensor
        Loss values per sample.


    """
    # tf tensor casting
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    # <--- squeeze-out length-1 dimensions.
    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)

    loss_val = 1 - tversky_coef(y_true, y_pred, alpha=alpha, const=const)

    return loss_val


@tf.function
def focal_tversky(y_true, y_pred, alpha: FloatTensorLike = 0.5, gamma: FloatTensorLike = 4 / 3, const: FloatTensorLike = K.epsilon(),) -> Tensor:

    """
    Focal Tversky Loss (FTL)

    Parameters
    ----------
    y_true : tensor-like
        The ground truth values.
    y_pred : tensor-like
        The predicted values.
    alpha : float-tensor-like, optional (default=0.5)
        The weight of `true positives` in the weighted average.
    gamma : float-tensor-like, optional (default=4/3)
        A tunable parameter within [1, 3].

    See Also
    --------
    Abraham, N. and Khan, N.M., 2019, April. A novel focal tversky loss function with improved
    attention u-net for lesion segmentation.https://arxiv.org/abs/1810.07842

    Returns
    -------
    loss : tensor
        Loss values per sample.


    """
    # tf tensor casting
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    # <--- squeeze-out length-1 dimensions.
    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)

    # (Tversky loss)**(1/gamma)
    loss_val = tf.math.pow((1 - tversky_coef(y_true, y_pred, alpha=alpha, const=const)), 1/gamma)

    return loss_val


class FocalTverskyLoss(LossFunctionWrapper):
    def __init__(self, name="focal_tversky_loss", alpha: FloatTensorLike = 0.5, gamma: FloatTensorLike = 4 / 3, const: FloatTensorLike = K.epsilon(), **kwargs):
        super().__init__(fn=focal_tversky, name=name, alpha=alpha, gamma=gamma, const=const, **kwargs)


__one_ring_losses__ = ["DiceLoss", "FocalTverskyLoss", "LogCoshDiceLoss"]
LOSSES = {
    'dice_loss': DiceLoss, 'focal_tversky_loss': FocalTverskyLoss,'log_cosh_dice_loss': LogCoshDiceLoss,
    "binary_crossentropy": BinaryCrossentropy, "categorical_crossentropy": CategoricalCrossentropy
}

__all__ = ["DiceLoss", "FocalTverskyLoss","LogCoshDiceLoss", "__one_ring_losses__", "LOSSES"]

