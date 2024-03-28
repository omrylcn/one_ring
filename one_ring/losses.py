"""


# check more information :
- https://github.com/yingkaisha/keras-unet-collection
- https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions
- https://github.com/tensorflow/addons

"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.python.util.tf_export import keras_export

from one_ring.utils.types import FloatTensorLike, TensorLike, Tensor
from one_ring.utils import is_tensor_or_variable

# import tensorflow as tf
# from tensorflow.keras.losses import LossFunctionWrapper
# from tensorflow.types.experimental import TensorLike, FloatTensorLike
# from tensorflow.keras.backend import epsilon as K_epsilon


def find_axis(data: TensorLike) -> tuple[int]:
    """
    Identifies spatial dimension indices in a tensor.

    Parameters
    ----------
    data : object
        A tensor-like object with `ndim` attribute, representing its number of dimensions.
        Expected to be 3D (H, W, C) or 4D (N, H, W, C).

    Returns
    -------
    tuple of int
        Indices of the height and width dimensions. Returns (0, 1) for 3D and (1, 2) for 4D tensors.

    Raises
    ------
    ValueError
        If `data` does not have an `ndim` attribute or if `ndim` is not 3 or 4.

    Example
    -------
    >>> find_axis(np.random.rand(64, 64, 3))
    (0, 1)
    >>> find_axis(np.random.rand(10, 64, 64, 3))
    (1, 2)
    """

    if data.ndim == 3:
        return (0, 1)
    elif data.ndim == 4:
        return (1, 2)
    else:
        raise ValueError(f"Data shape {data.shape} is not supported. Expected 3 or 4 dimensions (HWC or BHWC).")


class LossFunctionWrapper(tf.keras.losses.Loss):
    """Wraps a loss function in the `Loss` class."""

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
        self.one_ring_type = "loss"

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


# ========================= #
# Jaccard loss and similarity


@tf.function
def jaccard_similarity(y_true: TensorLike, y_pred: TensorLike, axis:tuple[int]) -> Tensor:
    """
    Computes the Jaccard similarity index for 2-d samples.

    Parameters
    ----------
    y_true : tensor-like
        The ground truth values.
    y_pred : tensor-like
        The predicted values.

    Returns
    -------
    loss : tensor
        Loss values per sample.

    """

    if axis is None:
        axis = find_axis(y_true)

    # y_true = tf.reshape(y_true, [-1])
    # y_pred = tf.reshape(y_pred, [-1])

    intersection = tf.keras.backend.sum(y_true * y_pred, axis=axis)
    union = tf.keras.backend.sum(y_true + y_pred - y_true * y_pred, axis=axis)
    iou_class = intersection / union

    return tf.keras.backend.mean(iou_class)


@tf.function
def jaccard_loss(y_true: TensorLike, y_pred: TensorLike, axis: tuple[int]) -> Tensor:
    """
    Computes the Jaccard loss for 2-d samples.

    Parameters
    ----------
    y_true : tensor-like
        The ground truth values.
    y_pred : tensor-like
        The predicted values.

    Returns
    -------
    loss : tensor
        Loss values per sample.

    """

    return 1 - jaccard_similarity(y_true, y_pred, axis)


class JaccardLoss(LossFunctionWrapper):
    """
    Implements the Jaccard loss for image segmentation.

    The Jaccard loss is a measure of how well the predicted set matches the true set.
    It is defined as the following::

        jaccard_loss = 1 - jaccard_similarity(y_true, y_pred)

    See Also
    --------
    https://en.wikipedia.org/wiki/Jaccard_index

    Examples
    --------
    >>> y_true = tf.constant([[[0, 1, 0], [0, 0, 1]]], dtype=tf.float32)
    >>> y_pred = tf.constant([[[0.1, 0.9, 0.1], [0.1, 0.8, 0.1]]], dtype=tf.float32)
    >>> print(y_true.shape)
    ... (1, 2, 3)
    >>> jaccard_loss = JaccardLoss()
    >>> print(jaccard_loss(y_true, y_pred).numpy())
    """

    def __init__(self, name="jaccard_loss", axis=(1, 2), **kwargs):
        super().__init__(fn=jaccard_loss, name=name, axis=axis, **kwargs)


# ========================= #
# Dice loss and variants


@tf.function()
def dice_coef(
    y_true: TensorLike, y_pred: TensorLike, axis: tuple[int], const: FloatTensorLike = K.epsilon()
) -> Tensor:
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

    # find spatial dimensions
    # print(axis)
    if axis is None:
        axis = find_axis(y_true)

    # get true pos (TP), false neg (FN), false pos (FP).

    true_pos = tf.keras.backend.sum(y_true * y_pred, axis=axis)
    false_neg = tf.keras.backend.sum(y_true * (1 - y_pred), axis=axis)
    false_pos = tf.keras.backend.sum((1 - y_true) * y_pred, axis=axis)

    # 2TP/(2TP+FP+FN) == 2TP/()
    coef_val = (2.0 * true_pos + const) / (2.0 * true_pos + false_pos + false_neg)

    return tf.keras.backend.mean(coef_val)


@tf.function()
def dice_loss(
    y_true: TensorLike, y_pred: TensorLike, axis:tuple[int], const: FloatTensorLike = K.epsilon()
) -> Tensor:
    """Sørensen–Dice Loss function for 2-d samples."""

    loss = 1 - dice_coef(y_true, y_pred, axis, const)
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

    def __init__(self, axis:tuple[int] = (1, 2), const: FloatTensorLike = K.epsilon(), name="dice_loss", **kwargs):
        super().__init__(fn=dice_loss, name=name, axis=axis, const=const, **kwargs)

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
def tversky_coef(
    y_true: TensorLike,
    y_pred: TensorLike,
    alpha: FloatTensorLike = 0.5,
    gamma: FloatTensorLike = 4 / 3,
    const: FloatTensorLike = K.epsilon(),
    return_classwise: bool = False,
) -> Tensor:
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
    return_classwise : bool, optional (default=False)
        If True, returns the Tversky coefficient for each class.

    See Also
    --------
    Tversky index, https://en.wikipedia.org/wiki/Tversky_index


    Returns
    -------
    loss : tensor
        Loss values per sample.

    """

    # flatten 2-d tensors
    axis = find_axis(y_true)

    # get true pos (TP), false neg (FN), false pos (FP).
    true_pos = tf.keras.backend.sum(y_true * y_pred, axis=axis)
    false_neg = tf.keras.backend.sum(y_true * (1 - y_pred), axis=axis)
    false_pos = tf.keras.backend.sum((1 - y_true) * y_pred, axis=axis)

    # TP/(TP + a*FN + b*FP); a+b = 1
    coef_val = (true_pos + const) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + const)

    if return_classwise:
        return coef_val

    return tf.keras.backend.mean(coef_val)


@tf.function
def tversky(
    y_true: TensorLike,
    y_pred: TensorLike,
    alpha: FloatTensorLike = 0.5,
    gamma: FloatTensorLike = 4 / 3,
    const: FloatTensorLike = K.epsilon(),
) -> Tensor:
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
def focal_tversky(
    y_true,
    y_pred,
    alpha: FloatTensorLike = 0.5,
    gamma: FloatTensorLike = 4 / 3,
    const: FloatTensorLike = K.epsilon(),
) -> Tensor:
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
    loss_val = tf.math.pow((1 - tversky_coef(y_true, y_pred, alpha=alpha, const=const)), 1 / gamma)

    return loss_val


class FocalTverskyLoss(LossFunctionWrapper):
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

    Examples
    --------
    >>> y_true = tf.constant([[0, 1, 0], [0, 0, 1]], dtype=tf.float32)
    >>> y_pred = tf.constant([[0.1, 0.9, 0.1], [0.1, 0.8, 0.1]], dtype=tf.float32)
    >>> focal_tversky_loss = FocalTverskyLoss()
    >>> print(focal_tversky_loss(y_true, y_pred).numpy())
    """

    def __init__(
        self,
        name="focal_tversky_loss",
        alpha: FloatTensorLike = 0.5,
        gamma: FloatTensorLike = 4 / 3,
        const: FloatTensorLike = K.epsilon(),
        **kwargs,
    ):
        super().__init__(fn=focal_tversky, name=name, alpha=alpha, gamma=gamma, const=const, **kwargs)


# ========================= #
#  Basnet Losses


@tf.function
def basnet_hybrid_loss(y_true: TensorLike, y_pred: TensorLike) -> Tensor:
    """
    Implements the hybrid loss proposed in BASNET, combining Binary Cross Entropy (BCE),
    Structural Similarity Index (SSIM), and Jaccard (IoU) losses.

    This hybrid approach aims to guide the network to learn three-level (pixel-, patch-,
    and map-level) hierarchy representations for improved segmentation performance.

    Parameters
    ----------
    y_true : tensor-like
        The ground truth values.
    y_pred : tensor-like
        The predicted values.

    Returns
    -------
    loss : tensor
        Loss values per sample.

    """

    bce_loss = BinaryCrossentropy(from_logits=False)(y_true, y_pred)
    ms_ssim_loss = 1 - tf.image.ssim(y_true, y_pred, max_val=1)
    jacard_loss = 1 - jaccard_similarity(y_true, y_pred)

    return bce_loss + ms_ssim_loss + jacard_loss


class BASNetHybridLoss(LossFunctionWrapper):
    """
    Implements the hybrid loss proposed in BASNET, combining Binary Cross Entropy (BCE),
    Structural Similarity Index (SSIM), and Jaccard (IoU) losses.

    This hybrid approach aims to guide the network to learn three-level (pixel-, patch-,
    and map-level) hierarchy representations for improved segmentation performance.

    See Also
    --------
    Qin, X., et al. BASNet: Boundary-aware salient object detection.
    https://arxiv.org/abs/2101.04704

    Examples
    --------
    >>> y_true = tf.constant([[0, 1, 0], [0, 0, 1]], dtype=tf.float32)
    >>> y_pred = tf.constant([[0.1, 0.9, 0.1], [0.1, 0.8, 0.1]], dtype=tf.float32)
    >>> hybrid_loss = BASNetHybridLoss()
    >>> print(hybrid_loss(y_true, y_pred).numpy())
    """

    def __init__(self, name="basnet_hybrid_loss", **kwargs):
        super().__init__(fn=basnet_hybrid_loss, name=name, **kwargs)

    # def get_config(self):
    #     config = {}
    #     base_config = super().get_config()
    #     return {**base_config, **config}


# ========================= #
# Combo Losses


@tf.function
def combo_loss(y_true: TensorLike, y_pred: TensorLike, alpha: FloatTensorLike = 0.5) -> Tensor:
    """
    Implements the Combo loss.

    The Combo loss is a combination of the Dice loss and the Cross-Entropy loss.

    Parameters
    ----------
    y_true : tensor-like
        The ground truth values.
    y_pred : tensor-like
        The predicted values.
    alpha : float-tensor-like, optional (default=0.5)
        The weight of the Dice loss in the weighted average.

    Returns
    -------
    loss : tensor
        Loss values per sample.

    """

    dice_loss_value = dice_loss(y_true, y_pred)
    cross_entropy_loss = BinaryCrossentropy(from_logits=False)(y_true, y_pred)

    return alpha * dice_loss_value + (1 - alpha) * cross_entropy_loss


class ComboLoss(LossFunctionWrapper):
    """Implements the Combo loss.

    The Combo loss is a combination of the Dice loss and the Cross-Entropy loss.

    See Also
    --------
    https://arxiv.org/pdf/1805.02798.pdf

    Examples
    --------
    >>> y_true = tf.constant([[0, 1, 0], [0, 0, 1]])
    >>> y_pred = tf.constant([[0.1, 0.9, 0.1], [0, 0.5, 0.5]])
    >>> combo_loss = ComboLoss()
    >>> combo_loss(y_true, y_pred)
    <tf.Tensor: shape=(), dtype=float32, numpy=0.8>
    """

    def __init__(self, name: str = "combo_loss", alpha: FloatTensorLike = 0.5, **kwargs):
        super().__init__(fn=combo_loss, name=name, alpha=alpha, **kwargs)


# ========================= #
# Losses dictionary

__one_ring_losses__ = ["JaccardLoss", "DiceLoss", "FocalTverskyLoss", "LogCoshDiceLoss", "BASNetHybridLoss"]
LOSSES = {
    "JaccardLoss": JaccardLoss,
    "dice_loss": DiceLoss,
    "focal_tversky_loss": FocalTverskyLoss,
    "log_cosh_dice_loss": LogCoshDiceLoss,
    "binary_crossentropy": BinaryCrossentropy,
    "categorical_crossentropy": CategoricalCrossentropy,
    "basnet_hybrid_loss": BASNetHybridLoss,
}
__all__ = ["DiceLoss", "FocalTverskyLoss", "LogCoshDiceLoss", "__one_ring_losses__", "LOSSES"]
