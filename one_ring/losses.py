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
def jaccard_similarity(y_true: TensorLike, y_pred: TensorLike, axis: tuple[int] = None,threshold:FloatTensorLike=None,const: FloatTensorLike = K.epsilon(),**kwargs) -> Tensor:
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

    # tf tensor casting

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    if threshold is not None:
        # Binarize predictions based on the specified threshold if provided
        y_pred = tf.cast(y_pred > threshold, y_true.dtype)
    
    

    if axis is None:
        axis = find_axis(y_true)

    # y_true = tf.reshape(y_true, [-1])
    # y_pred = tf.reshape(y_pred, [-1])

    intersection = tf.keras.backend.sum(y_true * y_pred, axis=axis)
    union = tf.keras.backend.sum(y_true + y_pred - y_true * y_pred, axis=axis)
    iou_class = (intersection+const)/ (union+const)

    return tf.keras.backend.mean(iou_class)


@tf.function
def jaccard_loss(y_true: TensorLike, y_pred: TensorLike, axis: tuple[int] = None,**kwargs) -> Tensor:
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


def dice_coef(
    y_true: TensorLike, y_pred: TensorLike, axis: tuple[int] = None,
    const: FloatTensorLike = K.epsilon(),threshold:FloatTensorLike=None,
    **kwargs
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

    # tf tensor casting
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    
    if threshold is not None:
        # Binarize predictions based on the specified threshold if provided
        y_pred = tf.cast(y_pred > threshold, y_true.dtype)

    # find spatial dimensions
    if axis is None:
        axis = find_axis(y_true)

    # get true pos (TP), false neg (FN), false pos (FP).
    true_pos = tf.keras.backend.sum(y_true * y_pred, axis=axis)
    false_neg = tf.keras.backend.sum(y_true * (1 - y_pred), axis=axis)
    false_pos = tf.keras.backend.sum((1 - y_true) * y_pred, axis=axis)

    # 2TP/(2TP+FP+FN) == 2TP/()
    coef_val = (2.0 * true_pos + const) / (2.0 * true_pos + false_pos + false_neg+const)

    return tf.keras.backend.mean(coef_val)


@tf.function()
def dice_loss(
    y_true: TensorLike, y_pred: TensorLike, axis: tuple[int] = None, const: FloatTensorLike = K.epsilon(),**kwargs
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

    def __init__(self, axis: tuple[int] = (1, 2), const: FloatTensorLike = K.epsilon(), name="dice_loss", **kwargs):
        super().__init__(fn=dice_loss, name=name, axis=axis, const=const, **kwargs)

    # def get_config(self):
    #     config = {"const": self._fn_kwargs["const"]}
    #     base_config = super().get_config()
    #     return {**base_config, **config}


@tf.function()
def log_cosh_dice_loss(
    y_true: TensorLike, y_pred: TensorLike, axis: tuple[int] = None, const: FloatTensorLike = K.epsilon(),**kwargs
) -> Tensor:
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

    dice_loss_value = dice_loss(y_true, y_pred, axis=axis, const=const)

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

    def __init__(
        self, name="log_cosh_dice_loss", axis: tuple[int] = (1, 2), const: FloatTensorLike = K.epsilon(), **kwargs
    ):
        super().__init__(fn=log_cosh_dice_loss, name=name, axis=axis, const=const, **kwargs)


# ========================= #
# Tversky loss and variants


@tf.function
def tversky_coef(
    y_true: TensorLike,
    y_pred: TensorLike,
    alpha: FloatTensorLike = 0.5,
    const: FloatTensorLike = K.epsilon(),
    return_classwise: bool = False,
    axis: tuple[int] = None,
    **kwargs
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

    # tf tensor casting
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    # flatten 2-d tensors
    if axis is None:
        axis = find_axis(y_true)

    # get true pos (TP), false neg (FN), false pos (FP).
    true_pos = keras.backend.sum(y_true * y_pred, axis=axis)
    false_neg = keras.backend.sum(y_true * (1 - y_pred), axis=axis)
    false_pos = keras.backend.sum((1 - y_true) * y_pred, axis=axis)

    # TP/(TP + a*FN + b*FP); a+b = 1
    coef_val = (true_pos + const) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + const)

    if return_classwise:
        return coef_val

    return keras.backend.mean(coef_val)


@tf.function
def tversky(
    y_true: TensorLike,
    y_pred: TensorLike,
    alpha: FloatTensorLike = 0.5,
    const: FloatTensorLike = K.epsilon(),
    **kwargs
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

    loss_val = 1 - tversky_coef(y_true, y_pred, alpha=alpha, const=const, return_classwise=False)

    return loss_val


@tf.function
def focal_tversky(
    y_true,
    y_pred,
    axis: tuple[int] = None,
    alpha: FloatTensorLike = 0.5,
    gamma: FloatTensorLike = 4 / 3,
    const: FloatTensorLike = K.epsilon(),
    **kwargs
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

    # (Tversky loss)**(1/gamma)
    loss_val = keras.backend.pow(
        (1 - tversky_coef(y_true, y_pred, axis=axis, alpha=alpha, const=const, return_classwise=True)), 1 / gamma
    )

    return keras.backend.mean(loss_val)


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
        axis: tuple[int] = (1, 2),
        alpha: FloatTensorLike = 0.5,
        gamma: FloatTensorLike = 4 / 3,
        const: FloatTensorLike = K.epsilon(),
        **kwargs,
    ):
        super().__init__(fn=focal_tversky, name=name, axis=axis, alpha=alpha, gamma=gamma, const=const, **kwargs)


# ========================= #
#  Basnet Losses


@tf.function
def basnet_hybrid_loss(y_true: TensorLike, y_pred: TensorLike, axis: tuple[int] = None, **kwargs) -> Tensor:
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
    jacard_loss = 1 - jaccard_similarity(y_true, y_pred, axis=axis)

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

    def __init__(self, name="basnet_hybrid_loss", axis: tuple[int] = (1, 2), **kwargs):
        super().__init__(fn=basnet_hybrid_loss, name=name, axis=axis, **kwargs)

    # def get_config(self):
    #     config = {}
    #     base_config = super().get_config()
    #     return {**base_config, **config}


# ========================= #
# Combo Losses


@tf.function
def combo_loss(y_true: TensorLike, y_pred: TensorLike, axis: tuple[int] = None, alpha: FloatTensorLike = 0.5, **kwargs) -> Tensor:
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

    dice_loss_value = dice_loss(y_true, y_pred, axis=axis)
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

    def __init__(self, name: str = "combo_loss", axis: tuple[int] = (1, 2), alpha: FloatTensorLike = 0.5, **kwargs):
        super().__init__(fn=combo_loss, name=name, axis=axis, alpha=alpha, **kwargs)


# ========================= #
# Unified Losses


@tf.function
def binary_focal_loss(
    y_true: TensorLike,
    y_pred: TensorLike,
    axis: tuple[int] = None,
    alpha: FloatTensorLike = 0.25,
    gamma: FloatTensorLike = 2.0,
    **kwargs,
) -> TensorLike:
    """
    Binary Focal Loss

    Parameters
    ----------
    y_true : TensorLike
        The ground truth values.
    y_pred : TensorLike
        The predicted values.
    alpha : FloatTensorLike, optional (default=0.25)
        The weighting factor for the positive class in the binary focal loss.
    gamma : FloatTensorLike, optional (default=2.0)
        The focusing parameter to modulate the loss for well-classified examples.

    Returns
    -------
    loss : TensorLike
        The computed binary focal loss.
    """
    epsilon_val = K.epsilon()
    y_pred = K.clip(y_pred, epsilon_val, 1.0 - epsilon_val)
    bce = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
    fl = alpha * K.pow((1 - y_pred), gamma) * bce + (1 - alpha) * K.pow(y_pred, gamma) * bce
    return K.mean(fl, axis=axis)


@tf.function
def categorical_focal_loss(
    y_true: TensorLike, y_pred: TensorLike, alpha: FloatTensorLike = 0.25, gamma: FloatTensorLike = 2.0, axis: int = -1,**kwargs
) -> TensorLike:
    """
    Categorical Focal Loss, with an adjustable axis parameter.

    Parameters
    ----------
    y_true : TensorLike
        The ground truth values, expected to be in one-hot encoded format.
    y_pred : TensorLike
        The predicted values.
    alpha : FloatTensorLike, optional (default=0.25)
        The weighting factor for the focal loss to address class imbalance.
    gamma : FloatTensorLike, optional (default=2.0)
        The focusing parameter to reduce the loss contribution from easy examples and focus more on hard examples.
    axis : int, optional (default=-1)
        The axis along which to sum the focal loss. Typically, this is the channel axis.

    Returns
    -------
    loss : TensorLike
        The computed categorical focal loss.
    """
    epsilon_val = K.epsilon()
    y_pred = K.clip(y_pred, epsilon_val, 1.0 - epsilon_val)
    cce = -y_true * K.log(y_pred)
    fl = alpha * K.pow((1 - y_pred), gamma) * cce
    return K.mean(K.sum(fl, axis=axis))


class FocalLoss(LossFunctionWrapper):
    """
    Implements Focal Loss for handling class imbalance by reducing the relative loss for well-classified examples
    and focusing more on hard to classify examples. Can be used for both binary and multi-class classification.

    The loss equation is -alpha * (1 - p_t)^gamma * log(p_t) for the binary case, and an extension of this for the
    multi-class case, where p_t is the model's estimated probability for each class.

    Parameters
    ----------
    mode : str, optional (default="binary")
        Mode of the focal loss function, "binary" or "categorical".
    axis : int, optional (default=-1)
        The axis along which the focal loss is applied. Typically, this is the channel axis.
    alpha : float, optional (default=0.25)
        The weighting factor alpha for the focal loss.
    gamma : float, optional (default=2.0)
        The focusing parameter gamma to balance the focal loss.

    See Also
    --------
    Lin, T.Y., et al. "Focal Loss for Dense Object Detection." IEEE Transactions on Pattern Analysis
    and Machine Intelligence, https://arxiv.org/abs/1708.02002

    Examples
    --------
    >>> y_true = tf.constant([[0, 1, 0], [0, 0, 1]], dtype=tf.float32)
    >>> y_pred = tf.constant([[0.1, 0.9, 0.1], [0.1, 0.8, 0.1]], dtype=tf.float32)
    >>> focal_loss = FocalLoss(mode="binary")
    >>> print(focal_loss(y_true, y_pred).numpy())
    """

    def __init__(self, name="focal_loss", mode="binary", axis=-1, alpha=0.25, gamma=2.0, **kwargs):
        if mode == "binary":
            fn = binary_focal_loss
        elif mode == "categorical":
            fn = categorical_focal_loss
        else:
            raise ValueError("FocalLoss mode must be 'binary' or 'categorical'")

        super().__init__(fn=fn, name=name, **kwargs)


def sym_unified_focal_loss(
    y_true: TensorLike,
    y_pred: TensorLike,
    mode: str = "binary",
    gamma: FloatTensorLike = 2.0,
    alpha: FloatTensorLike = 0.25,
    loss_weight: FloatTensorLike = 0.5,
    axis: tuple[int] = -1,
    const: FloatTensorLike = K.epsilon(),
    **kwargs,
) -> TensorLike:
    """
    Implements the Symmetric Unified Focal Loss (SUFL), a combination of Focal loss and
    Tversky loss (as approximated by the focal_tversky function), designed to balance the
    importance of true positives and false positives, without directly accounting for class imbalance.

    Parameters
    ----------
    y_true : TensorLike
        The ground truth values.
    y_pred : TensorLike
        The predicted values.
    mode : str
        The mode of operation, either "binary" or "categorical".
    gamma : FloatTensorLike
        The focusing parameter for the Focal loss.
    alpha : FloatTensorLike
        The weight of the Dice loss (via Tversky index) in the weighted average.
    loss_weight : FloatTensorLike
        The weight of the Focal Loss component in the combined loss calculation.
    axis : tuple[int]
        The spatial dimensions over which to calculate the loss.
    const : FloatTensorLike
        A constant that smooths the loss gradient and reduces numerical instabilities.

    Returns
    -------
    TensorLike
        The computed Symmetric Unified Focal Loss values per sample.
    """

    ftl = focal_tversky(y_true, y_pred, alpha=alpha,axis=(1,2),gamma=gamma, const=const)

    if mode == "binary":
        fl = binary_focal_loss(y_true, y_pred, axis=None, alpha=alpha, gamma=gamma)
    elif mode == "categorical":
        fl = categorical_focal_loss(y_true, y_pred, alpha=alpha, gamma=gamma, axis=axis)

    else:
        raise ValueError("Invalid mode specified. Choose either 'binary' or 'categorical'.")

    # Combine Focal Loss and Tversky-based Loss with adjustable weighting
    sufl_loss = loss_weight * fl + (1 - loss_weight) * ftl

    return sufl_loss


class SymmetricUnifiedFocalLoss(LossFunctionWrapper):
    """
    Implements the Symmetric Unified Focal Loss (SUFL), a combination of Focal loss and
    Tversky loss (as approximated by the focal_tversky function), designed to balance the
    importance of true positives and false positives, without directly accounting for class imbalance.

    Parameters
    ----------
    mode : str, optional (default="binary")
        The mode of operation, either "binary" or "categorical".
    gamma : float, optional (default=2.0)
        The focusing parameter for the Focal loss.
    alpha : float, optional (default=0.25)
        The weight of the Dice loss (via Tversky index) in the weighted average.
    loss_weight : float, optional (default=0.5)
        The weight of the Focal Loss component in the combined loss calculation.
    axis : tuple[int], optional (default=-1)
        The spatial dimensions over which to calculate the loss.
    const : float, optional (default=K.epsilon())
        A constant that smooths the loss gradient and reduces numerical instabilities.

    Returns
    -------
    TensorLike
        The computed Symmetric Unified Focal Loss values per sample.
    """

    def __init__(
        self,
        name: str = "symmetric_unified_focal_loss",
        mode: str = "binary",
        gamma: FloatTensorLike = 2.0,
        alpha: FloatTensorLike = 0.25,
        loss_weight: FloatTensorLike = 0.5,
        axis: tuple[int] = -1,
        const: FloatTensorLike = K.epsilon(),
        **kwargs,
    ):
        super().__init__(
            fn=sym_unified_focal_loss,
            name=name,
            mode=mode,
            gamma=gamma,
            alpha=alpha,
            loss_weight=loss_weight,
            axis=axis,
            const=const,
            **kwargs,
        )



# ========================= #
# Boundary Difference Over Union Loss

def boundary_dou_loss(y_true: TensorLike, y_pred: TensorLike, n_classes: int,**kwargs) -> tf.Tensor:
    """
    Computes the Boundary Distance Output Loss for image segmentation tasks with multi-class outputs.

    Parameters
    ----------
    y_true : TensorLike
        The ground truth values, one-hot encoded.
    y_pred : TensorLike
        The predicted values.
    n_classes : int
        The number of classes.

    Returns
    -------
    loss : tf.Tensor
        Computed loss per sample.
    """

    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
  
    loss = 0.0
    for i in range(n_classes):
        target = y_true[:, :, :, i:i+1]
        score = y_pred[:, :, :, i:i+1]
        loss += _adaptive_size(score, target)
    return loss / n_classes

def _adaptive_size(score, target,**kwargs):
    """
    Helper function to compute adaptive size component of the loss.

    Parameters
    ----------
    score : tf.Tensor
        Predicted scores for a single class.
    target : tf.Tensor
        Ground truth for the same class.
    
    Returns
    -------
    loss : tf.Tensor
        Computed loss for the class.
    """
    kernel = tf.constant([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=tf.float32)
    kernel = tf.reshape(kernel, [3, 3, 1, 1])
    padding_out = tf.pad(target, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT", constant_values=0)
    
    Y = tf.nn.conv2d(padding_out, kernel, strides=[1, 1, 1, 1], padding="VALID")
    Y = Y * target
    Y = tf.where(Y == 5, x=tf.zeros_like(Y), y=Y)
    
    C = tf.math.count_nonzero(Y, dtype=tf.float32)
    S = tf.math.count_nonzero(target, dtype=tf.float32)
    smooth = 1e-5
    alpha = 1 - (C + smooth) / (S + smooth)
    alpha = 2 * alpha - 1
    alpha = tf.minimum(alpha, 0.8)  # Truncated alpha
    
    intersect = tf.reduce_sum(score * target)
    y_sum = tf.reduce_sum(target * target)
    z_sum = tf.reduce_sum(score * score)
    loss = (z_sum + y_sum - 2 * intersect + smooth) / (z_sum + y_sum - (1 + alpha) * intersect + smooth)

    return loss

class BoundaryDoULoss(LossFunctionWrapper):
    """
    Implements the Boundary Distance Output Loss for image segmentation tasks.

    This loss function enhances the boundary distinction between segmented regions, improving model accuracy on the edges.
    It adjusts the loss based on the proximity to the boundary regions, effectively giving more importance to getting the boundaries right.

    Examples
    --------
    >>> y_true = tf.constant([[[0, 1, 0], [0, 0, 1]]], dtype=tf.float32)
    >>> y_pred = tf.constant([[[0.1, 0.9, 0.1], [0.1, 0.8, 0.1]]], dtype=tf.float32)
    >>> boundary_loss = BoundaryDoULoss(n_classes=3)
    >>> print(boundary_loss(y_true, y_pred).numpy())
    """

    def __init__(self, n_classes=1, name="boundary_dou_loss", **kwargs):
        
          super().__init__(
            fn=boundary_dou_loss,
            name=name,
            n_classes=n_classes,
            # name=name,
            # mode=mode,
            # gamma=gamma,
            # alpha=alpha,
            # loss_weight=loss_weight,
            # axis=axis,
            # const=const,
            **kwargs,
        )


        #super(BoundaryDoULoss, self).__init__(fn=lambda y_true, y_pred: boundary_dou_loss(y_true, y_pred, n_classes), name=name, **kwargs)


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
