import tensorflow as tf
from typing import Callable, Optional
from tensorflow.keras.metrics import BinaryAccuracy, MeanIoU, CategoricalAccuracy
from tensorflow.keras.metrics import binary_accuracy, categorical_accuracy

from one_ring.losses import dice_coef, jaccard_similarity
from one_ring.utils.types import AcceptableDTypes

# from tensorflow.keras.metrics import MeanMetricWrapper


class MeanMetricWrapper(tf.keras.metrics.Mean):
    """Wraps a stateless metric function with the Mean metric."""

    one_ring_type = "metric"

    def __init__(
        self,
        fn: Callable,
        name: Optional[str] = None,
        dtype: AcceptableDTypes = None,
        **kwargs,
    ):
        """Creates a `MeanMetricWrapper` instance.
        Parameters
        ----------
        fn: Callable
            Function that computes the metric to wrap.
        name: Optional[str]
            Name of the metric.
        dtype: Optional[tf.DType]
            Data type of the metric result.

        kwargs:
            Keyword arguments to pass to the metric function.

        See Also
        --------
        - `tf.keras.metrics.Mean`
        - https://github.com/tensorflow/addons

        """
        super().__init__(name=name, dtype=dtype)
        self._fn = fn
        self._fn_kwargs = kwargs

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates metric statistics.

        `y_true` and `y_pred` should have the same shape.

        Parameters
        ----------
        y_true: Tensor
            The ground truth values.
        y_pred: Tensor
            The predicted values.
        sample_weight: Optional[Tensor]
            The weights for each value in `y_true`.

        Returns
        -------
          Update op.

        """
        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)
        # TODO: Add checks for ragged tensors and dimensions:
        #   `ragged_assert_compatible_and_get_flat_values`
        #   and `squeeze_or_expand_dimensions`
        matches = self._fn(y_true, y_pred, **self._fn_kwargs)
        return super().update_state(matches, sample_weight=sample_weight)

    def get_config(self):
        config = {k: v for k, v in self._fn_kwargs.items()}
        base_config = super().get_config()
        return {**base_config, **config}


class DiceScore(MeanMetricWrapper):
    """Computes the Dice score."""

    def __init__(self, name: str = "dice_score", **kwargs):
        super().__init__(fn=dice_coef, name=name, **kwargs)


class JaccardScore(MeanMetricWrapper):
    """Computes the Jaccard score."""

    def __init__(self, name: str = "jaccard_score", **kwargs):
        super().__init__(fn=jaccard_similarity, name=name, **kwargs)


__one_ring_metrics__ = ["DiceScore", "JaccardScore"]

# to control choosing right metrics for segmentation
METRICS = {'dice_score': DiceScore,"jaccard_score": JaccardScore}

__all__ = ["DiceScore","JaccardScore", "__one_ring_metrics__", "METRICS"]

