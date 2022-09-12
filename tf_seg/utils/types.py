"""
Types for typing functions signatures.

# Check more types for :
- https://github.com/tensorflow/agents/blob/v0.13.0/tf_agents/typing/types.py

"""

from typing import Union, Callable, List

import importlib
import numpy as np
import tensorflow as tf
from tensorflow import python


# TODO: Remove once https://github.com/tensorflow/tensorflow/issues/44613 is resolved
if tf.__version__[:3] > "2.5":
    from keras.engine import keras_tensor
else:
    from tensorflow.python.keras.engine import keras_tensor


Number = Union[
    float,
    int,
    np.float16,
    np.float32,
    np.float64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]

Initializer = Union[None, dict, str, Callable, tf.keras.initializers.Initializer]
Regularizer = Union[None, dict, str, Callable, tf.keras.regularizers.Regularizer]
Constraint = Union[None, dict, str, Callable, tf.keras.constraints.Constraint]
Activation = Union[None, str, Callable]
if importlib.util.find_spec("tensorflow.keras.optimizers.legacy") is not None:
    Optimizer = Union[tf.keras.optimizers.Optimizer, tf.keras.optimizers.legacy.Optimizer, str]
else:
    Optimizer = Union[tf.keras.optimizers.Optimizer, str]

TensorLike = Union[
    List[Union[Number, list]],
    tuple,
    Number,
    np.ndarray,
    tf.Tensor,
    tf.SparseTensor,
    tf.Variable,
    keras_tensor.KerasTensor,
]
FloatTensorLike = Union[tf.Tensor, float, np.float16, np.float32, np.float64]
AcceptableDTypes = Union[tf.DType, np.dtype, type, int, str, None]


Tensor = Union[python.framework.ops.Tensor, python.framework.sparse_tensor.SparseTensor, python.ops.ragged.ragged_tensor.RaggedTensor, python.framework.ops.EagerTensor]
