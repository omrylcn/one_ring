import os
import logging
from typing import Dict, Union, Any
from abc import ABC, abstractmethod

import numpy as np
import onnxruntime
import tensorflow as tf

#from tensorflow.keras.models import load_model, Model
from one_ring.config import ONNX_PROVIDERS
from one_ring.utils import TensorLike, is_tensor_or_variable

class BaseModel(ABC):
    """Abstract base class for model wrappers."""

    def __init__(self, model_path: str, device: str, **kwargs):
        self.model_path = self._validate_model_path(model_path)
        self.device = device
        self.kwargs = kwargs
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _validate_model_path(model_path: str) -> str:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return model_path

    @staticmethod
    def _preprocess_input(x: TensorLike) -> np.ndarray:
        if is_tensor_or_variable(x):
            x = x.numpy()
        elif not isinstance(x, np.ndarray):
            raise ValueError(f"Unsupported input type: {type(x)}")

        x = x.astype(np.float32)
        return np.expand_dims(x, axis=0) if x.ndim == 3 else x

    @abstractmethod
    def __call__(self, x: TensorLike) -> np.ndarray:
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the model."""
        pass

class OnnxModel(BaseModel):
    """ONNX model wrapper for prediction"""

    def __init__(self, model_path: str, device: str, input_name: str = "input", 
                 providers: Dict[str, str] = ONNX_PROVIDERS, **kwargs):
        super().__init__(model_path, device, **kwargs)
        self.input_name = input_name
        provider = [providers[device]]
        self.model_path = os.path.join(self.model_path,"onnx/model.onnx")
        self.model = onnxruntime.InferenceSession(self.model_path, providers=provider)
        self.logger.info(f"ONNX model loaded from {model_path}")

    def __call__(self, x: TensorLike) -> np.ndarray:
        x = self._preprocess_input(x)
        self.logger.debug("Running ONNX inference")
        
        return self.model.run(None, {self.input_name: x})[0]

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "type": "ONNX",
            "input_name": self.input_name,
            "model_path": self.model_path,
            "inputs": [input.name for input in self.model.get_inputs()],
            "outputs": [output.name for output in self.model.get_outputs()],
        }


model_wrapper_lib = {"onnx": OnnxModel}#, "tf": TFModel}

# import os
# import onnxruntime
# from typing import Dict
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model, Model
# from one_ring.config import ONNX_PROVIDERS
# from one_ring.utils import TensorLike
# from one_ring.utils import is_tensor_or_variable


# class OnnxModel(object):
#     "Onnx model wrapper for prediction"

#     def __init__(self, model_path: str, device: str, input_name: str = "input", providers: Dict[str, str] = ONNX_PROVIDERS, **kwargs):
#         """
#         Parameters
#         ----------
#         model_path : str
#             Path of the model
#         device : str, {"cpu", "gpu"}
#             Device to run the model on
#         input_name : str, optional
#             Name of the input name in onnx session , by default "input"
#         providers : Dict[str, str], optional
#             Providers for onnx runtime, by default ONNX_PROVIDERS that is came from contant.py


#         """
#         self.model_path = model_path
#         self.device = device
#         self.input_name = input_name
#         self.kwargs = kwargs

#         if model_path.split(".")[-1] == "onnx":
#             model_path = model_path
#         else:
#             try:
#                 model_path = os.path.join(model_path, "model.onnx")

#             except Exception as e:
#                 print(f"Error in loading model: {e}")

#         self.model_path = model_path
#         provdier = [providers[device]]
#         self.model = onnxruntime.InferenceSession(model_path, providers=provdier)

#     def __call__(self, x: TensorLike, input_name: str = None):
#         input_name = input_name if input_name else self.input_name

#         # checking input type
#         if is_tensor_or_variable(x):
#             x = x.numpy()
#         elif type(x) == np.ndarray:
#             pass
#         else:
#             raise ValueError(f"Error in input type {type(x)}")

#         x = x.astype(np.float32)

#         # checking input shape
#         if x.ndim == 4:
#             pass
#         elif x.ndim == 3:
#             x = np.expand_dims(x, axis=0)
#         else:
#             raise ValueError(f"Error in input shape {x.shape}")

       
#         x = self.model.run(None, {input_name: x})[0]
#         return x


# class TFModel(object):
#     "Tensorflow-keras model wrapper for prediction"

#     def __init__(self, model_path: str, device: str, **kwargs) -> None:
#         """
#         Parameters
#         ----------
#         model_path : str
#             Path to the model
#         device : str, {"cpu", "gpu"}
#             Device name to run the model on
#         """
#         self.model_path = model_path
#         self.device = device
#         self.devices = tf.config.list_logical_devices()
#         self.kwargs = kwargs

#         self.model: Model
#         self.model = load_model(model_path)

#         self._fidn_device()

#     def _fidn_device(self):
#         """
#         Find and choose  a device in  TensorFlow devices according to the device parameter.
#         """

#         d = [device.name for device in self.devices if device.name.lower().find(self.device.lower()) > 0]

#         if len(d) == 1:
#             self.device_name = self.device
#             self.device = d[0]
#         elif len(d) > 1:
#             self.device_name = self.device
#             self.device = d[0]
#         else:
#             raise Exception(f"{self.device} device is not found in the list of devices: {self.devices}")

#     def __call__(self, x: TensorLike):

#         if is_tensor_or_variable(x):
#             if x.ndim == 4:
#                 pass
#             elif x.ndim == 3:
#                 x = tf.expand_dims(x, axis=0)
#             else:
#                 raise ValueError(f"Error in input shape {x.shape}")

#         elif type(x) == np.ndarray:
#             if x.ndim == 4:
#                 pass
#             elif x.ndim == 3:
#                 x = np.expand_dims(x, axis=0)
#             else:
#                 raise ValueError(f"Error in input shape {x.shape}")
#         else:
#             raise ValueError(f"Error in input type {type(x)}")

#         with tf.device(self.device):
#             x = self.model.predict(x)

#         return x


# model_wrapper_lib = {"onnx": OnnxModel, "tf": TFModel}
