import os
import onnxruntime
from typing import Dict
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tf_seg.config import ONNX_PROVIDERS
from tf_seg.utils import TensorLike
from tf_seg.utils import is_tensor_or_variable


class OnnxModel(object):
    "Onnx model wrapper for prediction"

    def __init__(self, model_path: str, device: str, input_name: str = "input", providers: Dict[str, str] = ONNX_PROVIDERS, **kwargs):
        """
        Parameters
        ----------
        model_path : str
            Path of the model
        device : str, {"cpu", "gpu"}
            Device to run the model on
        input_name : str, optional
            Name of the input name in onnx session , by default "input"
        providers : Dict[str, str], optional
            Providers for onnx runtime, by default ONNX_PROVIDERS that is came from contant.py


        """
        self.model_path = model_path
        self.device = device
        self.input_name = input_name
        self.kwargs = kwargs

        if model_path.split(".")[-1] == "onnx":
            model_path = model_path
        else:
            try:
                model_path = os.path.join(model_path, "model.onnx")

            except Exception as e:
                print(f"Error in loading model: {e}")

        self.model_path = model_path
        provdier = [providers[device]]
        self.model = onnxruntime.InferenceSession(model_path, providers=provdier)

    def __call__(self, x: TensorLike, input_name: str = None):
        input_name = input_name if input_name else self.input_name

        # checking input type
        if is_tensor_or_variable(x):
            x = x.numpy()
        elif type(x) == np.ndarray:
            pass
        else:
            raise ValueError(f"Error in input type {type(x)}")

        x = x.astype(np.float32)

        # checking input shape
        if x.ndim == 4:
            pass
        elif x.ndim == 3:
            x = np.expand_dims(x, axis=0)
        else:
            raise ValueError(f"Error in input shape {x.shape}")

       
        x = self.model.run(None, {input_name: x})[0]
        return x


class TFModel(object):
    "Tensorflow-keras model wrapper for prediction"

    def __init__(self, model_path: str, device: str, **kwargs) -> None:
        """
        Parameters
        ----------
        model_path : str
            Path to the model
        device : str, {"cpu", "gpu"}
            Device name to run the model on
        """
        self.model_path = model_path
        self.device = device
        self.devices = tf.config.list_logical_devices()
        self.kwargs = kwargs

        self.model: Model
        self.model = load_model(model_path)

        self._fidn_device()

    def _fidn_device(self):
        """
        Find and choose  a device in  TensorFlow devices according to the device parameter.
        """

        d = [device.name for device in self.devices if device.name.lower().find(self.device.lower()) > 0]

        if len(d) == 1:
            self.device_name = self.device
            self.device = d[0]
        elif len(d) > 1:
            self.device_name = self.device
            self.device = d[0]
        else:
            raise Exception(f"{self.device} device is not found in the list of devices: {self.devices}")

    def __call__(self, x: TensorLike):

        if is_tensor_or_variable(x):
            if x.ndim == 4:
                pass
            elif x.ndim == 3:
                x = tf.expand_dims(x, axis=0)
            else:
                raise ValueError(f"Error in input shape {x.shape}")

        elif type(x) == np.ndarray:
            if x.ndim == 4:
                pass
            elif x.ndim == 3:
                x = np.expand_dims(x, axis=0)
            else:
                raise ValueError(f"Error in input shape {x.shape}")
        else:
            raise ValueError(f"Error in input type {type(x)}")

        with tf.device(self.device):
            x = self.model.predict(x)

        return x


model_wrapper_lib = {"onnx": OnnxModel, "tf": TFModel}
