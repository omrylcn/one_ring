from typing import Dict, Union, Optional
from omegaconf import DictConfig, ListConfig
from tf_seg.deploy import preprocessor_lib, postprocessor_lib
from tf_seg.utils import TensorLike
import tensorflow as tf
import onnxruntime as ort
from tensorflow.keras.models import load_model
import os
import numpy as np
from tf_seg.transformers import normalize



class Inferencer:
    """
    Inference class for tf_seg models
    """

    def __init__(self, config: Union[Dict, DictConfig, ListConfig]):
        # print("config", config)
        """
        Parameters
        ----------
        config : Union[Dict, DictConfig, ListConfig]
            Configuration for the inferencer
        """

        self.config = config

        # load preprocessor
        preprocessor_type = config["preprocessor_type"]
        assert preprocessor_type in preprocessor_lib.keys(), f"preprocessor_type {preprocessor_type} not in {preprocessor_lib.keys()}"
        self._preprocessor = preprocessor_lib[preprocessor_type](config)

        # load model
        self._model = self.load(config)

        # load postprocessor
        postprocessor_type = config["postprocessor_type"]
        assert postprocessor_type in postprocessor_lib.keys(), f"postprocessor_type {postprocessor_type} not in {postprocessor_lib.keys()}"
        self._postprocessor = postprocessor_lib[postprocessor_type](config)

    def _check_params(self):
        pass

    def load(self, config: Optional[Union[Dict, DictConfig, ListConfig]] = None):
        """
        Load onnx or tf model

        Parameters
        ----------
        config : Optional[Union[Dict, DictConfig, ListConfig]], optional
            Configuration for the inferencer, by default None

        Returns
        -------
        model
            Loaded model

        """
        config = config if config else self.config

        # TODO: get support model list format from constant.py
        assert config["model_type"] in ["tf", "onnx"], f"model_type {config['model_type']} not supported"

        if config["model_type"] == "tf":
            model_path = config["model_path"]
            model = tf.keras.models.load_model(model_path)
            return model

        elif config["model_path"].split(".")[-1] == "onnx":
            model_path = config["model_path"]

        else:
            model_path = os.path.join(config["model_path"], "model.onnx")

        sess = ort.InferenceSession(model_path)
        return sess

    @property
    def prediction_model(self):
        if self.config["model_type"] == "tf":
            return self._model.predict

        # onnxruntime
        else:
            return self._model.run

    @property
    def preprocessor(self):
        return self._preprocessor

    @property
    def postprocessor(self):
        return self._postprocessor

    def pre_process(self, x: TensorLike) -> TensorLike:
        """
        Preprocess the input image

        Parameters
        ----------
        x : TensorLike
            Input image

        Returns
        -------
        TensorLike
            Preprocessed image

        """
        # TODO: check data type for model types parameters
        x = self.preprocessor(x)
        if self.config["normalize"]:
            x = self._normalize(x)

        if x.ndim == 3:
            x = np.expand_dims(x, axis=0)

        # convert type
        #if self.config["model_type"] == "onnx":
        #    x = x.astype(np.float32)
        #   
        #elif self.config["model_type"] == "tf":
        #    x = tf.cast(x, tf.float32)

        return x

    def post_process(self, x: TensorLike) -> TensorLike:
        """
        Postprocess the output of the model

        Parameters
        ----------
        x : TensorLike
            Output of the model

        Returns
        -------
        TensorLike

        """
        return self.postprocessor(x)

    # @tf.function
    def _normalize(self, image: TensorLike):
        # return tf.image.per_image_standardization(image)
        return normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def predict(self, image: TensorLike, input_name: str = "input"):
        x = self.pre_process(image)
        if self.config["model_type"] == "tf":
            x = self.prediction_model(x)
        else:
            x = self.prediction_model(None, {input_name: x})[0]

        # x = self.post_process(x)
        return x
