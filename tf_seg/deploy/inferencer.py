from typing import Dict, Union, Optional, List, Tuple
from omegaconf import DictConfig, ListConfig
from tf_seg.deploy import preprocessor_lib, postprocessor_lib
from tf_seg.utils import TensorLike
import tensorflow as tf
import onnxruntime as ort
from tensorflow.keras.models import load_model
import os
import numpy as np
from tf_seg.transformers import normalize
from tf_seg.config import MODEL_TYPE_LIB
from tf_seg.deploy import model_wrapper_lib


class Inferencer:
    """
    Inference class for tf_seg models
    """

    def __init__(
        self,
        image_size: Union[List[int], Tuple[int]],
        normalizing: bool,
        model_type: str,
        model_path: str,
        preprocessor_type: str,
        postprocessor_type: str,
        preprocessor_path: str = None,
        postprocessor_path: str = None,
        seed: int = 48,
        device: bool = "cpu",
    ):
        # print("config", config)
        """
        Parameters
        ----------
        image_size : Union[List[int], Tuple[int]]
            Size of the input image
        normalizing : bool
            Whether to normalize the input image
        model_type : str, {"tf", "onnx"}
            Type of the model
        model_path : str
            Path of the model folder
        preprocessor_type : str
            Type of the preprocessor object that will be used to process data before prediction
        preprocessor_path : str
            Path of the preprocessor object
        postprocessor_type : str
            Type of the postprocessor object that will be used to process after prediction
        postprocessor_path : str
            Path of the postprocessor object

        """
        self.image_size = image_size
        self.model_type = model_type
        self.preprocessor_type = preprocessor_type
        self.preprocessor_path = preprocessor_path
        self.model_type = model_type
        self.model_path = model_path
        self.postprocessor_type = postprocessor_type
        self.postprocessor_path = postprocessor_path
        self.normalizing = normalizing
        self.seed = seed
        self.device = device

        config = self.config

        # load model
        self._model = self.load(config)
        self._preprocessor = None
        self._postprocessor = None

        # load preprocessor
        preprocessor_type = config["preprocessor_type"]
        if preprocessor_type:
            assert preprocessor_type in preprocessor_lib.keys(), f"preprocessor_type {preprocessor_type} not in {preprocessor_lib.keys()}"
            self._preprocessor = preprocessor_lib[preprocessor_type](config)

        # load postprocessor
        postprocessor_type = config["postprocessor_type"]
        if postprocessor_type:
            assert postprocessor_type in postprocessor_lib.keys(), f"postprocessor_type {postprocessor_type} not in {postprocessor_lib.keys()}"
            self._postprocessor = postprocessor_lib[postprocessor_type](config)

    @property
    def config(self):
        config = {}
        config["image_size"] = self.image_size
        config["model_type"] = self.model_type
        config["model_path"] = self.model_path
        config["normalizing"] = self.normalizing
        config["preprocessor_type"] = self.preprocessor_type
        config["preprocessor_path"] = self.preprocessor_path
        config["postprocessor_type"] = self.postprocessor_type
        config["postprocessor_path"] = self.postprocessor_path
        config["device"] = self.device
        config["seed"] = self.seed

        return config

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
        # TODO: add logger to inform model is loaded

        config = config if config else self.config

        assert config["model_type"] in MODEL_TYPE_LIB, f"model_type {config['model_type']} not supported"
        self.model = model_wrapper_lib[config["model_type"]](**config)


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

        
        if self.preprocessor:
            config = self.config

            x = self.preprocessor(x)
            if config["normalizing"]:
                x = self._normalize(x)

            if x.ndim == 3:
                x = np.expand_dims(x, axis=0)

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
        if self.postprocessor:
            x = self.postprocessor(x)
        
        return x

    # @tf.function
    def _normalize(self, image: TensorLike):
        # return tf.image.per_image_standardization(image)
        return normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def predict(self, image: TensorLike, input_name: str = "input"):
        x = self.pre_process(image)
        x = self.model(image)

        # x = self.post_process(x)
        return x
