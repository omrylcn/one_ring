"""

trainer class yapısı
- objelerin parametrelerini kontrol eder.
- objelerin eğitim ve evaluate işlemlerini yapar.
- 
"""
import os
import pickle
import tf2onnx

from typing import Union, Optional, List, Dict
from omegaconf import DictConfig, ListConfig

from tensorflow.keras.models import Model, load_model
from tensorflow.data import Dataset
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import *

from tf_seg.losses import *
from tf_seg.metrics import *
from tf_seg.utils import snake_case_to_pascal_case
from tf_seg.deploy import export_to_onnx


# TODO : add logger
# TODO : change default values from constant.py
class Trainer:
    "Trainer class for training a model"

    def __init__(
        self,
        config: Union[DictConfig, ListConfig],
        model: Model,
        train_data: Dataset,
        val_data: Optional[Dataset] = None,
        callbacks: Optional[Union[Dict[str, Callback], List[Callback]]] = None,
    ) -> None:
        """

        Parameters
        ----------
        config: Union[DictConfig, ListConfig]
            Configuration for training
        model: tf.keras.Model
            Model to be trained
        train_data: tf.data.Dataset
            Dataset for training
        val_data: tf.data.Dataset,default: None
            Dataset for validation
        callbacks: tf.keras.callbacks.Callback,default: None
            keras Callbacks for training

        """
        self.all_config = config.copy()
        self.config = config["trainer"].copy()
        self._model = model
        self.train_data = train_data
        self.val_data = val_data
        self._callbacks = callbacks

        self._check_params()

    def _check_params(self) -> None:
        "Check parameters types"
        assert isinstance(self.config, (dict, DictConfig, ListConfig))
        assert isinstance(self._model, Model)
        assert isinstance(self.train_data, Dataset)

        if self.val_data:
            assert isinstance(self.val_data, Dataset)

        if self.callbacks:
            if len(self.callbacks) > 0:
                for callback in self.callbacks:
                    assert isinstance(callback, Callback)

    def _check_trainer_params(self):
        "Check trainer config parameters"
        assert self.config["epochs"] > 0
        #assert self.config["batch_size"] > 0 # remove batch size
        assert self.config["optimizer"] is not None
        assert self.config["losses"] is not None

    @property
    def callbacks(self) -> None:
        "Convert  list of callbacks"
        if self._callbacks:
            assert isinstance(self._callbacks, (List, Dict))
            if type(self._callbacks) == List:
                return self._callbacks
            else:
                return list(self._callbacks.values())

        return None

    @property
    def model(self):
        self._check_trainer_params()

        optimizer_conf = self.config["optimizer"]
        optimizer = eval(snake_case_to_pascal_case(optimizer_conf["name"]))(**optimizer_conf["params"])

        losses = [eval(snake_case_to_pascal_case(i))() for i in self.config["losses"]]
        metrics = [eval(snake_case_to_pascal_case(i))() for i in self.config["metrics"]]

        self._model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
        return self._model

    def train(self, continue_training: bool = False) -> None:
        "Train tf keras model"

        # TODO: merge continue training and first histories
        if continue_training and self._model._is_compiled:
            self.history = self._model.fit(self.train_data, epochs=self.config["epochs"], callbacks=self.callbacks, validation_data=self.val_data)

        else:
            self.history = self.model.fit(self.train_data, epochs=self.config["epochs"], callbacks=self.callbacks, validation_data=self.val_data)

        if self.config["save_model"]:
            # TODO: add logger
            self.save(path= self.config["save_name"])

    def evaluate(self) -> None:
        if self.val_data:
            self._model.evaluate(self.val_data)
        else:
            raise ValueError("Validation data is not provided")

    def save(self, path: str, meta_data_name="meta_data.pkl", onnx_name="model.onnx") -> None:
        """
        Saves the model to Tensorflow SavedModel

        Parameters
        ----------
        path: str
            Path to save model
        filename: str, default: None
            Filename to save model

        """

        self._model.save(path)
        self._save_meta_data(path, meta_data_name)

        if self.config["deploy_onnx"]:
            self._export_to_onnx(model_name=path, onnx_name=onnx_name)

    def _export_to_onnx(self, model_name: str, onnx_name: str, opset: int = 13)-> None:
        """
        Export model to onnx

        Parameters
        ----------
        model_name: str
            Path to model
        onnx_name: str
            Name of onnx file
        opset: int, default: 13
           opset version for onnx

        """

        export_to_onnx(model_name, onnx_name, opset=opset, optimizer="onnx")

    def _save_meta_data(self, path: str, filename: str) -> None:
        "Save meta data to model path"
        assert os.path.exists(path), f"Path: {path} does not exist"
        with open(path + "/" + filename, "wb") as f:
            pickle.dump(self.all_config, f)

    def load(self, path: str, checking_parameters: bool = True) -> None:
        """
        Load model from Tensorflow SavedModel

        Parameters
        ----------
        path: str
            Path to load model
        checking_parameters: bool, default: True
            Check parameters of model and config file

        """
        try:
            _model = load_model(path)
            if checking_parameters:
                config = self.all_config["model"]
                print(config)
                print(config["input_shape"])

                # input shape check config and loaded model
                assert list(config["input_shape"]) == list(_model.input_shape[1:]), f"Input shape of model is not equal to config input shape {config['input_shape']} != {_model.input_shape[1:]}"

                # output shape check config and loaded model
                config_output_shape = list(config["input_shape"])
                config_output_shape[-1] = config["output_size"]
                assert config_output_shape == list(_model.output_shape[1:])

                # final activation check config and loaded model
                model_last_activation = _model.layers[-1].activation.__name__
                config_last_activation = config["final_activation"]
                assert model_last_activation == config_last_activation, f"Final activation of model is not equal to config final activation {config_last_activation} != {model_last_activation}"

            self._model = _model

        except Exception as e:
            raise ValueError("Model could not be loaded",e)

    def test(self, data) -> None:
        self._model.evaluate(data)
