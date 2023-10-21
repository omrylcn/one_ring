"""

trainer class yapısı
- objelerin parametrelerini kontrol eder.
- objelerin eğitim ve evaluate işlemlerini yapar.
- 
"""
import os
import uuid
import datetime
import pickle
import tf2onnx


from typing import Union, Optional, List
from omegaconf import DictConfig, ListConfig

from tensorflow.keras.models import Model, load_model
from tensorflow.data import Dataset
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Optimizer, SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric

from tf_seg.losses import LOSSES
from tf_seg.metrics import METRICS
from tf_seg.logger import Logger

# from tf_seg.utils import snake_case_to_pascal_case
from tf_seg.deploy import save_model_as_onnx
from tf_seg.save import ModelSaver


OPTIMIZERS = {"sgd": SGD, "rmsprop": RMSprop, "adam": Adam, "adadelta": Adadelta, "adagrad": Adagrad, "adamax": Adamax, "nadam": Nadam, "ftrl": Ftrl}


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
        optimizer: Optional[Optimizer] = None,
        loss: Optional[Loss] = None,
        metrics: Optional[List[Metric]] = None,
        callbacks: Optional[List[Callback]] = None,
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
        self.logger = Logger("tf_seg", log_file="training_log.txt")

        self.config = config.copy()
        self.trainer_config = config["trainer"].copy()
        self._model = model
        self.train_data = train_data
        self.val_data = val_data
        self._optimizer = optimizer
        self._loss = loss
        self._metrics = metrics
        self._callbacks = callbacks

        self.saver = ModelSaver
        self._check_trainer_objects()  # check objects
        self._check_trainer_params()  # check parameters

        self.history_dict = {}

    def _check_trainer_objects(self) -> None:
        "Check parameters types"
        assert isinstance(self.config, (dict, DictConfig, ListConfig))
        assert isinstance(self._model, Model)
        assert isinstance(self.train_data, Dataset)

        if self.val_data:
            assert isinstance(self.val_data, Dataset)

        if self._callbacks:

            if isinstance(self._callbacks, (List)):
                assert isinstance(self._callbacks, (List)), "'callbacks' object should be list"
                if len(self._callbacks) > 0:
                    for callback in self._callbacks:
                        assert isinstance(callback, Callback)
            else:
                assert isinstance(self._callbacks, (Callback)), "a callback should be instance of tf.keras.callbacks.Callback"

        if self._optimizer:
            assert isinstance(self._optimizer, Optimizer)

        if self._loss:
            assert isinstance(self._loss, (List)), "'loss' should be instance of tf.keras.losses.Loss"
            if len(self._loss) > 0:
                for loss in self._loss:
                    assert isinstance(loss, Loss), " a member of 'loss' should be instance of tf.keras.losses.Loss"

        if self._metrics:
            assert isinstance(self._metrics, (List)), "'metrics' object should be list[tf.keras.metrics.Metric]"
            if len(self._metrics) > 0:
                for metric in self._metrics:
                    assert isinstance(metric, Metric), " a member of 'metrics' should be instance of tf.keras.metrics.Metric"

    def _check_trainer_params(self):
        "Check trainer config parameters"
        assert self.trainer_config["epochs"] > 0
        assert self.trainer_config["optimizer"] is not None
        assert self.trainer_config["losses"] is not None

    def log(self, message: str, level: int = 2) -> None:
        """
        Logs a message with a certain level of severity.

        Parameters
        ----------
        message : str
            The message to log.
        level : int, optional
            The level of the message (0 - Error, 1 - Warning, 2 - Info).
            Default is 2 (Info).
        """
        if level == 0:
            self.logger.error(message)
        elif level == 1:
            self.logger.warning(message)
        elif level == 2:
            self.logger.info(message)
        else:
            raise ValueError("Invalid level. Level should be 0 (Error), 1 (Warning), or 2 (Info).")

    @property
    def trainer_callbacks(self) -> None:
        "Convert  list of callbacks"
        callbacks = []

        if self._callbacks:
            assert isinstance(self._callbacks, (List))
            if type(self._callbacks) == list:
                return self._callbacks
            else:
                return list(self._callbacks)

        else:
            return callbacks

    @property
    def trainer_loss(self):
        losses = []
        if self._loss is not None:
            # assert isinstance(self._loss, tf.keras.losses.Loss), f"Expected instance of tf.keras.losses.Loss, but got {type(self._loss)}"
            losses = self._loss

        elif "losses" in self.trainer_config.keys():
            for loss_name in self.trainer_config["losses"]:
                loss = LOSSES.get(loss_name)

                if loss is None:
                    message = f"{loss_name} is not implemented in LOSSES dict. Please check tf.keras.losses or provide a custom loss instance."
                    # warning
                    self.log(message=message, level=1)
                    # raise ValueError(message=message)
                else:
                    losses.append(loss())

        if len(losses) == 0:
            # message = "loss is not specified in config file or in loss object in __init__() method "
            message = "No loss is specified in config file or in loss object in __init__() method."
            self.log(message, level=0)
            raise ValueError(message)
        return losses

    @property
    def trainer_metrics(self):
        metrics = []
        if "metrics" in self.trainer_config.keys():
            for metric_name in self.trainer_config["metrics"]:
                metric = METRICS.get(metric_name)

                if metric is None:
                    message = f"{metric_name} is not implemented in METRICS dict. Please check tf.keras.metrics or provide a custom metric instance."
                    self.log(message=message, level=1)  # warning
                    continue

                metrics.append(metric())

        if self._metrics is not None:
            for metric in self._metrics:
                assert isinstance(metric, Metric), f"Expected instances of tf.keras.metrics.Metric, but got {type(metric)}"
                metrics.append(metric)

        return metrics

    @property
    def trainer_optimizer(self):
        optimizer = None
        if self._optimizer is not None:
            optimizer = self._trainer_optimizer
        # else:
        elif "optimizer" in self.trainer_config.keys():
            optimizer_params = self.trainer_config["optimizer"]
            optimizer_name = optimizer_params["name"]
            optimizer_params = optimizer_params["params"]
            optimizer = OPTIMIZERS.get(optimizer_name)
            if optimizer is None:
                message = f"{optimizer_name} is not implemented in OPTIMIZERS dict. Please check tf.keras.optimizers or provide a custom optimizer instance."
                self.log(message=message, level=1)
        else:
            message = "optimizer is not specified in config file or in optimizer object in __init__() method "
            self.log(message, level=0)
            raise ValueError(message)

        optimizer = optimizer(**optimizer_params)
        return optimizer

    def compile_model(self):
        self.uuid = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        self.loss = self.trainer_loss
        self.metrics = self.trainer_metrics
        self.optimizer = self.trainer_optimizer
        self.callbacks = self.trainer_callbacks

        self._model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        self.log(message="Model is completed", level=2)

    def fit(self, continue_training: bool = True) -> None:
        """
        Train tf keras model

        Parameters
        ----------
        continue_training : bool
            If True, continue training from last epoch. If False, recompile model and start training from scratch.
        """

        if not self._model._is_compiled or not continue_training:
            self.compile_model()

        self.log("Start training  ...", level=2)

        history = self._model.fit(self.train_data, epochs=self.trainer_config["epochs"], callbacks=self.callbacks, validation_data=self.val_data)
        self.history_dict[str(datetime.datetime.now())] = {"h": history.history, "id": self.uuid}

        current_time = str(datetime.datetime.now())
        self.log(f"Training ended at {current_time}. History saved with id {self.uuid}", level=2)

        if self.trainer_config["save_model"]:
            path = self.trainer_config["save_model_path"]
            path = os.path.join(path, self.uuid) 
            os.makedirs(path, exist_ok=True)
            self.save(path)
        #     self.save(self.trainer_config["save_model_path"])
    
    @property
    def history(self):
        return self.history_dict

    def evaluate(self) -> None:
        if self.val_data:
            self._model.evaluate(self.val_data)
        else:
            self.log(message="Validation data is not provided", level=0)
           
    def save(self, path: str, meta_data_name="meta_data.pkl", onnx_name="model.onnx") -> None:
        """
        Saves the model to Tensorflow SavedModel and optionally to ONNX format. Also saves related metadata and preprocessor objects.

        Parameters
        ----------
        path: str
            Directory path to save model and related files.
        meta_data_name: str, default: "meta_data.pkl"
            Filename to save metadata.
        onnx_name: str, default: "model.onnx"
            Filename to save model in ONNX format.
        """

        self.saver.save(path, meta_data_name, onnx_name)

    #     # check path 
    #     #if not os.path.exists(path):
    #     os.makedirs(path, exist_ok=True)
       
    #     tf_path = os.path.join(path, "tensorflow")
    #     os.makedirs(tf_path, exist_ok=True)

    #     onnx_path = os.path.join(path, "onnx")
    #     os.makedirs(onnx_path, exist_ok=True)

    #     meta_data_path = os.path.join(path, "meta_data")
    #     os.makedirs(name=meta_data_path, exist_ok=True)

    #     processors_path = os.path.join(path, "processors")
    #     os.makedirs(name=processors_path, exist_ok=True)

    #     # save model
    #     self._model.save(tf_path)
    #     self.log(message=f"Model is saved to {path}", level=2)

    #     # save meta data
    #     self._save_meta_data(meta_data_path, meta_data_name)
    #     self.log(message=f"Meta data is saved to {meta_data_path}", level=2)
        
    #     # save processors
    #     self._save_processors(processors_path, "processors.pkl")
    #     self.log(message=f"Processors are saved to {processors_path}", level=2)

    #     # save onnx
    #     if self.trainer_config["deploy_onnx"]:
    #         onnx_name = os.path.join(onnx_path, onnx_name)
    #         self._save_as_onnx(model=self._model, onnx_name=onnx_name)
    #         self.log(message=f"ONNX model is saved to {onnx_path}", level=2)


    # def _save_as_onnx(self, model, onnx_name: str, opset: int = 13) -> None:
    #     """
    #     Save model as onnx file

    #     Parameters
    #     ----------
    #     model: tf.keras.models.Model
    #         Model to save
    #     onnx_name: str
    #         Name of onnx file
    #     opset: int, default: 13
    #        opset version for onnx

    #     """

    #     save_model_as_onnx(model, onnx_name, opset=opset)

    # def _save_meta_data(self, path: str, filename: str) -> None:
    #     "Save meta data to model path"
    #     assert os.path.exists(path), f"Path: {path} does not exist"
    #     with open(path + "/" + filename, "wb") as f:
    #         pickle.dump(self.config, f)
        
    # def _save_processors(self, path: str, filename: str) -> None:
    #     "Save proto data to model path"
    #     pass
    #     # assert os.path.exists(path), f"Path: {path} does not exist"
    #     # with open(path + "/" + filename, "wb") as f:
    #     #     pickle.dump(self.proto_data, f)


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

        pass
        # try:
        #     _model = load_model(path)
        #     if checking_parameters:
        #         config = self.all_config["model"]
        #         print(config)
        #         print(config["input_shape"])

        #         # input shape check config and loaded model
        #         assert list(config["input_shape"]) == list(_model.input_shape[1:]), f"Input shape of model is not equal to config input shape {config['input_shape']} != {_model.input_shape[1:]}"

        #         # output shape check config and loaded model
        #         config_output_shape = list(config["input_shape"])
        #         config_output_shape[-1] = config["output_size"]
        #         assert config_output_shape == list(_model.output_shape[1:])

        #         # final activation check config and loaded model
        #         model_last_activation = _model.layers[-1].activation.__name__
        #         config_last_activation = config["final_activation"]
        #         assert model_last_activation == config_last_activation, f"Final activation of model is not equal to config final activation {config_last_activation} != {model_last_activation}"

        #     self._model = _model

        # except Exception as e:
        #     raise ValueError("Model could not be loaded", e)

    def test(self, data) -> None:
        self._model.evaluate(data)
