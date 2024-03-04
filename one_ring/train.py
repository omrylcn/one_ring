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


from typing import Union, Optional, List, Dict
from omegaconf import DictConfig, ListConfig

from tensorflow.keras.models import Model, load_model
from tensorflow.data import Dataset
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Optimizer, SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric

from one_ring.losses import LOSSES
from one_ring.metrics import METRICS
from one_ring.logger import Logger

# from one_ring.utils import snake_case_to_pascal_case
import mlflow
from one_ring.save import ModelSaver
from one_ring.trace import initialize_mlflow, log_history, log_params, log_model


OPTIMIZERS = {
    "sgd": SGD,
    "rmsprop": RMSprop,
    "adam": Adam,
    "adadelta": Adadelta,
    "adagrad": Adagrad,
    "adamax": Adamax,
    "nadam": Nadam,
    "ftrl": Ftrl,
}


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
        losses: Optional[List[Loss]] = None,
        metrics: Optional[List[Metric]] = None,
        callbacks: Optional[List[Callback]] = None,
        compiled_model: bool = False,
        tracing_object: Optional[Dict] = None,
    ) -> None:
        """
        A class for training a model, encapsulating the setup for training and validation processes.

        Parameters
        ----------
        config : Union[DictConfig, ListConfig]
            Configuration settings for training, including hyperparameters and environment settings.
        model : tf.keras.Model
            The model to train.
        train_data : tf.data.Dataset
            The dataset used for training.
        val_data : tf.data.Dataset, optional
            The dataset used for validation. Default is None.
        optimizer : tf.keras.optimizers.Optimizer, optional
            The optimizer to use during training. Default is None.
        loss : List[tf.keras.losses.Loss], optional
            The loss function or functions to use. Default is None.
        metrics : List[tf.keras.metrics.Metric], optional
            The list of metrics to evaluate during training. Default is None.
        callbacks : List[tf.keras.callbacks.Callback], optional
            A list of callbacks for custom actions during training. Default is None.
        compiled_model : bool, optional
            Indicates if the model is already compiled. If False, the model will be compiled in the fit method. Default is False.
        tracing_object : Dict, optional
            A dictionary containing tracing objects such as mlflow, tensorboard, etc. Default is None.
        Attributes
        ----------
        The Trainer class does not explicitly define attributes in the __init__ method documentation, as attributes are typically
        those that are accessible on the class instance and are intended for use outside of the class's internal mechanisms.
        However, the parameters provided to __init__ are used to initialize the class's internal state.

        Notes
        -----
        The Trainer class is designed to abstract away the boilerplate code associated with training loops in TensorFlow,
        allowing users to focus on configuring their models, datasets, and training procedures without worrying about the
        underlying mechanics of the training process.

        Examples
        --------
        Assuming `config`, `model`, `train_data`, and optionally `val_data`, `optimizer`, `loss`, `metrics`, `callbacks`
        are defined appropriately:

        >>> trainer = Trainer(
                config=config,
                model=model,
                train_data=train_data,
                val_data=val_data,
                optimizer=optimizer,
                loss=loss,
                metrics=metrics,
                callbacks=callbacks,
                compiled_model=False
            )
        >>> trainer.fit()  # This method would need to be defined within the Trainer class.
        """
        self.logger = Logger("one_ring", log_file="trainer_log.log")

        self.config = config.copy()
        self.trainer_config = config["trainer"].copy()
        self._model = model
        self.train_data = train_data
        self.val_data = val_data
        self._optimizer = optimizer
        self._loss = losses
        self._metrics = metrics
        self._callbacks = list(callbacks.values()) if callbacks else None
        self._tracing_object = tracing_object
        self.saver = ModelSaver(model=self._model, config=self.config, processors=None)

        self._check_trainer_objects()  # check objects
        self._check_trainer_params()  # check parameters

        self.history = {}
        self.fit_counter = 0

        if compiled_model is False:
            self._model._is_compiled = False

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
                assert isinstance(
                    self._callbacks, (Callback)
                ), "a callback should be instance of tf.keras.callbacks.Callback"

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
                    assert isinstance(
                        metric, Metric
                    ), " a member of 'metrics' should be instance of tf.keras.metrics.Metric"

    def _check_trainer_params(self):
        "Check trainer config parameters"
        assert self.trainer_config["epochs"] > 0
        assert self.trainer_config["experiment_name"] is not None
        assert self.trainer_config["save_model_path"] is not None
        assert (
           "start" in self.trainer_config["lr"].keys() and "end" in self.trainer_config["lr"].keys()
        ), f"lr should have start and end keys, not :{self.trainer_config['lr']}"
        assert self.trainer_config["optimizer"] is not None or self._optimizer is not None
        assert self.trainer_config["losses"] is not None or self._loss is not None

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
                    self.logger.warning(message=message)
                    # raise ValueError(message=message)
                else:
                    losses.append(loss())

        if len(losses) == 0:
            # message = "loss is not specified in config file or in loss object in __init__() method "
            message = "No loss is specified in config file or in loss object in __init__() method."
            self.logger.error(message)
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
                assert isinstance(
                    metric, Metric
                ), f"Expected instances of tf.keras.metrics.Metric, but got {type(metric)}"
                metrics.append(metric)

        return metrics

    @property
    def trainer_optimizer(self):
        optimizer: Optimizer = None
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
                self.logger.warning(message=message)
        else:
            message = "optimizer is not specified in config file or in optimizer object in __init__() method "
            self.logger.error(message)
            raise ValueError(message)

        optimizer = optimizer(learning_rate=self.trainer_config["lr"]["start"], **optimizer_params)
        return optimizer

    def _initialize_tracing(self):
        """
        Initialize mlflow
        """
        initialize_mlflow(self.trainer_config["experiment_name"], self.uuid)

    def _log_params(self):
        """
        Log configs to mlflow
        """
        log_params(self.config)

    def _log_model(self):
        """
        Log model to mlflow
        """
        log_model(self._model, str(self.uuid))

    def _log_history(self, history, **kwargs):
        """
        Log history to mlflow
        """
        log_history(history, **kwargs)

    def _log_object(self):
        if self._tracing_object:
            if "mlflow" in list(self._tracing_object.keys()):
                for key, value in self._tracing_object["mlflow"].items():
                    for k, v in value.items():
                        new_k = f"{key}_{k}"
                        mlflow.log_param(new_k, v)

    def compile_model(self):
        self.uuid = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        self.loss = self.trainer_loss
        self.metrics = self.trainer_metrics
        self.optimizer = self.trainer_optimizer
        self.callbacks = self.trainer_callbacks

        self._model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        self.logger.info(message="Model is completed")

    def fit(self, continue_training: bool = True) -> None:
        """
        Train tf keras model

        Parameters
        ----------
        continue_training : bool
            If True, continue training from last epoch. If False, recompile model and start training from scratch.
        """

        # initiliaze everything
        if not self._model._is_compiled or not continue_training:
            self.compile_model()
            self.fit_counter = 0
            self.history = {}
            initial_epoch = 0
            self._initialize_tracing()
            self._log_params()
            self._log_object()
            self.logger.info("Start training  ...")
        else:
            self.logger.info("Continue training ...")
            initial_epoch = self.trainer_config["epochs"] * self.fit_counter

        epochs = self.trainer_config["epochs"] * (self.fit_counter + 1)

        history = self._model.fit(
            self.train_data,
            epochs=epochs,
            callbacks=self.callbacks,
            validation_data=self.val_data,
            initial_epoch=initial_epoch,
        )

        current_time = str(datetime.datetime.now())
        self.logger.info(f"Training ended at {current_time}. History saved with id {self.uuid}")
        self.fit_counter += 1

        self._log_history(history, initial_epoch=initial_epoch)
        self._update_history(history)

        if self.trainer_config["save_model"]:
            path = self.trainer_config["save_model_path"]
            self.save(path)

    def _update_history(self, history):
        for key, value in history.history.items():
            if key in self.history:
                self.history[key].extend(value)
            else:
                self.history[key] = value

    def evaluate(self) -> None:
        if self.val_data:
            self._model.evaluate(self.val_data)
        else:
            self.logger.error(message="Validation data is not provided")

    def save(self, path: str, root_dir="runs", meta_data_name="meta_data.pkl", onnx_name="model.onnx") -> None:
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

        path = os.path.join(root_dir, path, self.uuid)
        print("PATH", path)
        os.makedirs(path, exist_ok=True)
        self.saver.save(path, meta_data_name, onnx_name)

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

    def end(self, log_model: bool = False) -> None:
        if log_model:
            try:
                self._log_model()

            except Exception as e:
                self.logger.error(message=f"log_model method could not be executed. {e}")

        self.logger.info("Training ended")
        mlflow.end_run()

    def test(self, data) -> None:

        self._model.evaluate(data)


# def log(self, message: str, level: int = 2) -> None:
#     """
#     Logs a message with a certain level of severity.

#     Parameters
#     ----------
#     message : str
#         The message to log.
#     level : int, optional
#         The level of the message (0 - Error, 1 - Warning, 2 - Info).
#         Default is 2 (Info).
#     """
#     if level == 0:
#         self.logger.error(message)
#     elif level == 1:
#         self.logger.warning(message)
#     elif level == 2:
#         self.logger.info(message)
#     else:
#         raise ValueError("Invalid level. Level should be 0 (Error), 1 (Warning), or 2 (Info).")
