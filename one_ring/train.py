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
from one_ring.trace import initialize_mlflow, log_history, log_params, log_model,log_albumentation
from one_ring.callbacks import get_callbacks

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
        loss: Optional[Loss] = None,
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
        loss : tf.keras.losses.Loss, optional
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
        self._loss = loss
        self._metrics = metrics
        self._callbacks = list(callbacks.values()) if callbacks else None
        self._tracing_object = tracing_object
        self.saver = ModelSaver(model=self._model, config=self.config, processors=None)

        self._check_trainer_objects()  # check objects
        self._check_trainer_params()  # check parameters

        self.history = {}
        self.fit_counter = 0
        self.save_path = None

        if compiled_model is False:
            self._model._is_compiled = False

    def _check_trainer_objects(self) -> None:
        "Check parameters types"
        types = [
            (self.config, (dict, DictConfig, ListConfig)),
            (self._model, Model),
            (self.train_data, Dataset),
            (self.val_data, (Dataset, type(None))),
            (self._optimizer, (Optimizer,type(None))),
            (self._loss, (Loss, type(None))),
            (self._metrics, (list, type(None))),
            (self._callbacks, (list, type(None))),
       
        ]

        for obj, t in types:
            assert isinstance(obj, t), f"'{obj}' should be instance of {t}"

        if self._callbacks:
            for callback in self._callbacks:
                assert isinstance(callback, Callback), "a callback should be instance of tf.keras.callbacks.Callback"

        if self._metrics:
            for metric in self._metrics:
                assert isinstance(metric, Metric), " a member of 'metrics' should be instance of tf.keras.metrics.Metric"

    def _check_trainer_params(self):
        required_params = ["epochs", "experiment_name","save_model","verbose","deploy_onnx"]
        for param in required_params:
            assert self.trainer_config.get(param) is not None, f"{param} is required in trainer_config"

    @property
    def trainer_callbacks(self):
        # Direct return as _callbacks is always a list now
        if self.config.callbacks:
            callbacks = get_callbacks(self.config.callbacks)
            callbacks = list(callbacks.values())
            self._callbacks.extend(callbacks)

        return self._callbacks

    @property
    def trainer_loss(self):
        # Simplified handling for loss configuration
        loss = self._loss or self.trainer_config.get("losses")
        if isinstance(loss, str) and loss in LOSSES:
            return LOSSES[loss]()
        elif loss is None:
            self.logger.error("No loss specified.")
            raise ValueError("No loss specified in configuration or provided directly.")
        return loss

    @property
    def trainer_metrics(self):
        # Streamlined metrics assembly
        metrics = []
        if self.trainer_config.get("metrics"):
            metrics += [METRICS[name]() for name in self.trainer_config["metrics"] if name in METRICS]
        if self._metrics:
            metrics += [metric for metric in self._metrics if isinstance(metric, Metric)]
        return metrics

    @property
    def trainer_optimizer(self):
        # Corrected and optimized optimizer retrieval
        if self._optimizer:
            return self._optimizer
        optimizer_config = self.trainer_config.get("optimizer")
        if optimizer_config and "name" in optimizer_config and optimizer_config["name"] in OPTIMIZERS:
            optimizer_class = OPTIMIZERS[optimizer_config["name"]]
            optimizer_params = optimizer_config.get("params", {})
            return optimizer_class(**optimizer_params)
        self.logger.error("Optimizer configuration invalid or missing.")
        raise ValueError("Invalid optimizer configuration.")

    def _initialize_tracing(self):
        """
        Initialize mlflow
        """
        initialize_mlflow(self.trainer_config["experiment_name"], self.uuid)

    def _log_params(self):
        """
        Log configs to mlflow
        """
        log_part= ["trainer","data","model","callbacks"]
        log_params(self.config, log_part=log_part)
    
    def _log_aug_params(self):
        prefix = "aug"
        mlflow.log_param(f"{prefix}_type", self.config["augmentation"].aug_type)
        if self.config["augmentation"].aug_type == "albumentations":
            log_albumentation(self.config["augmentation"]["train"],prefix="aug_train")
            log_albumentation(self.config["augmentation"]["test"],prefix="aug_test")
        else:
            self.logger.error(f"Invalid augmentation type: {self.config['augmentation'].aug_type}")
            raise NotImplementedError

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
            self._log_aug_params()
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

    def save(self, path: str, root_dir="runs") -> None:
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
        # print("PATH", path)
        os.makedirs(path, exist_ok=True)
        self.save_path = path
        self.saver.save(path, train_ds=None, val_ds=None)

    def load(self, path: str = None, compile: bool = True, objects: dict = None) -> None:
        """
        Load a model from a TensorFlow SavedModel format.

        Parameters
        ----------
        path : str, optional
            The path to the directory where the model is saved. If not specified, `self.save_path` is used.
        compile : bool, default True
            Whether to compile the model after loading.
        objects : dict, optional
            A dictionary of custom objects necessary for loading the model, such as custom layers.
        """

        path = path or self.save_path

        if not os.path.isdir(path):
            self.logger.error("Path is not a valid directory.")
            raise ValueError("Path is not a valid directory.")

        try:
            self.model = self.saver.load(path, compile=compile, custom_objects=objects)
            self.logger.info(f"Model loaded successfully from '{path}'.")
        except Exception as e:
            self.logger.error(f"Failed to load the model from '{path}'. Error: {e}")
            raise

    def get_custom_objects(self):
        """
        Get custom objects from the model.

        Returns
        -------
        dict
            A dictionary of custom objects used in the model.
        """

        custom_objects = {}
        if self._model._is_compiled:
            for m in self.metrics:
                if hasattr(m, "one_ring_type"):
                    custom_objects[m.__class__.__name__] = m.__class__

            for l in self.loss:
                if hasattr(l, "one_ring_type"):
                    custom_objects[l.__class__.__name__] = l.__class__

        return custom_objects

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
