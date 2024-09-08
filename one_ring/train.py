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
import time
from tqdm.auto import tqdm
from typing import Union, Optional, List, Dict, Tuple
from omegaconf import DictConfig, ListConfig

import tensorflow as tf
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
from one_ring.trace import initialize_mlflow, log_history, log_params, log_model, log_albumentation
from one_ring.callbacks import get_callbacks
import numpy as np


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
        save_object: Optional[Dict] = None,
    ) -> None:
        """
        Initialize the Trainer with model, data, and training configurations.

        Parameters:
        -----------
        config : Union[DictConfig, ListConfig]
            Configuration settings for training.
        model : Model
            The TensorFlow model to train.
        train_data : Dataset
            The training dataset.
        val_data : Optional[Dataset]
            The validation dataset.
        optimizer : Optional[Optimizer]
            The optimizer to use for training.
        loss : Optional[Loss]
            The loss function for training.
        metrics : Optional[List[Metric]]
            List of metrics to evaluate during training.
        callbacks : Optional[List[Callback]]
            List of callbacks for the training process.
        save_object : Optional[Dict]
            Dictionary containing tracing objects (e.g., MLflow).
        """
        self.logger = Logger("one_ring", log_file="trainer_log.log")
        self.config = config.copy()
        self.train_config = config["train"].copy()
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.save_object = save_object

        self._setup_training_components(optimizer, loss, metrics, callbacks)
        self.saver = ModelSaver(model=self.model, config=self.config, processors=None)

        self.history = {"loss": [], "val_loss": [], "metrics": {}, "val_metrics": {}, "epoch_time": []}
        self.total_epoch = 0
        self.uuid = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.best_model = None
        self.best_performance = float("inf")

    def _setup_training_components(self, optimizer, loss, metrics, callbacks):
        """Set up optimizer, loss, metrics, and callbacks for training."""
        self.optimizer = self._create_optimizer(optimizer)
        self.loss = self._create_loss(loss)
        self.metrics = self._create_metrics(metrics)
        self.callbacks = self._create_callbacks(callbacks)

    def _create_optimizer(self, optimizer):
        """Create or validate the optimizer."""
        if isinstance(optimizer, Optimizer):
            return optimizer
        optimizer_config = self.train_config.get("optimizer", {})
        optimizer_name = optimizer_config.get("name")
        if optimizer_name in OPTIMIZERS:
            return OPTIMIZERS[optimizer_name](**optimizer_config.get("params", {}))
        raise ValueError(f"Invalid optimizer configuration. Supported optimizers: {', '.join(OPTIMIZERS.keys())}")

    def _create_loss(self, loss):
        """Create or validate the loss function."""
        if loss:
            return loss
        loss_name = self.config.train.loss
        if isinstance(loss_name, str) and loss_name in LOSSES:
            return LOSSES[loss_name]()
        raise ValueError(f"Invalid loss configuration. Supported losses: {', '.join(LOSSES.keys())}")

    def _create_metrics(self, metrics):
        """Create or validate the metrics."""
        config_metrics = [METRICS[name]() for name in self.config.train.metrics if name in METRICS]
        return (metrics or []) + config_metrics

    def _create_callbacks(self, callbacks):
        """Create or validate the callbacks."""
        config_callbacks = get_callbacks(self.config.callbacks)
        return (callbacks or []) + list(config_callbacks.values())

    def _to_serializable(self, obj: any) -> any:
        """Convert a TensorFlow object to a JSON-serializable format."""
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, tf.Tensor):
            return obj.numpy().tolist()
        elif isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: self._to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._to_serializable(value) for value in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif hasattr(obj, "__dict__"):
            return self._to_serializable(obj.__dict__)
        else:
            return str(obj)

    def get_metadata(self) -> Dict[str, any]:
        """Gather comprehensive metadata about the model and training process."""
        metadata = {
            "model": {
                "name": self.model.name,
                "layers": [layer.name for layer in self.model.layers],
                "trainable_params": int(self.model.count_params()),
            },
            "optimizer": {
                "name": self.optimizer.__class__.__name__,
                "config": self._to_serializable(self.optimizer.get_config()),
            },
            "loss": {
                "name": self.loss.__class__.__name__,
                "config": self._to_serializable(self.loss.get_config() if hasattr(self.loss, "get_config") else {}),
            },
            "metrics": [
                {"name": metric.__class__.__name__, "config": self._to_serializable(metric.get_config())}
                for metric in self.metrics
            ],
            "training": {
                "total_epochs": self.total_epoch,
                "best_performance": float(self.best_performance),
                "training_started": self.uuid,
                "training_completed": datetime.datetime.now().isoformat(),
            },
            "history": self.history,
            "transformer": {
                "val": self.save_object["transformer"].get("val", None),
                "train": self.save_object["transformer"].get("train", None),
            },
        }
        return metadata

    # def compile(self):
    #     self.uuid = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

    #     self.is_first = True
    #     self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
    #     self.logger.info(message="Model is completed")

    @tf.function
    def train_step(self, x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Perform a single training step.

        Parameters:
        -----------
        x : tf.Tensor
            Input features.
        y : tf.Tensor
            Target values.

        Returns:
        --------
        Tuple[tf.Tensor, tf.Tensor]
            Loss value and model predictions.
        """
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = self.loss(y, y_pred)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self._update_metrics(y, y_pred)
        return loss, y_pred

    def _update_metrics(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> None:
        """Update all metrics."""
        for metric in self.metrics:
            metric.update_state(y_true, y_pred)

    def fit(self, epochs: int = None, verbose: int = 1, finalize: bool = False) -> Dict[str, List]:
        """
        Train the model for a specified number of epochs.

        Parameters:
        -----------
        epochs : int, optional
            Number of epochs to train. If not provided, uses the value from the config.
        verbose : int, optional
            Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.

        Returns:
        --------
        Dict[str, List]
            Training history containing loss, metric values, and epoch times.
        """
        self._initialize_training(epochs)

        epoch_iterator = range(self.start_epoch, self.finish_epoch)

        for epoch in epoch_iterator:
            epoch_start_time = time.time()
            self._train_epoch(epoch, verbose)
            if self.val_data:
                self._validate_epoch(epoch, verbose)

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            self.history["epoch_time"].append(epoch_duration)

            self._log_epoch_results(epoch, verbose, epoch_duration)
            self._check_early_stopping()
            self._reset_metrics()

        if finalize:
            self.finalize_training()

        return self.history

    def _initialize_training(self, epochs):
        """Initialize training parameters and MLflow."""
        if self.total_epoch == 0:
            self._initialize_mlflow()

        self.total_epoch += epochs or self.train_config.epochs
        self.start_epoch = self.total_epoch - (epochs or self.train_config.epochs) + 1
        self.finish_epoch = self.total_epoch + 1

    def _train_epoch(self, epoch: int, verbose: int) -> None:
        """Train for one epoch."""
        epoch_loss = tf.keras.metrics.Mean()

        train_iterator = self.train_data
        if verbose == 1:
            total_steps = self.train_data.cardinality().numpy()
            if total_steps == tf.data.UNKNOWN_CARDINALITY:
                total_steps = None  # tqdm will show ? instead of total steps
            train_iterator = tqdm(self.train_data, total=total_steps, desc=f"Epoch {epoch}", leave=False)

        for x, y in train_iterator:
            loss, _ = self.train_step(x, y)
            epoch_loss.update_state(loss)

            if verbose == 1:
                train_iterator.set_postfix({"batch_loss": f"{loss.numpy():.4f}"})

        train_loss = epoch_loss.result().numpy().astype(float)
        train_metrics = {metric.name: metric.result().numpy().astype(float) for metric in self.metrics}
        self._add_history("train", train_loss, train_metrics)

        self._reset_metrics()

    def _validate_epoch(self, epoch: int, verbose: int) -> None:
        """Validate the model after an epoch."""
        val_loss, val_metrics = self.evaluate(self.val_data)
        self._update_best_model(epoch, val_loss)
        self._add_history("val", val_loss, val_metrics)
        self._reset_metrics()

    def _log_epoch_results(self, epoch: int, verbose: int, epoch_duration: float) -> None:
        """Log the results of an epoch."""
        if verbose > 0:
            train_results = self.history["metrics"]
            val_results = self.history["val_metrics"]

            log_message = (
                f"Epoch {epoch}/{self.total_epoch} - {epoch_duration:.2f}s - " f"loss: {self.history['loss'][-1]:.4f}"
            )

            for metric_name, metric_values in train_results.items():
                log_message += f" - {metric_name}: {metric_values[-1]:.4f}"

            if self.val_data:
                log_message += f" - val_loss: {self.history['val_loss'][-1]:.4f}"
                for metric_name, metric_values in val_results.items():
                    log_message += f" - val_{metric_name}: {metric_values[-1]:.4f}"

            self.logger.info(log_message)

    def evaluate(self, data: Dataset) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate the model on the given dataset.

        Parameters:
        -----------
        data : Dataset
            The dataset to evaluate on.

        Returns:
        --------
        Tuple[float, Dict[str, float]]
            A tuple containing the average loss and a dictionary of metric results.
        """
        total_loss = tf.keras.metrics.Mean()

        for x, y in data:
            y_pred = self.model(x, training=False)
            total_loss.update_state(self.loss(y, y_pred))
            self._update_metrics(y, y_pred)

        avg_loss = total_loss.result().numpy().astype(float)
        metric_results = {metric.name: metric.result().numpy().astype(float) for metric in self.metrics}

        return avg_loss, metric_results

    def _check_early_stopping(self) -> None:
        """Check if early stopping criteria are met."""
        if self.train_config.get("early_stopping", False):
            patience = self.train_config.get("early_stopping_patience", 5)
            min_delta = self.train_config.get("early_stopping_min_delta", 0.001)

            if len(self.history["val_loss"]) > patience:
                recent_losses = self.history["val_loss"][-patience:]
                if all(recent_losses[i] - recent_losses[i + 1] < min_delta for i in range(len(recent_losses) - 1)):
                    self.logger.info("Early stopping criteria met. Halting training.")
                    self.finish_epoch = self.total_epoch + 1

    def _reset_metrics(self) -> None:
        """Reset all metrics."""
        for metric in self.metrics:
            metric.reset_states()

    def _initialize_mlflow(self) -> None:
        """Initialize MLflow tracking."""
        initialize_mlflow(self.train_config["experiment_name"], self.uuid)
        self._log_params()

    def _log_params(self) -> None:
        """Log configuration parameters to MLflow."""
        log_params(self.config, log_part=["train", "data", "model", "callbacks"])

    def _update_best_model(self, epoch: int, val_loss: float) -> None:
        """Update the best model if current performance is better."""
        if val_loss < self.best_performance:
            self.best_performance = val_loss
            self.best_model = self.model.get_weights()
            self.logger.info(f"Epoch {epoch} - New best model found with validation loss: {val_loss:.4f}")

    def finalize_training(self) -> None:
        """Perform final operations after training."""
        if self.best_model is not None:
            self.model.set_weights(self.best_model)
            self.logger.info("Restored best model weights.")

        self.save()
        self._log_final_results()
        mlflow.end_run()
        self.logger.info("Finished model training")
        

    def _add_history(self, mode: str, loss: float, metrics: Dict[str, float]) -> None:
        """Add training or validation results to history."""
        if mode == "train":
            self.history["loss"].append(loss)
            for name, value in metrics.items():
                self.history["metrics"].setdefault(name, []).append(value)
        else:
            self.history["val_loss"].append(loss)
            for name, value in metrics.items():
                self.history["val_metrics"].setdefault(name, []).append(value)

    def save(self) -> None:
        """Save the trained model and comprehensive metadata using ModelSaver."""
        save_path = os.path.join(self.train_config.get("model_save_dir", "models"), f"{self.uuid}")
        os.makedirs(save_path, exist_ok=True)

        self.metadata = self.get_metadata()

        self.saver.save(
            path=save_path, train_ds=self.train_data, val_ds=self.val_data, additional_metadata=self.metadata
        )

        self.logger.info(f"Model and comprehensive metadata saved to {save_path}")
        self.save_path = save_path

    def load(self, path: str, setup_components: bool = False) -> None:
        """
        Load a previously saved model and its associated data.

        Parameters:
        -----------
        path : str
            The path to the directory where the model and associated data are saved.

        setup_components : bool
            Recreate training components from config files

        """

        loaded_data = self.saver.load(path)

        self.model = loaded_data["model"]
        self.loaded_metadata = loaded_data["metadata"]

        if setup_components:
            self._setup_training_components(None, None, None, None)

        self.logger.info(f"Model and associated data loaded from {path}")

    def _log_final_results(self) -> None:
        """Log final training results."""
        log_history(self.history)
        # log_model(self.model, self.train_config.get("model_save_dir", "models"))

    # def _log_aug_params(self):
    #     prefix = "aug"
    #     mlflow.log_param(f"{prefix}_type", self.config["augmentation"].aug_type)
    #     if self.config["augmentation"].aug_type == "albumentations":
    #         log_albumentation(self.config["augmentation"]["train"],prefix="aug_train")
    #         log_albumentation(self.config["augmentation"]["test"],prefix="aug_test")
    #     else:
    #         self.logger.error(f"Invalid augmentation type: {self.config['augmentation'].aug_type}")
    #         raise NotImplementedError

    # def _log_model(self):
    #     """
    #     Log model to mlflow
    #     """
    #     log_model(self.model, str(self.uuid))

    # def _log_history(self, history, **kwargs):
    #     """
    #     Log history to mlflow
    #     """
    #     log_history(history, **kwargs)

    # def _log_object(self):
    #     if self._tracing_object:
    #         if "mlflow" in list(self._tracing_object.keys()):
    #             for key, value in self._tracing_object["mlflow"].items():
    #                 for k, v in value.items():
    #                     new_k = f"{key}_{k}"
    #                     mlflow.log_param(new_k, v)

    # if self.val_data:
    #     val_loss = self.evaluate(self.val_data)
    #     self.logger.info(f"Validation Loss: {val_loss:.4f}")

    #     self._log_history({'loss': epoch_loss.result().numpy()}, epoch=epoch)

    # self.fit_counter += 1

    # if self.config.save_model:
    #     self.save(self.config.save_model_path)

    # initiliaze everything

   

    # def end(self, log_model: bool = False) -> None:
    #     if log_model:
    #         try:
    #             self._log_model()

    #         except Exception as e:
    #             self.logger.error(message=f"log_model method could not be executed. {e}")

    #     self.logger.info("Training ended")
    #     mlflow.end_run()

    # def test(self, data) -> None:
    #     self.model.evaluate(data)


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
