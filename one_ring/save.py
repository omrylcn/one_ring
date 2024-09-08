import os
import pickle
import json
from typing import Dict, Any, Optional
from omegaconf import OmegaConf
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.models import load_model
from one_ring.config import ONNX_OPSET, META_DATA_NAME, ONNX_NAME, PROCESSORS_NAME
from one_ring.deploy import save_model_as_onnx
from one_ring.logger import Logger

class ModelSaver:
    """
    A utility class for saving and loading machine learning models, their metadata,
    and processors to/from disk, using TensorFlow SavedModel format and optionally ONNX format.
    """

    def __init__(self, model: tf.keras.Model, config: OmegaConf, processors: Dict[str, Any]):
        self.model = model
        self.config = config
        self.processors = processors
        self.onnx_opset = ONNX_OPSET
        self.logger = Logger("one_ring", log_file="model_saver_log.log")

    def save(
        self,
        path: str,
        train_ds: Dataset,
        val_ds: Optional[Dataset] = None,
        meta_data_name: str = META_DATA_NAME,
        processors_name: str = PROCESSORS_NAME,
        onnx_name: str = ONNX_NAME,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Saves the model, metadata, processors, and additional information to disk using TensorFlow SavedModel format.

        Parameters:
        -----------
        path : str
            Base directory path for saving.
        train_ds : Dataset
            Training dataset used for model training.
        val_ds : Optional[Dataset]
            Validation dataset used during model training.
        meta_data_name : str
            Filename for saving metadata.
        processors_name : str
            Filename for saving processors.
        onnx_name : str
            Filename for saving the model in ONNX format.
        additional_metadata : Optional[Dict[str, Any]]
            Additional metadata to save with the model.
        """
        os.makedirs(path, exist_ok=True)

        # Save TensorFlow model
        tf_path = os.path.join(path, "saved_model")
        self._save_model(self.model, tf_path)
        self.logger.info(f"Saved TensorFlow model to {tf_path}")

        # Save metadata
        meta_data_path = os.path.join(path, "meta_data")
        os.makedirs(meta_data_path, exist_ok=True)
        self._save_metadata(meta_data_path, meta_data_name, additional_metadata)
        self.logger.info(f"Saved metadata to {meta_data_path}")

        # Save processors
        processors_path = os.path.join(path, "processors")
        os.makedirs(processors_path, exist_ok=True)
        self._save_processors(processors_path, processors_name)
        self.logger.info(f"Saved processors to {processors_path}")

        # Save ONNX model if configured
        if self.config.get("deploy_onnx", True):
            onnx_path = os.path.join(path, "onnx")
            os.makedirs(onnx_path, exist_ok=True)
            onnx_filepath = os.path.join(onnx_path, onnx_name)
            self._save_as_onnx(self.model, onnx_filepath)
            self.logger.info(f"Saved ONNX model to {onnx_filepath}")

        # Save dataset information
        self._save_dataset_info(path, train_ds, val_ds)

    def load(
        self,
        path: str,
    ) -> Dict[str, Any]:
        """
        Load the model and associated data from the provided filepath.

        Parameters:
        -----------
        path : str
            Directory path where the model and related files are saved.
        compile : bool
            Whether to compile the model after loading.

        Returns:
        --------
        Dict[str, Any]
            Dictionary containing the loaded model, metadata, processors, and other information.
        """
        loaded_data = {}

        # Load TensorFlow model
        tf_path = os.path.join(path, "saved_model")
        loaded_data['model'] = self._load_model(tf_path)
        self.logger.info(f"Loaded TensorFlow model from {tf_path}")

        # Load metadata
        meta_data_path = os.path.join(path, "meta_data", META_DATA_NAME)
        loaded_data['metadata'] = self._load_metadata(meta_data_path)
        self.logger.info(f"Loaded metadata from {meta_data_path}")

        # Load processors
        processors_path = os.path.join(path, "processors", PROCESSORS_NAME)
        loaded_data['processors'] = self._load_processors(processors_path)
        self.logger.info(f"Loaded processors from {processors_path}")

        # Load dataset information
        dataset_info_path = os.path.join(path, "dataset_info.json")
        if os.path.exists(dataset_info_path):
            with open(dataset_info_path, 'r') as f:
                loaded_data['dataset_info'] = json.load(f)
            self.logger.info(f"Loaded dataset information from {dataset_info_path}")

        return loaded_data

    def _save_model(self, model: tf.keras.Model, path: str) -> None:
        model.save(path)

    def _save_as_onnx(self, model: tf.keras.Model, onnx_name: str) -> None:
        save_model_as_onnx(model, onnx_name=onnx_name, opset=self.onnx_opset)

    def _save_metadata(self, path: str, filename: str, additional_metadata: Optional[Dict[str, Any]] = None) -> None:
        filepath = os.path.join(path, filename)
        metadata = OmegaConf.to_container(self.config, resolve=True)
        if additional_metadata:
            metadata.update(additional_metadata)
        with open(filepath, "w") as f:
            json.dump(metadata, f, indent=2)

    def _save_processors(self, path: str, filename: str) -> None:
        filepath = os.path.join(path, filename)
        with open(filepath, "wb") as f:
            pickle.dump(self.processors, f)

    def _save_dataset_info(self, path: str, train_ds: Dataset, val_ds: Optional[Dataset] = None) -> None:
        dataset_info = {
            "train_size": self._get_dataset_size(train_ds),
            "val_size": self._get_dataset_size(val_ds) if val_ds else None,
            "train_steps": self._get_dataset_steps(train_ds),
            "val_steps": self._get_dataset_steps(val_ds) if val_ds else None,
        }
        filepath = os.path.join(path, "dataset_info.json")
        with open(filepath, "w") as f:
            json.dump(dataset_info, f, indent=2)

    def _load_model(self, path: str) -> tf.keras.Model:
        model = tf.keras.models.load_model(path)
        return model

    def _load_metadata(self, path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return json.load(f)

    def _load_processors(self, path: str) -> Dict[str, Any]:
        with open(path, "rb") as f:
            return pickle.load(f)

    def _get_dataset_size(self, dataset: Optional[Dataset]) -> Optional[int]:
        if dataset is None:
            return None
        try:
            return int(dataset.cardinality().numpy())
        except:
            return None

    def _get_dataset_steps(self, dataset: Optional[Dataset]) -> Optional[int]:
        if dataset is None:
            return None
        try:
            return len(list(dataset.as_numpy_iterator()))
        except:
            return None

    def update_model(self, new_model: tf.keras.Model, path: str, additional_metadata: Optional[Dict[str, Any]] = None):
        """
        Update the saved model with a new model, typically after retraining.

        Parameters:
        -----------
        new_model : tf.keras.Model
            The new model to save.
        path : str
            Directory path where the model is saved.
        additional_metadata : Optional[Dict[str, Any]]
            Additional metadata to update.
        """
        tf_path = os.path.join(path, "saved_model")
        self._save_model(new_model, tf_path)
        self.logger.info(f"Updated TensorFlow model in {tf_path}")

        if additional_metadata:
            meta_data_path = os.path.join(path, "meta_data", META_DATA_NAME)
            current_metadata = self._load_metadata(meta_data_path)
            current_metadata.update(additional_metadata)
            self._save_metadata(os.path.join(path, "meta_data"), META_DATA_NAME, current_metadata)
            self.logger.info(f"Updated metadata in {meta_data_path}")

        if self.config.get("deploy_onnx", True):
            onnx_path = os.path.join(path, "onnx", ONNX_NAME)
            self._save_as_onnx(new_model, onnx_path)
            self.logger.info(f"Updated ONNX model in {onnx_path}")