import os
import pickle
import json
from omegaconf import OmegaConf
from tensorflow.data import Dataset
from one_ring.config import ONNX_OPSET, META_DATA_NAME, ONNX_NAME, PROCESSORS_NAME
from one_ring.deploy import save_model_as_onnx
from one_ring.logger import Logger


class ModelSaver:
    """
    A utility class for saving machine learning models, their metadata, and processors
    to disk, with support for TensorFlow SavedModel format and optionally ONNX format.

    Parameters
    ----------
    model : tf.keras.Model
        The TensorFlow model to be saved.
    config : OmegaConf
        Configuration object containing model settings and parameters.
    processors : dict
        A dictionary of preprocessing objects that need to be saved along with the model.

    Attributes
    ----------
    model : tf.keras.Model
        TensorFlow model to be saved.
    config : OmegaConf
        Configuration object containing model settings and parameters.
    processors : dict
        Preprocessing objects to be saved with the model.
    onnx_opset : int
        The ONNX opset version to use for conversion.
    logger : Logger
        Logger object for logging information during the save process.

    Methods
    -------
    save(path: str, train_ds: Dataset, val_ds: Dataset = None,
         meta_data_name: str = 'meta_data.json', processors_name: str = 'processors.pkl',
         onnx_name: str = 'model.onnx') -> None
        Saves the model, metadata, and processors to the specified path.
    """

    def __init__(self, model, config, processors):
        self.model = model
        self.config = config
        self.processors = processors
        self.onnx_opset = ONNX_OPSET
        self.logger = Logger("one_ring", log_file="trainer_log.log")

    def save(
        self,
        path: str,
        train_ds: Dataset,
        val_ds: Dataset = None,
        meta_data_name: str = META_DATA_NAME,
        processors_name: str = PROCESSORS_NAME,
        onnx_name: str = ONNX_NAME,
    ) -> None:
        """
        Saves the model in TensorFlow SavedModel format and optionally in ONNX format.
        Also saves related metadata and preprocessor objects to disk.

        Parameters
        ----------
        path : str
            The base directory path where the model and related files will be saved.
        train_ds : Dataset
            The training dataset used for model training.
        val_ds : Dataset, optional
            The validation dataset used during model training (default is None).
        meta_data_name : str, optional
            The filename for saving the metadata (default is 'meta_data.json').
        processors_name : str, optional
            The filename for saving the processors (default is 'processors.pkl').
        onnx_name : str, optional
            The filename for saving the model in ONNX format (default is 'model.onnx').

        Returns
        -------
        None
        """

        tf_path = os.path.join(path, "tensorflow")
        os.makedirs(tf_path, exist_ok=True)
        self.model.save(tf_path)
        self.logger.info(f"Saved model to {tf_path}")

        meta_data_path = os.path.join(path, "meta_data")
        os.makedirs(meta_data_path, exist_ok=True)
        self._save_metadata(meta_data_path, meta_data_name)
        self.logger.info(f"Saved metadata to {meta_data_path}")

        processors_path = os.path.join(path, "processors")
        os.makedirs(processors_path, exist_ok=True)
        self._save_processors(processors_path, processors_name)
        self.logger.info(f"Saved processors to {processors_path}")

        if self.config.get("deploy_onnx", True):
            onnx_path = os.path.join(path, "onnx")
            os.makedirs(onnx_path, exist_ok=True)
            onnx_filepath = os.path.join(onnx_path, onnx_name)
            self._save_as_onnx(self.model, onnx_filepath)
            self.logger.info(f"Saved model to {onnx_filepath}")

    def _save_as_onnx(self, model: str, onnx_name: str, opset: int = 13) -> None:
        # Convert the tf.keras model to ONNX format and save to the provided filepath
        save_model_as_onnx(model, onnx_name=onnx_name, opset=self.onnx_opset)

    def _save_metadata(self, path: str, filename: str) -> None:
        filepath = os.path.join(path, filename)
        with open(filepath, "w") as f:
            OmegaConf.save(self.config, f)

    def _save_processors(self, path: str, filename: str) -> None:
        filepath = os.path.join(path, filename)
        with open(filepath, "wb") as f:
            pickle.dump(self.processors, f)


# Example usage
# model_saver = ModelSaver(model=my_tf_model, config=config_data, processors=processor_data)
# model_saver.save("/path/to/save")
