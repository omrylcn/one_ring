import os
import pickle
from tensorflow.data import Dataset
from one_ring.config import ONNX_OPSET
from one_ring.deploy import save_model_as_onnx
from one_ring.logger import Logger


class ModelSaver:
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
        meta_data_name: str = "meta_data.pkl",
        onnx_name: str = "model.onnx",
    ) -> None:
        """
        Saves the model to Tensorflow SavedModel and optionally to ONNX format.
        Also saves related metadata and preprocessor objects.
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
        self._save_processors(processors_path, "processors.pkl")
        self.logger.info(f"Saved processors to {processors_path}")

        if self.config.get("deploy_onnx", False):
            onnx_path = os.path.join(path, "onnx")
            os.makedirs(onnx_path, exist_ok=True)
            onnx_filepath = os.path.join(onnx_path, onnx_name)
            self._save_as_onnx(onnx_filepath)
            self.logger.info(f"Saved model to {onnx_filepath}")
            

    def _save_as_onnx(self, model, onnx_name: str, opset: int = 13) -> None:
        # Convert the tf.keras model to ONNX format and save to the provided filepath
        save_model_as_onnx(model, onnx_name, opset=self.onnx_opset)

    def _save_metadata(self, path: str, filename: str) -> None:
        filepath = os.path.join(path, filename)
        with open(filepath, "wb") as f:
            pickle.dump(self.config, f)

    def _save_processors(self, path: str, filename: str) -> None:
        filepath = os.path.join(path, filename)
        with open(filepath, "wb") as f:
            pickle.dump(self.processors, f)


# Example usage
# model_saver = ModelSaver(model=my_tf_model, config=config_data, processors=processor_data)
# model_saver.save("/path/to/save")
