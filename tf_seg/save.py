import os
import pickle
from tensorflow.data import Dataset


class ModelSaver:
    # def __init__(self):
    #     pass

    @staticmethod
    def save(model, path: str, train_ds: Dataset, val_ds: Dataset = None, meta_data_name: str = "meta_data.pkl", onnx_name: str = "model.onnx") -> None:
        """
        Saves the model to Tensorflow SavedModel and optionally to ONNX format. Also saves related metadata and preprocessor objects.
        """

        tf_path = os.path.join(path, "tensorflow")
        os.makedirs(tf_path, exist_ok=True)

        onnx_path = os.path.join(path, "onnx")
        os.makedirs(onnx_path, exist_ok=True)

        meta_data_path = os.path.join(path, "meta_data")
        os.makedirs(meta_data_path, exist_ok=True)

        processors_path = os.path.join(path, "processors")
        os.makedirs(processors_path, exist_ok=True)

        # save model
        self._model.save(tf_path)

        # save model
        model.save(tf_path)

        # save meta data
        self._save_metadata(meta_data_path, meta_data_name)

        # save processors
        self._save_processors(processors_path, "processors.pkl")

        # save onnx
        if self.trainer_config.get("deploy_onnx", False):
            onnx_filepath = os.path.join(onnx_path, onnx_name)
            self._save_as_onnx(onnx_filepath)
            print(f"ONNX model is saved to {onnx_path}")

    @staticmethod
    def _save_as_onnx(self, filepath: str, opset: int = 13) -> None:
        # Convert the tf.keras model to ONNX format and save to the provided filepath
        # Implement ONNX saving logic here
        pass

    @staticmethod
    def _save_metadata(self, path: str, filename: str) -> None:
        filepath = os.path.join(path, filename)
        with open(filepath, "wb") as f:
            pickle.dump(self.config, f)

    @staticmethod
    def _save_processors(self, path: str, filename: str) -> None:
        filepath = os.path.join(path, filename)
        with open(filepath, "wb") as f:
            if self.processors:
                pickle.dump(self.processors, f)


# Example usage
# model_saver = ModelSaver(model=my_tf_model, config=config_data, processors=processor_data)
# model_saver.save("/path/to/save")
