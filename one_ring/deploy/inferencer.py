import os
import logging
import hashlib
from functools import lru_cache
from typing import Dict,Tuple, Any
#import json
import yaml

# import matplotlib.pyplot as plt 
# import cv2
import numpy as np
#from omegaconf import DictConfig, ListConfig
from one_ring.utils import TensorLike
from one_ring.config import MODEL_TYPE_LIB
from one_ring.transformers import normalize
from one_ring.deploy import model_wrapper_lib,preprocessor_lib,postprocessor_lib


import numpy as np
from functools import lru_cache
from typing import Tuple
import matplotlib.pyplot as plt

class Inferencer:
    """
    Inference class for one_ring models with improved caching and logging
    """

    def __init__(self, config: Dict[str, Any], cache_size: int = 128, log_level: str = "INFO"):
        """
        Initialize the Inferencer with a configuration dictionary.

        Parameters
        ----------
        config : Dict
            Configuration dictionary containing all necessary parameters.
        cache_size : int, optional
            Size of the LRU cache for preprocessing results, by default 128
        log_level : str, optional
            Logging level, by default "INFO"
        """
        self.config = config
        self.cache_size = cache_size
        self._setup_logging(log_level)
        self._validate_config()

        self._model = self._load_model()
        self.metadata = self._load_metadata()

        self.preprocessor = self.load_processor("preprocessor")
        self.postprocessor = self.load_processor("postprocessor")

    def _load_metadata(self):
        self.metadata_path = os.path.join(self.config["model_path"], "meta_data/meta_data.yaml")
        with open(self.metadata_path, "r") as file:
            metadata = yaml.safe_load(file)

        return metadata
        # self.meta_data = open(self.meta_data_path,"r").read()

    def _setup_logging(self, log_level: str):
        """Set up logging for the Inferencer."""
        logging.basicConfig(level=log_level)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _validate_config(self):
        """Validate the configuration dictionary."""
        required_keys = [
            "model_type",
            "model_path",
            "preprocessor_type",
            "postprocessor_type",
        ]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")

        if self.config["model_type"] not in MODEL_TYPE_LIB:
            raise ValueError(f"Unsupported model type: {self.config['model_type']}")

    def _load_model(self):
        """Load the model based on the configuration."""
        self.logger.info(f"Loading model of type {self.config['model_type']}")
        return model_wrapper_lib[self.config["model_type"]](**self.config)

    def load_processor(self, processor_type: str):
        """Load a processor (preprocessor or postprocessor) based on the configuration."""
        processor_config_type = f"{processor_type}_type"
        #     processor_config_path = f"{processor_type}_path"

        if not self.config.get(processor_config_type):
            return None

        processor_lib = preprocessor_lib if processor_type == "preprocessor" else postprocessor_lib
        if self.config[processor_config_type] not in processor_lib:
            raise ValueError(f"Unsupported {processor_type} type: {self.config[processor_config_type]}")

        self.logger.info(f"Loading {processor_type} of type {self.config[processor_config_type]}")

        return processor_lib[self.config[processor_config_type]](
            self.config, metadata=self.metadata["transformer"]["val"]
        )

    def _hash_array(self, arr: np.ndarray) -> str:
        """Create a hash of a numpy array for caching purposes."""
        return hashlib.md5(arr.data.tobytes()).hexdigest()

    @lru_cache(maxsize=128)
    def _cached_preprocess(self, arr_hash: str) -> np.ndarray:
        """Cached preprocessing for improved performance."""
        # This method now only serves as a cache key
        # The actual preprocessing is done in pre_process
        pass

    def pre_process(self, x: np.ndarray) -> np.ndarray:
        """Preprocess the input data."""
        self.logger.debug("Preprocessing input")
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        arr_hash = self._hash_array(x)
        
        # Check if the result is in cache
        cached_result = self._cached_preprocess(arr_hash)
        if cached_result is not None:
            return cached_result

        if self.preprocessor:
            x = self.preprocessor(x)

        x = np.expand_dims(x, axis=0) if x.ndim == 3 else x

        self._cached_preprocess.cache_clear()
        self._cached_preprocess(arr_hash)
        
        return x
    
    def post_process(self, x: TensorLike) -> Tuple[TensorLike, TensorLike]:
        """Postprocess the model output."""
        self.logger.debug("Postprocessing output")
        return self.postprocessor(x) if self.postprocessor else (x, x)
    
    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform prediction on the input image.
        """
        self.logger.info("Starting prediction")
        try:
            processed_input = self.pre_process(image)
           
            self.logger.debug("Input processed successfully")

            prediction = self._model(processed_input)
            self.logger.debug("Model prediction completed")

            pred_image, pred_mask = self.post_process(prediction)
            self.logger.info("Prediction completed successfully")

           
            return pred_image, pred_mask
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise

    def __call__(self, image: TensorLike) -> Tuple[TensorLike, TensorLike]:
        """Allow the Inferencer to be called as a function."""
        return self.predict(image)

    def clear_cache(self):
        """Clear the preprocessing cache."""
        self._cached_preprocess.cache_clear()
        self.logger.info("Preprocessing cache cleared")