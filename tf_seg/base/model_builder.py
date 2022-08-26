"""
Model builder abstract class.

"""

from abc import ABC, abstractmethod
from tensorflow.keras.models import Model

class ModelBuilder(ABC):
    """Model builder abstract class. It is inherited by all the model builders."""

    @abstractmethod
    def build_model(self) -> Model:
        """Builds the model.

        Returns
        -------
        Model : tf.keras.model.Model
            The model.

        """
        pass