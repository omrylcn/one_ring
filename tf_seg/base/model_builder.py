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

    def __str__(self):

        return f"model builder class for : {self.__class__.__name__}"

    def __repr__(self):

        return f"model builder class for : {self.__class__.__name__}"
