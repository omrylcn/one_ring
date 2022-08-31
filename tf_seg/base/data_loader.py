"""
Data loader  abstract class.

"""

from abc import ABC, abstractclassmethod
import tensorflow as tf


class DataLoaderAbstract(ABC):
    """Data  loader abstract class. It is inherited bt all the data loaders"""

    @abstractclassmethod
    def load_data(self) -> tf.data.Dataset:
        """
        load data abstract method
        """

        pass
