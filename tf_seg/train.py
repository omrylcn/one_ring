"""

trainer class yapısı
- objelerin parametrelerini kontrol eder.
- objelerin eğitim ve evaluate işlemlerini yapar.
- 
"""


from typing import Union, Optional, List
from omegaconf import DictConfig, ListConfig
from tensorflow.keras import Model
from tensorflow.data import Dataset
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import *
from tf_seg.losses import *
from tf_seg.metrics import *
from tf_seg.utils import snake_case_to_pascal_case


class Trainer:
    "Trainer class for training a model"

    def __init__(
        self,
        config: Union[DictConfig, ListConfig],
        model: Model,
        train_data: Dataset,
        val_data: Optional[Dataset] = None,
        callbacks: Optional[List[Callback]] = None,
    ) -> None:
        """


        Parameters
        ----------
        config: Union[DictConfig, ListConfig]
            Configuration for training
        model: tf.keras.Model
            Model to be trained
        train_data: tf.data.Dataset
            Dataset for training
        val_data: tf.data.Dataset,default: None
            Dataset for validation
        callbacks: tf.keras.callbacks.Callback,default: None
            keras Callbacks for training

        """
        self.config = config
        self._model = model
        self.train_data = train_data
        self.val_data = val_data
        self.callbacks = callbacks

        self._check_params()

    def _check_params(self) -> None:
        "Check parameters types"
        assert isinstance(self.config, (DictConfig, ListConfig))
        assert isinstance(self._model, Model)
        assert isinstance(self.train_data, Dataset)

        if self.val_data:
            assert isinstance(self.val_data, Dataset)

        if self.callbacks:
            assert isinstance(self.callbacks, List)
            if len(self.callbacks) > 0:
                for callback in self.callbacks:
                    assert isinstance(callback, Callback)

    def _check_trainer_params(self):
        "Check trainer config parameters"

        assert self.config.epochs > 0
        assert self.config.batch_size > 0
        assert self.config.optimizer is not None
        assert self.config.losses is not None

    @property
    def model(self):
        print("test")
        self._check_trainer_params()

        optimizer_conf = self.config.optimizer
        optimizer = eval(snake_case_to_pascal_case(optimizer_conf["name"]))(
            **optimizer_conf["params"]
        )

        losses = [eval(snake_case_to_pascal_case(i))() for i in self.config.losses]
        metrics = [eval(snake_case_to_pascal_case(i))() for i in self.config.metrics]

        self._model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
        return self._model

    def train(self):
        "Train tf keras model"

        self.model.fit(
            self.train_data,
            epochs=self.config.epochs,
            callbacks=self.callbacks,
            validation_data=self.val_data,
        )

    def evaluate(self):
        if self.val_data:
            self.model.evaluate(self.val_data)
        else:
            raise ValueError("Validation data is not provided")

    def save(self, filename):
        self.model.save(filename)

    def test(self, data):
        self.model.evaluate(data)
