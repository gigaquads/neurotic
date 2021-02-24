import os

from typing import List, Union, Optional, Text, Callable

from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.keras import Model
from tensorflow.keras.callbacks import History
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.metrics import Metric
from tensorflow.keras.losses import Loss


class Trainer:
    def __init__(
        self,
        loss: Union[Text, Loss] = 'mean_squared_error',
        optimizer: Union[Text, Optimizer] = 'adam',
        metrics: Optional[List[Union[Text, Metric]]] = None,
        callbacks: Optional[List[Callable]] = None,
    ):
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.callbacks = callbacks

    def train(
        self,
        model: Model,
        epochs: int,
        ds_train: Dataset,
        ds_validation: Optional[Dataset] = None,
        workers: Optional[int] = None,
    ) -> History:
        model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=self.metrics
        )
        history = model.fit(
            x=ds_train,
            epochs=epochs,
            validation_data=ds_validation,
            workers=workers or max(1, os.cpu_count() // 2),
            callbacks=self.callbacks,
        )
        return history