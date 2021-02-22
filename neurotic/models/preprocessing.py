import pandas as pd
import numpy as np

import tensorflow as tf

from pandas import DataFrame

from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers.experimental.preprocessing import (
    Normalization, StringLookup, CategoryEncoding
)


class DataframePreprocessor(Sequential):
    def __init__(self, data: DataFrame = None):
        super().__init__()
        self.features = data.to_numpy()

        inputs = []
        numeric_inputs = {}

        # create symbolic inputs for string and non-string columns
        for k, v in data.items():
            if v.dtype is object:
                # create an one-hot encoded input
                x = Input(shape=(1, ), name=k, dtype=tf.string)
                x = self.one_hot(data[k].to_numpy())
                inputs.append(x)
            else:
                # create a numeric input and collect it in numeric_outputs.
                # next, we'll concat and normalize these inputs into a single
                # one.
                x = Input(shape=(1, ), name=k, dtype=tf.float32)
                numeric_inputs[k] = x

        # combine numeric features in a Concatenation layer
        # and then pass it through a value Normalization layer...
        normalization = Normalization()
        normalization.adapt(data[numeric_inputs.keys()].to_numpy())
        normalized_numeric_input = normalization(
            Concatenate()(list(numeric_inputs.values()))
        )
        # add normalized concatenated numeric layer to inputs
        inputs.append(normalized_numeric_input)

        # Now we create a final computed layer that concatenates all of the
        # prepared inputs (encoded and normalized)
        if len(inputs) > 1:
            self.layers.append(Concatenate()(inputs))
        else:
            self.layers.append(inputs[0])

    @staticmethod
    def one_hot(names: np.array) -> Input:
        lookup = StringLookup(vocabulary=np.unique(names))
        one_hot = CategoryEncoding(max_tokens=lookup.vocab_size())
        return one_hot(lookup(names))

