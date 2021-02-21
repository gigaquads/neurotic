import pandas as pd
import numpy as np

import tensorflow as tf

from pandas import DataFrame

from tensorflow.keras import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers.experimental.preprocessing import (
    Normalization, StringLookup, CategoryEncoding
)


class DataLoader:
    def __init__(self, data: DataFrame = None):
        self.data = data
        self.features = np.array(self.data)
        # self.normalization_layer.adapt(self.features)

        # build dict of symbolic inputs
        inputs = {}
        string_inputs = []
        numeric_input_names = []
        numeric_inputs = []

        for k, v in self.data.items():
            dtype = tf.string if v.dtype is object else tf.float32
            if v.dtype is object:
                dtype = tf.string
                inputs[k] = Input(shape=(1, ), name=k, dtype=dtype)
                string_inputs.append(inputs[k])
            else:
                dtype = tf.float32
                inputs[k] = Input(shape=(1, ), name=k, dtype=dtype)
                numeric_input_names.append(k)
                numeric_inputs.append(inputs[k])


        normalization_layer = Normalization()
        normalization_layer.adapt(np.array(data[numeric_input_names]))

        preprocessed_inputs = [
            normalization_layer(Concatenate()(numeric_input_names))
        ]


    @staticmethod
    def one_hot_encode(names: np.array) -> Input:
        lookup = StringLookup(vocabulary=np.unique(names))
        one_hot = CategoryEncoding(max_tokens=lookup.vocab_size())
        return one_hot(lookup(names))

