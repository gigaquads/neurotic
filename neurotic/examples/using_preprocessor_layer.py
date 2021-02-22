import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from neurotic.layers.preprocessing import DataframePreprocessor

from .internal.data_loaders import titanic_train_test_split


def main():
    # load training/test data
    (
        features,
        training_data,   _test_data,
        training_labels, _test_labels
    ) = (
        titanic_train_test_split(
            # % data reserved for testing:
            test_size=0.05 + (0.45 * np.random.random())
        )
    )

    model = Sequential([
        DataframePreprocessor(features),
        Dense(64),
        Dense(1),
    ])

    model.compile(
        loss=tf.losses.BinaryCrossentropy(from_logits=True),
        optimizer='adam'
    )

    model.fit(training_data, training_labels, epochs=16)


if __name__ == '__main__':
    exit(main())