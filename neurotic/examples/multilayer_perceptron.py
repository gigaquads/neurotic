"""
This example trains multiple MultilayerPerceptronClassifiers, visualizing the
effects of different learning rates, PRNG seeds, and train/test splits.
"""

import os

from datetime import datetime

import numpy as np

from matplotlib import pyplot as pp

from neurotic.classifiers import MultilayerPerceptronClassifier, Layer
from .internal.data_loaders import titanic_train_test_split


def main():
    figure, axes = pp.subplots(7, 2)

    for i in range(len(axes)):
        # load training/test data
        (
            training_data,   _test_data,
            training_labels, _test_labels
        ) = (
            titanic_train_test_split(
                # % data reserved for testing:
                test_size=0.05 + (0.45 * np.random.random())
            )
        )

        # build and train the network
        classifier = MultilayerPerceptronClassifier([
            Layer(24), Layer(4), Layer(1)
        ]).train(
            labels=training_labels,
            inputs=training_data,
            epochs=np.random.randint(100, 7000),
            rate=0.0001 + (0.3 * np.random.random())
        )

        # draw accuracy and error plots
        show_labels = not bool(i)
        classifier.plot(axes[i,:], show_labels=show_labels, show=False)

    pp.show()


if __name__ == '__main__':
    exit(main())