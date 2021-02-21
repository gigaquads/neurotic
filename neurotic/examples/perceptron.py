"""
This example plots a handful of perceptrons, each trained on 2 random
clusters of points with varying degrees of overlap, size, and density.

The test data consists of clusters that are similar to the training clusters
in their distributions.
"""

import numpy as np

from matplotlib import pyplot as pp

from neurotic.classifiers import PerceptronClassifier


def build_cluster_centered_on(count, center, radius=None) -> np.array:
    radius = radius or (0.75 + np.random.random() * 2)
    return center + (radius * np.random.randn(count, 2))


if __name__ == '__main__':
    figure, axes = pp.subplots(3, 5)

    for i in range(len(axes)):
        for j in range(len(axes[0, :])):
            size = np.random.randint(50, 500)
            centers = np.random.randint(1, 16, (2, 2))
            clusters = [
                # training clusters:
                build_cluster_centered_on(size, center=centers[0]),
                build_cluster_centered_on(size, center=centers[1]),

                # test clusters:
                build_cluster_centered_on(size, center=centers[0]),
                build_cluster_centered_on(size, center=centers[1]),
            ]

            classifier = PerceptronClassifier()
            classifier.train(clusters[:2], epochs=5000, rate=0.075)

            accuracy = classifier.test(clusters[2:])
            print(f'Perceptron Classifier Accuracy: {100 * accuracy:.2f}%')

            axes[i, j].set_xlabel(f'Accuracy: {100 * accuracy:.2f}%')

            classifier.plot(axes[i, j], clusters[2:], show=False)

    pp.show()
