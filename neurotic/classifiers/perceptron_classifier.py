"""
Perception (Binary Classifier).

NOTE: To run, you must `pip install numpy matplotlib`

The perceptron is a single neuron. It takes a bunch of input vectors, assigns
initially random weights, and computes a linear combination thereof (dot
product). We look at the output, threshold it to a full 0 or 1, and compare it
to what we expect (either a 0 or 1). If it differs from the expected value for
some of the input vectors, then we move the weights a bit in the direction of
those vectors. We do this until the weights represent the coefficients in the
equation for the line that represents the "decision boundary" between the two
training clusters. This process repeats for a given number of "epochs", or time
steps.
"""

import json

from typing import Text, List, Optional

import numpy as np

from matplotlib import pyplot as pp


class PerceptronClassifier:
    """
    Binary classifier for linearly separable data, implementing the
    Perceptron machine-learning algorithm.
    """

    def __init__(self, weights: Optional[np.array] = None):
        self.weights = weights  # trained weights

    def train(self, clusters: List[np.array], epochs=2048, rate=0.1):
        """
        # Notes:
            - The positional index of each cluster in the `clusters` arg is
              used as its label. EG: With clusters == [c0, c1], c0's cluster
              label is 0. Classifying an input returns one of these two
              label values.

        """
        # Take each 2D input vector and make it 3D through the addition
        # of a "bias" component -- in this case, the bias is 1, transforming
        # the cluster vectors from
        #
        # clusters == [[x0, y0], [x1, y1], ...]
        #
        # to
        #
        # clusters == [[x0, y0, 1], [x1, y1, 1], ...].
        #
        # Work on a copy of clusters so as not to mutate the list passed in,
        # avoiding any unintended side-effects in the caller.
        clusters = clusters.copy()
        for i, cluster in enumerate(clusters):
            n = cluster.shape[0]
            biases_column = np.ones((n, 1), dtype=float)
            clusters[i] = np.concatenate([cluster, biases_column], axis=1)

        # Initialize the perceptron's weights to random floats. the number of
        # weights is equal to the dimension of the inputs, plus 1 for the added
        # "bias" component. Since our input vectors are 2D, we have 2 + 1
        # weights.
        self.weights = np.random.random(3)

        for _ in range(epochs):
            for expected_activation, cluster in enumerate(clusters):
                # NOTE: Keeping in mind that this is a *binary* classifier, we
                # only have two possible output values: 0 and 1. Because
                # of this, we can use each cluster's positional index as its
                # corresponding expected activation state (also a 0 or 1).
                #
                # Each output row is just the dot product of an input vector
                # with the weight vector, which has the "bias" value built in.
                # (i.e.  w1*x1 + w2*x2 + ... + wN*b)
                outputs = np.dot(cluster, self.weights)

                # Now we threshold each raw activation to one or the two
                # cluster IDs, which mathematically are 0 and 1.
                activations = outputs.copy()

                is_active = activations > 0.0       # array filters
                is_inactive = activations < 0.0

                activations[is_active] = 1.0
                activations[is_inactive] = 0.0

                # calculate the error. this is a simply a vector that shows us
                # which "direction" we want to shift each of the current
                # weights, such that the decision boundary is shifted to a
                # position where the maximum possible number input vectors in
                # this cluster are on one side of the bounardy.
                error = expected_activation - activations

                # move in the direction of each input vector who is on the
                # wrong side of the line. We're moving each time by `rate` (EG:
                # 0.1). A rate that's too high can overshoot. Too low, and we
                # won't converge.
                self.weights += rate * np.dot(error, cluster)

    def classify(self, vec: np.array) -> int:
        """
        Classify the input vector as belonging to class 0 or 1.
        """
        # Note that the 3rd component of the weight vector is the "bias"
        # constant. This is why it is not included in the dot product.
        # Geometrically speaking, it's the y-intercept of the decision
        # boundary.
        return (
            1 if (np.dot(self.weights[:2], vec) + self.weights[2]) > 0
            else 0
        )

    def test(self, clusters: List[np.array]) -> float:
        """
        Return the accuracy of a trained classifier.
        """
        n_matches = 0
        n_total_inputs = 0

        for cluster_id, cluster in enumerate(clusters):
            n_total_inputs += len(cluster)
            for vec in cluster:
                classified_as_cluster_id = self.classify(vec)
                n_matches += int(classified_as_cluster_id == cluster_id)

        return n_matches / n_total_inputs

    def plot(self, axes, clusters: List[np.array], show=True):
        """
        Draw each cluster and the trained decision boundary.
        """

        def plot_cluster(cluster, color, marker):
            """
            Draw a scatter plot for a group of vectors AKA a cluster.
            """
            x = cluster[:, 0]
            y = cluster[:, 1]
            axes.scatter(x, y, s=20, c=color, alpha=0.6, marker=marker)

        # draw scatter plot for both clusters
        plot_cluster(clusters[0], 'red',  'o')
        plot_cluster(clusters[1], 'blue', '^')

        # x range for plotting
        x_min = min(np.min(clusters[0][:, 0]), np.min(clusters[1][:, 0]))
        x_max = max(np.max(clusters[0][:, 0]), np.max(clusters[1][:, 0]))
        x = np.arange(x_min - 2, x_max + 2)

        # slope of decision boundary
        m = -self.weights[0] / self.weights[1]

        # y-intercept of decision boundary
        b = -self.weights[2] / self.weights[1]

        # plot decision boundary
        y = (m * x) + b
        axes.plot(x, y, c='black', linewidth=3)

        if show:
            pp.show()

    def save(self, filepath: Text):
        """
        Save classifier to a JSON file.
        """
        with open(filepath, 'w') as json_file:
            json.dump({
                'type': type(self).__name__,
                'weights': list(self.weights or []),
            }, json_file)

    @classmethod
    def load(cls, filepath: Text) -> 'PerceptronClassifier':
        """
        Instantiate a classifier using the weights defined in a JSON file
        generated by `self.save`.
        """
        with open(filepath, 'r') as json_file:
            data = json.load(json_file)
            return cls(weights=np.array(data['weights'], dtype=float))