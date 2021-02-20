from typing import List

import numpy as np
import pandas as pd

from numpy import ndarray as Array
from matplotlib import pyplot as pp

from .error_functions import MSE
from .activation_functions import softmax, sigmoid, sigmoid_derivative


class Layer:
    """
    Layer abstraction used internally by MultilayerPerceptronClassifier for mostly
    code-structuring purposes.
    """
    def __init__(self, size: int, weights: Array = None):
        self.size = size
        self.weights = weights
        self.inputs = None
        self.outputs = None

    def __call__(self, inputs: Array) -> Array:
        """
        Perform the matrix transformation of the inputs with respect to the
        weights. We memoize the inputs and outputs, as they are needed in
        training the network.
        """
        self.inputs = inputs

        # NOTE: the first layer in a network will have no weights
        if self.weights is not None:
            self.outputs = sigmoid(np.dot(inputs, self.weights))
        else:
            # this is an input layer
            self.outputs = inputs

        return self.outputs


class MultilayerPerceptronClassifier:
    def __init__(self, layers: List[Layer], softmax=True):
        self.layers = layers
        self.meta = {}
        self.use_softmax = softmax

    def __call__(self, inputs: Array) -> Array:
        """
        A trained classifier is treated as a function-like object. It is
        intended to be used like this:

        classify = MultilayerPerceptronClassifier(layers).train(**kwargs)
        output = classify(input)

        The network takes the input, feeds it forward, and returns the
        corresponding raw output or softmax logits. 
        
        """
        for layer in self.layers:
            outputs = layer(inputs)
            inputs = outputs

        # return the raw output or softmax logits
        outputs = self.layers[-1].outputs
        return softmax(outputs) if self.use_softmax else outputs

    def train(
        self,
        labels: Array,
        inputs: Array,
        epochs: int = 1000,
        rate: float = 0.1
    ) -> 'MultilayerPerceptronClassifier':
        """
        - Initialize each weight matrix to small random values.
        - For each epoch:
            - Call the network, AKA feed-forward
            - Compare output with expected (AKA labels) to measure the error
            - Update weights via back-propagation with gradient descent,
                with respect to the negative gradient of the error.
        """
        self.meta.update({
            'epochs': epochs,
            'rate': rate,
            'labels': labels,
            'series': {
                'error': [],
                'accuracy': [],
            },
        })

        # initialize random weights between each successive layer pair
        # unless the weights matrix is already initialized (non-None)
        for layer_from, layer_to in zip(self.layers, self.layers[1:]):
            if layer_to.weights is None:
                n, m = layer_from.size, layer_to.size
                layer_to.weights = np.random.normal(scale=0.5, size=(n, m))

        for t in range(epochs):
            # feed-forward, setting layer.inputs and layer.outputs
            # on each layer. on the terminal layer, the Layer component simply
            # sets outputs to the inputs directly.
            self(inputs)

            # update weights & biases (back-prop)
            self.update(labels, rate)

        return self

    def update(self, labels: Array, rate: float):
        """
        Update weights and biases with back propagation.

        ## Original Paper:
        Learning Representations by Back-propagating Errors;
        Rumelhart, et al.
        https://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf

        ## Formal Derivation
        Great video series for the mathematical derivation back propagation
        with gradient descent:

        https://www.youtube.com/watch?v=4bvJRE5K5p4&list=PLPOTBrypY74wOpTIWQhqNdfV5gIt1h1fa
        """
        # compute delta for the output layer. this base case differs
        # from how delta is computed for the other layers.
        output_layer = self.layers[-1]
        error = output_layer.outputs - labels

        # self.meta holds "metadata" generated during training, namely 
        # error and accuracy time series.
        self.meta['series']['error'].append(MSE(error))
        self.meta['series']['accuracy'].append(
            self.accuracy(output_layer.outputs, labels)
        )

        # now we compute the gradient (a matrix) for each layer's weights with
        # respect to the network's overall error, starting from the output
        # layer, going backwards.

        # NOTE: each gradient, called "delta" here, is a product of several
        # partial derivatives, identified using the chain rule for the sum of
        # several partial derivatives. Basically, there's a formula...
        delta = error * sigmoid_derivative(output_layer.outputs)
        deltas = [delta]

        # Compute the gradient (denoted as "delta") between each layer.
        # We skip the the first layer, as weight matrices are stored in the
        # second layer in each bigram.
        for layer, to_layer in reversed(
            list(zip(self.layers[1:], self.layers[2:]))
        ):
            # compute the partial derivatives the go into the gradient
            dx_dw = np.dot(delta, to_layer.weights.T)
            di_do = sigmoid_derivative(layer.outputs)
            delta = dx_dw * di_do
            deltas.append(delta)

        # now that we've accumulated the "deltas", go ahead and update
        # each weight matrix.
        for layer, delta in zip(reversed(self.layers[1:]), deltas):
            layer.weights -= rate * np.dot(layer.inputs.T, delta)

    def plot(self, axes, show_labels=True, show=True):
        """
        Plot the error and accuracy, as collected during training.
        """
        stats = pd.DataFrame(self.meta['series'])

        if show_labels:
            for ax in axes:
                ax.set_xlabel('Epoch')

        stats.error.plot(
            ax=axes[0],
            linewidth=1,
            title='Mean Squared Error' if show_labels else None,
        )

        stats.accuracy.plot(
            ax=axes[1],
            linewidth=1,
            color='green',
            title='Accuracy' if show_labels else None,
        )

        info = (
            f'Training Set Size: {len(self.meta["labels"])}\n'
            f'Learning Rate: {self.meta["rate"]:.3f}\n'
            f'Epochs: {self.meta["epochs"]}\n'
        )

        axes[0].text(
            0.15, 0.90, info,
            transform=axes[0].transAxes,
            verticalalignment='top'
        )

        if show:
            pp.show()

    @staticmethod
    def accuracy(outputs, labels) -> float:
        """
        an output matches its label if, when rounded, each component has
        exactly the same value.
        """
        n_total = len(outputs)
        n_matches = np.sum(np.all(outputs.round() == labels, axis=1))
        return n_matches / n_total