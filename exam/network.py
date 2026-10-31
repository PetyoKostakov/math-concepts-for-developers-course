import numpy as np

from activations import sigma


class Network:
    """Simple Network([784, 16, 16, 10]).

    Layer l has a weight matrix of shape (neurons_l, neurons_l-1) and a bias
    column vector of shape (neurons_l, 1). Activations are column vectors.
    """
    def __init__(self, sizes=(784, 16, 16, 10), seed=None):
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.weights = self._init_weights(rng)
        self.biases = self._init_biases(rng)

    def feedforward(self, a):
        """Return the output activations for input column vector `a` (n*1)."""
        for w, b in zip(self.weights, self.biases):
            a = sigma(w @ a + b)
        return a

    def predict(self, a):
        """Return the digit whose output neuron has the highest activation."""
        return int(np.argmax(self.feedforward(a)))

    def _init_weights(self, rng):
        """Create one random weight matrix for every pair of neighbouring layers."""
        weights = []
        for neurons_in, neurons_out in zip(self.sizes[:-1], self.sizes[1:]): # self.sizes[:-1] → [784, 16, 16] ; self.sizes[1:] → [16, 16, 10]
            # Row i holds the weights of all connections into neuron i of the next layer
            layer_weights = rng.standard_normal((neurons_out, neurons_in))
            weights.append(layer_weights)
        return weights

    def _init_biases(self, rng):
        """Create one random bias column vector for every layer after the input layer."""
        biases = []
        for neurons in self.sizes[1:]:  # the input layer has no biases
            # One bias per neuron in the layer, stored as a column vector (neurons, 1)
            layer_biases = rng.standard_normal((neurons, 1))
            biases.append(layer_biases)
        return biases
