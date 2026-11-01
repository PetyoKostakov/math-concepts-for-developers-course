import numpy as np

from activations import sigma, sigma_prime


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

    def backprop(self, x, y):
        """Return (grad_weights, grad_biases) of the cost 1/2 * ||a_out - y||^2 for one example.

        Shapes match self.weights / self.biases.
        """
        # Forward pass: remember weighted inputs (z) and activations (a) of every layer
        activation = x
        activations = [x]
        weighted_inputs = []
        for w, b in zip(self.weights, self.biases):
            z = w @ activation + b
            activation = sigma(z)
            weighted_inputs.append(z)
            activations.append(activation)

        grad_weights = [None] * len(self.weights)
        grad_biases = [None] * len(self.biases)

        # Output layer: error = how wrong the output is, scaled by how sensitive the neuron is
        error = (activations[-1] - y) * sigma_prime(weighted_inputs[-1])
        grad_biases[-1] = error
        grad_weights[-1] = error @ activations[-2].T

        # Hidden layers, going backwards: hand the error of the next layer back through its weights
        for layer in range(len(self.weights) - 2, -1, -1):
            error = (self.weights[layer + 1].T @ error) * sigma_prime(weighted_inputs[layer])
            grad_biases[layer] = error
            grad_weights[layer] = error @ activations[layer].T

        return grad_weights, grad_biases

    def gradient_descent_step(self, grad_weights, grad_biases, eta):
        """Move every weight and bias a small step (eta) against its gradient.

        grad_weights / grad_biases have the same shapes as self.weights / self.biases
        and say how much the cost grows when that parameter grows.
        """
        for layer in range(len(self.weights)):
            self.weights[layer] -= eta * grad_weights[layer]
            self.biases[layer] -= eta * grad_biases[layer]

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
