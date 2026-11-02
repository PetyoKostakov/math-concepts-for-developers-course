import numpy as np

from activations import relu, relu_prime, sigma, sigma_prime


class Network:
    """Simple Network([784, 16, 16, 10]).

    Layer l has a weight matrix of shape (neurons_l, neurons_l-1) and a bias
    column vector of shape (neurons_l, 1). Activations are column vectors.
    """
    def __init__(self, sizes=(784, 16, 16, 10), seed=None, hidden_activation="sigmoid"):
        # The output layer always uses sigmoid; hidden layers use sigmoid or relu.
        self.hidden_activation, self.hidden_activation_prime = {
            "sigmoid": (sigma, sigma_prime),
            "relu": (relu, relu_prime),
        }[hidden_activation]
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.weights = self._init_weights(rng)
        self.biases = self._init_biases(rng)

    def feedforward(self, a):
        """Return the output activations for input column vector `a` (n*1)."""
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            a = self.hidden_activation(w @ a + b)
        return sigma(self.weights[-1] @ a + self.biases[-1])

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
            is_output = len(weighted_inputs) == len(self.weights) - 1
            activation = sigma(z) if is_output else self.hidden_activation(z)
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
            error = (self.weights[layer + 1].T @ error) * self.hidden_activation_prime(weighted_inputs[layer])
            grad_biases[layer] = error
            grad_weights[layer] = error @ activations[layer].T

        return grad_weights, grad_biases

    def accuracy(self, data):
        """Fraction of (input, target) pairs where the predicted digit is the target digit."""
        correct = sum(self.predict(x) == int(np.argmax(y)) for x, y in data)
        return correct / len(data)

    def train(self, training_data, epochs, batch_size, eta, test_data=None, seed=None, verbose=True):
        """Train with mini-batch stochastic gradient descent.

        One epoch = one pass over all training data, in random order, split into
        batches. For each batch the gradients of its examples are averaged and
        one gradient descent step is taken. Returns the accuracy after every epoch.
        """
        rng = np.random.default_rng(seed)
        history = {"train_accuracy": [], "test_accuracy": []}

        for epoch in range(epochs):
            order = rng.permutation(len(training_data))
            shuffled = [training_data[i] for i in order]

            for start in range(0, len(shuffled), batch_size):
                batch = shuffled[start:start + batch_size]
                self._train_on_batch(batch, eta)

            history["train_accuracy"].append(self.accuracy(training_data))
            message = f"epoch {epoch + 1:3d}: train accuracy = {history['train_accuracy'][-1]:.2%}"
            if test_data is not None:
                history["test_accuracy"].append(self.accuracy(test_data))
                message += f", test accuracy = {history['test_accuracy'][-1]:.2%}"
            if verbose:
                print(message)

        return history

    def _train_on_batch(self, batch, eta):
        """One gradient descent step using the average gradient of the batch."""
        sum_grad_w = [np.zeros_like(w) for w in self.weights]
        sum_grad_b = [np.zeros_like(b) for b in self.biases]

        for x, y in batch:
            grad_w, grad_b = self.backprop(x, y)
            for layer in range(len(self.weights)):
                sum_grad_w[layer] += grad_w[layer]
                sum_grad_b[layer] += grad_b[layer]

        average_w = [g / len(batch) for g in sum_grad_w]
        average_b = [g / len(batch) for g in sum_grad_b]
        self.gradient_descent_step(average_w, average_b, eta)

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
            if self.hidden_activation is relu:
                # ReLU does not squash large values, so start with small weights (He initialisation)
                layer_weights *= np.sqrt(2 / neurons_in)
            weights.append(layer_weights)
        return weights

    def _init_biases(self, rng):
        """Create one random bias column vector for every layer after the input layer."""
        biases = []
        for neurons in self.sizes[1:]:  # the input layer has no biases
            # One bias per neuron in the layer, stored as a column vector (neurons, 1)
            layer_biases = rng.standard_normal((neurons, 1))
            if self.hidden_activation is relu:
                layer_biases = np.zeros((neurons, 1))
            biases.append(layer_biases)
        return biases
