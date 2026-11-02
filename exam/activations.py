import numpy as np

def sigma(x):
    return 1 / (1 + np.exp(-x))


def sigma_prime(x):
    """Derivative of sigmoid: sigma(x) * (1 - sigma(x))."""
    s = sigma(x)
    return s * (1 - s)


def relu(x):
    return np.maximum(0, x)


def relu_prime(x):
    """Derivative of ReLU: 1 where x > 0, otherwise 0."""
    return (x > 0).astype(float)
