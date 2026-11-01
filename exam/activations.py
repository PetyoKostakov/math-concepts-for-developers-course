import numpy as np

def sigma(x):
    return 1 / (1 + np.exp(-x))


def sigma_prime(x):
    """Derivative of sigmoid: sigma(x) * (1 - sigma(x))."""
    s = sigma(x)
    return s * (1 - s)
