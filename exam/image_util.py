"""Image utilities for turning image files into neural network input."""
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve(path) -> Path:
    """Resolve a path; relative paths are looked up from the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_image_matrix(path, size=(28, 28)) -> np.ndarray:
    """Read an image and return a 2D matrix (height x width) of floats in [0, 1].

    0.0 is black, 1.0 is white. The image is converted to grayscale and
    resized to `size` (width, height) if it differs; pass size=None to keep it.
    """
    img = Image.open(_resolve(path)).convert("L")
    if size is not None and img.size != tuple(size):
        img = img.resize(tuple(size))
    return np.asarray(img, dtype=np.float64) / 255.0


def load_image_vector(path, size=(28, 28)) -> np.ndarray:
    """Read an image and return a column vector (n*1) usable as the first layer."""
    return load_image_matrix(path, size).reshape(-1, 1)
