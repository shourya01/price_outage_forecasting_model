"""All-zeros forecast."""

import numpy as np

NAME = "All-Zeros"
NEEDS_GPU = False


def run(exp, **_):
    """Zeros with the shape of the test targets."""
    return np.zeros_like(exp.truth)
