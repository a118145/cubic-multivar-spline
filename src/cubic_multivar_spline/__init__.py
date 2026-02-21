"""cubic_multivar_spline

A tiny placeholder package for cubic multivariate spline utilities.
"""
from .Spline1D import Spline1D

__version__ = "0.1.0"

__all__ = ["version", "example_function", "Spline1D"]

version = __version__

def example_function(x):
    """Example placeholder function.

    Parameters
    - x: numeric

    Returns
    - x squared
    """
    return x * x
