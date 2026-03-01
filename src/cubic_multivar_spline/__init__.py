"""cubic_multivar_spline

A tiny placeholder package for cubic multivariate spline utilities.
"""
from .Spline1D import Spline1D
from .Spline import Spline
from .spline_eval import eval_spline

__version__ = "0.1.0"

__all__ = [
    "version", 
    "Spline1D", 
    "Spline",
    "eval_spline"
]

version = __version__
