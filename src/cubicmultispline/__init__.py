"""cubicmultispline

A tiny placeholder package for cubic multivariate spline utilities.
"""
from .Spline1D import Spline1D
from .Spline import Spline

__all__ = [
    "Spline1D",
    "Spline"
]

try:
    from .TorchSpline1D import TorchSpline1D
    from .TorchSpline import TorchSpline
    __all__ += ["TorchSpline1D", "TorchSpline"]
except ImportError:
    pass
