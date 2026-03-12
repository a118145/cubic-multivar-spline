# CubicMultiSpline documentation
Library for cubic, **multivariate spline interpolation** from samples with **arbitrary boundary conditions for each dimension**. Optional GPU backend available via PyTorch. The full documentation can be found at <a href="https://cubicmultispline.readthedocs.io" target="_blank">https://cubicmultispline.readthedocs.io</a>.

## Overview
This library implements the recursive algorithm by <a href="https://link.springer.com/article/10.1007/s10614-007-9092-4" target="_blank">Habermann and Kindermann</a> in the `Spline` class. The 1-dimensional base case, which is needed during recursion is implemented in the `Spline1D` class. In contrast to other multivariate spline implementations, this library allows for arbitrary boundary conditions for each dimension, that is
1. not-a-knot
2. first order (clamped)
3. second order (natural)
4. periodic 

Additionally, the library provides an efficient function `eval_spline` to evaluate the spline at arbitrary points inside the domain. The GPU backend is accessed via the `TorchSpline` and `TorchSpline1D` classes.

## Installation

The easiest way to install the library is to use `pip`:

```bash
pip install cubicmultispline
```

If you want to use the PyTorch backend, install with:

```bash
pip install cubicmultispline[torch]
```

Alternatively, you can install the library from source:

```bash
python setup.py install
```

## License

This library is licensed under the MIT License.
