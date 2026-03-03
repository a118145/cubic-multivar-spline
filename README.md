# cubicmultispline
Library for cubic, multivariate spline interpolation from samples with arbitrary boundary conditions for each dimension.

## Overview
This library implements the recursive algorithm by [Habermann and Kindermann](https://link.springer.com/article/10.1007/s10614-007-9092-4) in the `Spline` class. The 1-dimensional base case, which is needed during recursion is implemented in the `Spline1D` class. In contrast to other multivariate spline implementations, this library allows for arbitrary boundary conditions for each dimension, that is
1. not-a-knot
2. first order (clamped)
3. second order (natural)
4. periodic 

Additionally, the library provides an efficient function `eval_spline` to evaluate the spline at arbitrary points in the domain.

## Installation

The easiest way to install the library is to use `pip`:

```bash
pip install cubicmultispline
```

Alternatively, you can install the library from source:

```bash
python setup.py install
```

## Usage

