# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Sacrifice grammar and spelling for brevity, clarity and conciseness. Prefer concise, clear bullet points over perfect grammar and full sentences.

## Code modifications

- Review files which are part of your modifications before making a plan -> I might have added small things in the meantime
- For new implementations: create new git branch "feature/<YYMMDD_feature-name>" based off of "develop"
- feature branches are merged into develop using the "--no-ff" flag

## Project Overview

`cubicmultispline` is a Python library for cubic multivariate spline interpolation from samples with arbitrary boundary conditions per dimension. It implements the recursive algorithm by Habermann and Kindermann (DOI: 10.1007/s10614-007-9092-4). Published on PyPI as `cubicmultispline`.

## Commands

### Install
```bash
pip install -e .                # editable install
pip install -e ".[dev]"         # with dev dependencies (twine, setuptools, matplotlib)
```

### Run tests
Tests use pytest. Run from the project root:
```bash
venv/bin/pytest tests/ -v                    # all accuracy + BC tests
venv/bin/pytest tests/ -m benchmark -v -s    # performance benchmarks only
venv/bin/pytest tests/ -v --tb=short         # full suite with short tracebacks
```

### Build docs
```bash
cd docs && make html            # Sphinx docs with Furo theme, output in docs/_build/html/
```

### Build & publish
```bash
python -m build                 # creates sdist + wheel in dist/
twine upload dist/*             # or push a v*.*.* tag to trigger GitHub Actions publish
```

## Architecture

Two core classes in `src/cubicmultispline/`:

- **`Spline1D`** (`Spline1D.py`): 1D cubic spline using B-spline basis functions. Solves a linear system (`np.linalg.solve`) for coefficients. Supports 4 boundary condition types: `not-a-knot`, `periodic`, `first_derivative`, `second_derivative`. `eval_spline()` returns `(value, d1, d2, d3)` — function value plus first three derivatives.

- **`Spline`** (`Spline.py`): N-dimensional cubic spline via tensor product of 1D B-splines. Uses `recursive_spline()` static method that recursively reduces dimensionality — at each level it fits 1D splines along one axis, then fits another 1D spline across the resulting coefficients. `eval_spline()` returns `(f, grad, hess)` — function values, gradient vectors, and Hessian matrices. Iterates over all 4^d basis function combinations per evaluation point.

### Key conventions
- `Spline` takes `interval` as tuple of `(start, end, n_points)` per dimension; `yv` is a flat array in row-major (C) order matching `meshgrid(..., indexing='ij')`
- `Spline1D` takes `interval` as `(start, end)` and `yv` as a 1D array of function values at equidistant points
- Coefficients have shape `(n1+2, n2+2, ..., nd+2)` — two extra per dimension for the B-spline basis
- Boundary condition values for derivative BCs in higher dimensions are internally scaled by `1/(6^(level-1))`

## Dependencies
- numpy >= 2.4, scipy >= 1.12, matplotlib >= 3.10.0
- Python >= 3.8 (developed on 3.12)
