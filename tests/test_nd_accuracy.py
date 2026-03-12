import numpy as np
import pytest
from test_functions import test_function_1 as _tf1
from cubicmultispline.Spline import Spline


def test_4d_value_accuracy():
    shape = (5, 5, 5, 5)
    xs = [np.linspace(0, 1, n) for n in shape]
    y, _, _, _ = _tf1(*xs)
    interval = tuple((0, 1, n) for n in shape)
    bc = (("not-a-knot", "not-a-knot"),) * 4
    bc_value = ((0.0, 0.0),) * 4
    spline = Spline(interval, y, bc, bc_value)

    xs_eval = [np.linspace(0, 1, 4) for _ in range(4)]
    y_ref, _, _, _ = _tf1(*xs_eval)
    grids = np.meshgrid(*xs_eval, indexing='ij')
    coords = np.column_stack([g.ravel() for g in grids])
    f, _, _ = spline.eval_spline(coords)
    assert np.max(np.abs(f - y_ref)) < 1e-5


def test_4d_interpolation_at_nodes():
    shape = (5, 5, 5, 5)
    xs = [np.linspace(0, 1, n) for n in shape]
    y, _, _, _ = _tf1(*xs)
    interval = tuple((0, 1, n) for n in shape)
    bc = (("not-a-knot", "not-a-knot"),) * 4
    bc_value = ((0.0, 0.0),) * 4
    spline = Spline(interval, y, bc, bc_value)

    grids = np.meshgrid(*xs, indexing='ij')
    coords = np.column_stack([g.ravel() for g in grids])
    f, _, _ = spline.eval_spline(coords)
    assert np.max(np.abs(f - y)) < 1e-8


def test_5d_construction_smoke():
    shape = (4, 4, 4, 4, 4)
    xs = [np.linspace(0, 1, n) for n in shape]
    y, _, _, _ = _tf1(*xs)
    interval = tuple((0, 1, n) for n in shape)
    bc = (("not-a-knot", "not-a-knot"),) * 5
    bc_value = ((0.0, 0.0),) * 5
    spline = Spline(interval, y, bc, bc_value)
    assert spline._coeff is not None
