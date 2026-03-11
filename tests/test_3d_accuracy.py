import numpy as np
import pytest
from test_functions import test_function_1 as _tf1
from cubicmultispline.Spline import Spline


def test_3d_value_accuracy(setup_3d):
    s = setup_3d
    spline = Spline(s["interval"], s["y"], s["bc"], s["bc_value"])
    x1e = np.linspace(0, 1, 8)
    x2e = np.linspace(0, 1, 8)
    x3e = np.linspace(0, 1, 8)
    y_ref, _, _, _ = _tf1(x1e, x2e, x3e)
    grids = np.meshgrid(x1e, x2e, x3e, indexing='ij')
    coords = np.column_stack([g.ravel() for g in grids])
    f, grad, hess = spline.eval_spline(coords)
    assert np.max(np.abs(f - y_ref)) < 1e-8


def test_3d_gradient_accuracy(setup_3d):
    s = setup_3d
    spline = Spline(s["interval"], s["y"], s["bc"], s["bc_value"])
    x1e = np.linspace(0, 1, 8)
    x2e = np.linspace(0, 1, 8)
    x3e = np.linspace(0, 1, 8)
    _, grad_ref, _, _ = _tf1(x1e, x2e, x3e)
    grids = np.meshgrid(x1e, x2e, x3e, indexing='ij')
    coords = np.column_stack([g.ravel() for g in grids])
    f, grad, hess = spline.eval_spline(coords)
    assert np.max(np.abs(grad - grad_ref)) < 1e-7


def test_3d_hessian_accuracy(setup_3d):
    s = setup_3d
    spline = Spline(s["interval"], s["y"], s["bc"], s["bc_value"])
    x1e = np.linspace(0, 1, 8)
    x2e = np.linspace(0, 1, 8)
    x3e = np.linspace(0, 1, 8)
    _, _, hess_ref, _ = _tf1(x1e, x2e, x3e)
    grids = np.meshgrid(x1e, x2e, x3e, indexing='ij')
    coords = np.column_stack([g.ravel() for g in grids])
    f, grad, hess = spline.eval_spline(coords)
    assert np.max(np.abs(hess - hess_ref)) < 1e-6


def test_3d_interpolation_at_nodes(setup_3d):
    s = setup_3d
    spline = Spline(s["interval"], s["y"], s["bc"], s["bc_value"])
    grids = np.meshgrid(s["x1"], s["x2"], s["x3"], indexing='ij')
    coords = np.column_stack([g.ravel() for g in grids])
    f, _, _ = spline.eval_spline(coords)
    assert np.max(np.abs(f - s["y"])) < 1e-10


def test_3d_mixed_bc():
    """Different BC per dimension.

    BC values are scalar per dim/side, but the true partial derivatives of
    _tf1 vary along the boundary (e.g. df/dx2 at x2=0 = dp(0)*p(x1)*p(x3)).
    So non-not-a-knot BCs with constant values can only approximate _tf1;
    tolerance is relaxed accordingly.
    """
    shape = (7, 9, 8)
    x1 = np.linspace(0, 1, shape[0])
    x2 = np.linspace(0, 1, shape[1])
    x3 = np.linspace(0, 1, shape[2])
    y, _, _, _ = _tf1(x1, x2, x3)
    interval = ((0, 1, shape[0]), (0, 1, shape[1]), (0, 1, shape[2]))
    bc = (
        ("not-a-knot", "not-a-knot"),
        ("first_derivative", "first_derivative"),
        ("second_derivative", "second_derivative"),
    )
    bc_value = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    spline = Spline(interval, y, bc, bc_value)

    x1e = np.linspace(0, 1, 6)
    x2e = np.linspace(0, 1, 6)
    x3e = np.linspace(0, 1, 6)
    y_ref, _, _, _ = _tf1(x1e, x2e, x3e)
    grids = np.meshgrid(x1e, x2e, x3e, indexing='ij')
    coords = np.column_stack([g.ravel() for g in grids])
    f, _, _ = spline.eval_spline(coords)
    assert np.max(np.abs(f - y_ref)) < 1e-3
