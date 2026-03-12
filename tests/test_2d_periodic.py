import numpy as np
import pytest
from test_functions import test_function_2 as _tf2
from cubicmultispline.Spline import Spline


def _build_periodic_spline(n1, n2):
    """Build 2D spline with not-a-knot in dim 1, periodic in dim 2."""
    y_min, y_max = 0.0, 2.0 * np.pi
    x = np.linspace(0, 1, n1)
    y = np.linspace(y_min, y_max, n2, endpoint=True)
    vals, _, _, _ = _tf2(x, y, y_bounds=(y_min, y_max))
    interval = ((0, 1, n1), (y_min, y_max, n2))
    bc = (("not-a-knot", "not-a-knot"), ("periodic", "periodic"))
    bc_value = ((0.0, 0.0), (0.0, 0.0))
    spline = Spline(interval, vals, bc, bc_value)
    return spline, x, y, y_min, y_max


def _eval_at_grid(spline, x_eval, y_eval):
    grids = np.meshgrid(x_eval, y_eval, indexing='ij')
    coords = np.column_stack([g.ravel() for g in grids])
    return spline.eval_spline(coords)


def test_2d_periodic_value_accuracy():
    spline, x, y, y_min, y_max = _build_periodic_spline(15, 25)
    x_eval = np.linspace(0, 1, 20)
    y_eval = np.linspace(y_min, y_max, 20)
    f, grad, hess = _eval_at_grid(spline, x_eval, y_eval)
    f_ref, _, _, _ = _tf2(x_eval, y_eval, y_bounds=(y_min, y_max))
    assert np.max(np.abs(f - f_ref)) < 1e-2


def test_2d_periodic_gradient_accuracy():
    spline, x, y, y_min, y_max = _build_periodic_spline(15, 25)
    x_eval = np.linspace(0, 1, 20)
    y_eval = np.linspace(y_min, y_max, 20)
    f, grad, hess = _eval_at_grid(spline, x_eval, y_eval)
    _, grad_ref, _, _ = _tf2(x_eval, y_eval, y_bounds=(y_min, y_max))
    assert np.max(np.abs(grad - grad_ref)) < 0.1


def test_2d_periodic_convergence():
    """Error decreases ~O(h^4) when doubling grid points."""
    y_min, y_max = 0.0, 2.0 * np.pi
    x_eval = np.linspace(0, 1, 30)
    y_eval = np.linspace(y_min, y_max, 30)
    f_ref, _, _, _ = _tf2(x_eval, y_eval, y_bounds=(y_min, y_max))

    errors = []
    for n2 in [10, 20, 40]:
        spline, _, _, _, _ = _build_periodic_spline(15, n2)
        f, _, _ = _eval_at_grid(spline, x_eval, y_eval)
        errors.append(np.max(np.abs(f - f_ref)))

    # when doubling points, error should decrease by factor ~16 (h^4)
    # be lenient: require at least factor 4
    ratio1 = errors[0] / errors[1]
    ratio2 = errors[1] / errors[2]
    assert ratio1 > 4, f"Convergence ratio {ratio1:.1f} too low (expected >4)"
    assert ratio2 > 4, f"Convergence ratio {ratio2:.1f} too low (expected >4)"


def test_2d_periodic_continuity():
    """dy, ddy match at periodic boundary."""
    spline, x, y, y_min, y_max = _build_periodic_spline(15, 25)
    x_eval = np.linspace(0.1, 0.9, 10)
    y_start = np.array([y_min])
    y_end = np.array([y_max])

    f_s, grad_s, hess_s = _eval_at_grid(spline, x_eval, y_start)
    f_e, grad_e, hess_e = _eval_at_grid(spline, x_eval, y_end)

    # dy/dy at boundary should match
    assert np.max(np.abs(grad_s[:, 1] - grad_e[:, 1])) < 1e-8
    # d2y/dy2 at boundary should match
    assert np.max(np.abs(hess_s[:, 1, 1] - hess_e[:, 1, 1])) < 1e-6


def test_2d_mixed_bc_periodic_first_deriv():
    """first_derivative in dim 1 + periodic in dim 2."""
    y_min, y_max = 0.0, 2.0 * np.pi
    n1, n2 = 15, 25
    x = np.linspace(0, 1, n1)
    y = np.linspace(y_min, y_max, n2, endpoint=True)
    vals, _, _, _ = _tf2(x, y, y_bounds=(y_min, y_max))
    interval = ((0, 1, n1), (y_min, y_max, n2))
    bc = (("first_derivative", "first_derivative"), ("periodic", "periodic"))
    bc_value = ((0.0, 0.0), (0.0, 0.0))
    spline = Spline(interval, vals, bc, bc_value)

    x_eval = np.linspace(0, 1, 20)
    y_eval = np.linspace(y_min, y_max, 20)
    f, grad, hess = _eval_at_grid(spline, x_eval, y_eval)
    f_ref, _, _, _ = _tf2(x_eval, y_eval, y_bounds=(y_min, y_max))
    # should still be reasonable
    assert np.max(np.abs(f - f_ref)) < 0.5
