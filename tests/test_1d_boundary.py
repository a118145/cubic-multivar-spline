import numpy as np
import pytest
from test_functions import test_function_1 as _tf1
from cubicmultispline.Spline1D import Spline1D
from cubicmultispline.Spline import Spline


@pytest.fixture
def base_data():
    shape = (11,)
    x = np.linspace(-1, 1, shape[0])
    y, _, _, _ = _tf1(x)
    interval = ((-1, 1, shape[0]),)
    x_eval = np.linspace(-1, 1, 100)
    return dict(shape=shape, x=x, y=y, interval=interval, x_eval=x_eval)


def _build_splines(base_data, bc, bc_value, make_periodic=False):
    s = base_data
    y = s["y"].copy()
    if make_periodic:
        y[-1] = y[0]
    spline_native = Spline1D(s["interval"][0], y, bc[0], bc_value[0])
    spline_multi = Spline(s["interval"], y, bc, bc_value)
    y_n, dy_n, ddy_n, _ = spline_native.eval_spline(s["x_eval"])
    y_m, dy_m, ddy_m = spline_multi.eval_spline(s["x_eval"])
    return y_n, dy_n, ddy_n, y_m, dy_m, ddy_m


def test_1d_not_a_knot_consistency(base_data):
    bc = (("not-a-knot", "not-a-knot"),)
    bc_value = ((0.0, 0.0),)
    y_n, dy_n, ddy_n, y_m, dy_m, ddy_m = _build_splines(base_data, bc, bc_value)
    assert np.max(np.abs(y_n - y_m.ravel())) < 1e-10
    assert np.max(np.abs(dy_n - dy_m.ravel())) < 1e-10
    assert np.max(np.abs(ddy_n - ddy_m.ravel())) < 1e-10


def test_1d_first_derivative_bc(base_data):
    bc = (("first_derivative", "second_derivative"),)
    bc_value = ((1.0, -2.0),)
    y_n, dy_n, ddy_n, y_m, dy_m, ddy_m = _build_splines(base_data, bc, bc_value)
    assert np.abs(dy_n[0] - bc_value[0][0]) < 1e-13
    assert np.abs(dy_m.ravel()[0] - bc_value[0][0]) < 1e-13


def test_1d_second_derivative_bc(base_data):
    bc = (("first_derivative", "second_derivative"),)
    bc_value = ((1.0, -2.0),)
    y_n, dy_n, ddy_n, y_m, dy_m, ddy_m = _build_splines(base_data, bc, bc_value)
    assert np.abs(ddy_n[-1] - bc_value[0][1]) < 1e-13
    assert np.abs(ddy_m.ravel()[-1] - bc_value[0][1]) < 1e-13


def test_1d_mixed_bc_consistency(base_data):
    bc = (("first_derivative", "second_derivative"),)
    bc_value = ((1.0, -2.0),)
    y_n, dy_n, ddy_n, y_m, dy_m, ddy_m = _build_splines(base_data, bc, bc_value)
    assert np.max(np.abs(y_n - y_m.ravel())) < 1e-10
    assert np.max(np.abs(dy_n - dy_m.ravel())) < 1e-10
    assert np.max(np.abs(ddy_n - ddy_m.ravel())) < 1e-10


def test_1d_periodic_derivative_continuity(base_data):
    bc = (("periodic", "periodic"),)
    bc_value = ((0.0, 0.0),)
    y_n, dy_n, ddy_n, y_m, dy_m, ddy_m = _build_splines(
        base_data, bc, bc_value, make_periodic=True
    )
    assert np.abs(dy_n[0] - dy_n[-1]) < 1e-13
    assert np.abs(ddy_n[0] - ddy_n[-1]) < 1e-13


def test_1d_periodic_consistency(base_data):
    bc = (("periodic", "periodic"),)
    bc_value = ((0.0, 0.0),)
    y_n, dy_n, ddy_n, y_m, dy_m, ddy_m = _build_splines(
        base_data, bc, bc_value, make_periodic=True
    )
    assert np.max(np.abs(y_n - y_m.ravel())) < 1e-10
    assert np.max(np.abs(dy_n - dy_m.ravel())) < 1e-10
    assert np.max(np.abs(ddy_n - ddy_m.ravel())) < 1e-10
