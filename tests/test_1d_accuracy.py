import numpy as np
import pytest
from test_functions import test_function_1 as _tf1
from cubicmultispline.Spline1D import Spline1D
from cubicmultispline.Spline import Spline


@pytest.fixture
def splines_and_ref(setup_1d):
    s = setup_1d
    spline_native = Spline1D(s["interval"][0], s["y"], s["bc"][0], s["bc_value"][0])
    spline_multi = Spline(s["interval"], s["y"], s["bc"], s["bc_value"])
    x_eval = np.linspace(-1, 1, 100)
    y_ref, dy_ref, ddy_ref, _ = _tf1(x_eval)
    y_n, dy_n, ddy_n, _ = spline_native.eval_spline(x_eval)
    y_m, dy_m, ddy_m = spline_multi.eval_spline(x_eval)
    return dict(
        x_eval=x_eval,
        y_ref=y_ref, dy_ref=dy_ref, ddy_ref=ddy_ref,
        y_n=y_n, dy_n=dy_n, ddy_n=ddy_n,
        y_m=y_m, dy_m=dy_m, ddy_m=ddy_m,
    )


def test_1d_value_accuracy_native(splines_and_ref):
    d = splines_and_ref
    assert np.max(np.abs(d["y_n"] - d["y_ref"].ravel())) < 1e-13


def test_1d_derivative_accuracy_native(splines_and_ref):
    d = splines_and_ref
    assert np.max(np.abs(d["dy_n"] - d["dy_ref"].ravel())) < 1e-13


def test_1d_second_derivative_accuracy_native(splines_and_ref):
    d = splines_and_ref
    assert np.max(np.abs(d["ddy_n"] - d["ddy_ref"].ravel())) < 1e-13


def test_1d_value_accuracy_multivar(splines_and_ref):
    d = splines_and_ref
    assert np.max(np.abs(d["y_m"] - d["y_ref"])) < 1e-13


def test_1d_gradient_accuracy_multivar(splines_and_ref):
    d = splines_and_ref
    assert np.max(np.abs(d["dy_m"] - d["dy_ref"])) < 1e-13


def test_1d_hessian_accuracy_multivar(splines_and_ref):
    d = splines_and_ref
    assert np.max(np.abs(d["ddy_m"] - d["ddy_ref"])) < 1e-13


def test_1d_native_vs_multivar_consistency(splines_and_ref):
    d = splines_and_ref
    assert np.max(np.abs(d["y_n"] - d["y_m"].ravel())) < 1e-10
    assert np.max(np.abs(d["dy_n"] - d["dy_m"].ravel())) < 1e-10
    assert np.max(np.abs(d["ddy_n"] - d["ddy_m"].ravel())) < 1e-10
