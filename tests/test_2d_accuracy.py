import numpy as np
import pytest
from test_functions import test_function_1 as _tf1
from cubicmultispline.Spline import Spline


def test_2d_value_accuracy(setup_2d):
    s = setup_2d
    spline = Spline(s["interval"], s["y"], s["bc"], s["bc_value"])
    x_eval = np.linspace(0, 1, 20)
    y_eval = np.linspace(0, 1, 20)
    y_ref, grad_ref, hess_ref, _ = _tf1(x_eval, y_eval)
    grids = np.meshgrid(x_eval, y_eval, indexing='ij')
    coords = np.column_stack([g.ravel() for g in grids])
    f, grad, hess = spline.eval_spline(coords)
    assert np.max(np.abs(f - y_ref)) < 1e-10


def test_2d_gradient_accuracy(setup_2d):
    s = setup_2d
    spline = Spline(s["interval"], s["y"], s["bc"], s["bc_value"])
    x_eval = np.linspace(0, 1, 20)
    y_eval = np.linspace(0, 1, 20)
    y_ref, grad_ref, hess_ref, _ = _tf1(x_eval, y_eval)
    grids = np.meshgrid(x_eval, y_eval, indexing='ij')
    coords = np.column_stack([g.ravel() for g in grids])
    f, grad, hess = spline.eval_spline(coords)
    assert np.max(np.abs(grad - grad_ref)) < 1e-10


def test_2d_hessian_accuracy(setup_2d):
    s = setup_2d
    spline = Spline(s["interval"], s["y"], s["bc"], s["bc_value"])
    x_eval = np.linspace(0, 1, 20)
    y_eval = np.linspace(0, 1, 20)
    y_ref, grad_ref, hess_ref, _ = _tf1(x_eval, y_eval)
    grids = np.meshgrid(x_eval, y_eval, indexing='ij')
    coords = np.column_stack([g.ravel() for g in grids])
    f, grad, hess = spline.eval_spline(coords)
    assert np.max(np.abs(hess - hess_ref)) < 1e-8


def test_2d_interpolation_at_nodes(setup_2d):
    s = setup_2d
    spline = Spline(s["interval"], s["y"], s["bc"], s["bc_value"])
    grids = np.meshgrid(s["x1"], s["x2"], indexing='ij')
    coords = np.column_stack([g.ravel() for g in grids])
    f, _, _ = spline.eval_spline(coords)
    assert np.max(np.abs(f - s["y"])) < 1e-12
