"""Accuracy tests for TorchSpline1D and TorchSpline — mirrors the numpy test suite."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_functions import test_function_1 as _tf1, test_function_2 as _tf2
from cubicmultispline.TorchSpline1D import TorchSpline1D
from cubicmultispline.TorchSpline import TorchSpline


# ── 1D tests ─────────────────────────────────────────────────────────


def test_1d_value_accuracy():
    shape = (11,)
    x = np.linspace(-1, 1, shape[0])
    y, _, _, _ = _tf1(x)
    bc = ("not-a-knot", "not-a-knot")
    spline = TorchSpline1D((-1, 1), torch.tensor(y), bc, (0.0, 0.0))
    x_eval_np = np.linspace(-1, 1, 100)
    x_eval = torch.tensor(x_eval_np)
    y_ref, _, _, _ = _tf1(x_eval_np)
    y_s, _, _, _ = spline.eval_spline(x_eval)
    assert torch.max(torch.abs(y_s - torch.tensor(y_ref.ravel()))).item() < 1e-13


def test_1d_derivative_accuracy():
    x = np.linspace(-1, 1, 11)
    y, _, _, _ = _tf1(x)
    bc = ("not-a-knot", "not-a-knot")
    spline = TorchSpline1D((-1, 1), torch.tensor(y), bc, (0.0, 0.0))
    x_eval_np = np.linspace(-1, 1, 100)
    x_eval = torch.tensor(x_eval_np)
    _, dy_ref, _, _ = _tf1(x_eval_np)
    _, dy_s, _, _ = spline.eval_spline(x_eval)
    assert torch.max(torch.abs(dy_s - torch.tensor(dy_ref.ravel()))).item() < 1e-13


def test_1d_second_derivative_accuracy():
    x = np.linspace(-1, 1, 11)
    y, _, _, _ = _tf1(x)
    bc = ("not-a-knot", "not-a-knot")
    spline = TorchSpline1D((-1, 1), torch.tensor(y), bc, (0.0, 0.0))
    x_eval_np = np.linspace(-1, 1, 100)
    x_eval = torch.tensor(x_eval_np)
    _, _, ddy_ref, _ = _tf1(x_eval_np)
    _, _, ddy_s, _ = spline.eval_spline(x_eval)
    assert torch.max(torch.abs(ddy_s - torch.tensor(ddy_ref.ravel()))).item() < 5e-13


# ── 1D boundary conditions ──────────────────────────────────────────


@pytest.mark.parametrize("bc", [
    "not-a-knot",
    "second_derivative",
    "first_derivative",
    "periodic",
])
def test_1d_boundary_conditions(bc):
    n = 21
    x = np.linspace(0, 2 * np.pi, n, endpoint=True)
    y = np.sin(x)
    if bc == "periodic":
        y[-1] = y[0]  # ensure exact match for periodic BC
    bc_type = (bc, bc)
    bc_val = (0.0, 0.0)
    if bc == "first_derivative":
        bc_val = (1.0, 1.0)  # cos(0)=1, cos(2pi)=1
    spline = TorchSpline1D((0, 2 * np.pi), torch.tensor(y), bc_type, bc_val)
    x_eval = torch.linspace(0.1, 2 * np.pi - 0.1, 50)
    y_s, _, _, _ = spline.eval_spline(x_eval)
    y_ref = torch.sin(x_eval)
    assert torch.max(torch.abs(y_s - y_ref)).item() < 1e-3


# ── 1D multivar wrapper ─────────────────────────────────────────────


def test_1d_via_torchspline():
    x = np.linspace(-1, 1, 11)
    y, _, _, _ = _tf1(x)
    interval = ((-1, 1, 11),)
    bc = (("not-a-knot", "not-a-knot"),)
    bc_val = ((0.0, 0.0),)
    spline = TorchSpline(interval, torch.tensor(y), bc, bc_val)
    x_eval = torch.linspace(-1, 1, 100).unsqueeze(1)
    y_ref, _, _, _ = _tf1(x_eval.squeeze().numpy())
    f, _, _ = spline.eval_spline(x_eval)
    assert torch.max(torch.abs(f - torch.tensor(y_ref))).item() < 1e-13


# ── 2D tests ─────────────────────────────────────────────────────────


def _make_2d_spline():
    shape = (11, 13)
    x1 = np.linspace(0, 1, shape[0])
    x2 = np.linspace(0, 1, shape[1])
    y, _, _, _ = _tf1(x1, x2)
    interval = ((0, 1, shape[0]), (0, 1, shape[1]))
    bc = (("not-a-knot", "not-a-knot"), ("not-a-knot", "not-a-knot"))
    bc_val = ((0.0, 0.0), (0.0, 0.0))
    return TorchSpline(interval, torch.tensor(y), bc, bc_val)


def _eval_2d_ref():
    x_eval = np.linspace(0, 1, 20)
    y_eval = np.linspace(0, 1, 20)
    y_ref, grad_ref, hess_ref, _ = _tf1(x_eval, y_eval)
    grids = np.meshgrid(x_eval, y_eval, indexing='ij')
    coords = torch.tensor(np.column_stack([g.ravel() for g in grids]))
    return coords, y_ref, grad_ref, hess_ref


def test_2d_value_accuracy():
    spline = _make_2d_spline()
    coords, y_ref, _, _ = _eval_2d_ref()
    f, _, _ = spline.eval_spline(coords)
    assert torch.max(torch.abs(f - torch.tensor(y_ref))).item() < 1e-10


def test_2d_gradient_accuracy():
    spline = _make_2d_spline()
    coords, _, grad_ref, _ = _eval_2d_ref()
    _, grad, _ = spline.eval_spline(coords)
    assert torch.max(torch.abs(grad - torch.tensor(grad_ref))).item() < 1e-10


def test_2d_hessian_accuracy():
    spline = _make_2d_spline()
    coords, _, _, hess_ref = _eval_2d_ref()
    _, _, hess = spline.eval_spline(coords)
    assert torch.max(torch.abs(hess - torch.tensor(hess_ref))).item() < 1e-8


def test_2d_interpolation_at_nodes():
    shape = (11, 13)
    x1 = np.linspace(0, 1, shape[0])
    x2 = np.linspace(0, 1, shape[1])
    y, _, _, _ = _tf1(x1, x2)
    interval = ((0, 1, shape[0]), (0, 1, shape[1]))
    bc = (("not-a-knot", "not-a-knot"), ("not-a-knot", "not-a-knot"))
    bc_val = ((0.0, 0.0), (0.0, 0.0))
    spline = TorchSpline(interval, torch.tensor(y), bc, bc_val)
    grids = np.meshgrid(x1, x2, indexing='ij')
    coords = torch.tensor(np.column_stack([g.ravel() for g in grids]))
    f, _, _ = spline.eval_spline(coords)
    assert torch.max(torch.abs(f - torch.tensor(y))).item() < 1e-12


# ── 2D periodic + convergence ────────────────────────────────────────


def test_2d_periodic_convergence():
    y_min, y_max = 0.0, 2.0 * np.pi
    x_eval = np.linspace(0.05, 0.95, 20)
    y_eval = np.linspace(y_min + 0.1, y_max - 0.1, 20)
    f_ref, _, _, _ = _tf2(x_eval, y_eval, y_bounds=(y_min, y_max))
    grids = np.meshgrid(x_eval, y_eval, indexing='ij')
    coords = torch.tensor(np.column_stack([g.ravel() for g in grids]))

    errors = []
    for n2 in [10, 20, 40]:
        x = np.linspace(0, 1, 15)
        y = np.linspace(y_min, y_max, n2, endpoint=True)
        vals, _, _, _ = _tf2(x, y, y_bounds=(y_min, y_max))
        interval = ((0, 1, 15), (y_min, y_max, n2))
        bc = (("not-a-knot", "not-a-knot"), ("periodic", "periodic"))
        bc_val = ((0.0, 0.0), (0.0, 0.0))
        spline = TorchSpline(interval, torch.tensor(vals), bc, bc_val)
        f, _, _ = spline.eval_spline(coords)
        errors.append(torch.max(torch.abs(f - torch.tensor(f_ref))).item())

    ratio = errors[0] / errors[1] if errors[1] > 0 else float('inf')
    assert ratio > 4, f"Convergence ratio {ratio:.1f} < 4 (expected ~16 for O(h^4))"


# ── 3D test ──────────────────────────────────────────────────────────


def test_3d_accuracy():
    shape = (7, 9, 8)
    xs = [np.linspace(0, 1, n) for n in shape]
    y, _, _, _ = _tf1(*xs)
    interval = tuple((0, 1, n) for n in shape)
    bc = (("not-a-knot", "not-a-knot"),) * 3
    bc_val = ((0.0, 0.0),) * 3
    spline = TorchSpline(interval, torch.tensor(y), bc, bc_val)

    x_eval = [np.linspace(0, 1, 5) for _ in range(3)]
    y_ref, grad_ref, _, _ = _tf1(*x_eval)
    grids = np.meshgrid(*x_eval, indexing='ij')
    coords = torch.tensor(np.column_stack([g.ravel() for g in grids]))
    f, grad, _ = spline.eval_spline(coords)
    assert torch.max(torch.abs(f - torch.tensor(y_ref))).item() < 1e-8
    assert torch.max(torch.abs(grad - torch.tensor(grad_ref))).item() < 1e-6


# ── nD smoke tests ───────────────────────────────────────────────────


@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
def test_nd_smoke(ndim):
    n = 6
    shape = (n,) * ndim
    xs = [np.linspace(0, 1, n) for _ in range(ndim)]
    y, _, _, _ = _tf1(*xs)
    interval = tuple((0, 1, n) for _ in range(ndim))
    bc = (("not-a-knot", "not-a-knot"),) * ndim
    bc_val = ((0.0, 0.0),) * ndim
    spline = TorchSpline(interval, torch.tensor(y), bc, bc_val)
    # Evaluate at a single interior point
    pt = torch.full((1, ndim), 0.5, dtype=torch.float64)
    f, grad, hess = spline.eval_spline(pt)
    assert f.shape == (1,)
    assert grad.shape == (1, ndim)
    assert hess.shape == (1, ndim, ndim)
