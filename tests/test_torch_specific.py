"""Torch-specific tests: device, dtype, numpy↔torch consistency."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_functions import test_function_1 as _tf1
from cubicmultispline.Spline1D import Spline1D
from cubicmultispline.Spline import Spline
from cubicmultispline.TorchSpline1D import TorchSpline1D
from cubicmultispline.TorchSpline import TorchSpline


# ── numpy vs torch consistency ───────────────────────────────────────


def test_torch_vs_numpy_1d_coefficients():
    x = np.linspace(-1, 1, 11)
    y, _, _, _ = _tf1(x)
    np_spline = Spline1D((-1, 1), y, ("not-a-knot", "not-a-knot"), (0.0, 0.0))
    t_spline = TorchSpline1D((-1, 1), torch.tensor(y), ("not-a-knot", "not-a-knot"), (0.0, 0.0))
    diff = np.max(np.abs(np_spline.coeff - t_spline.coeff.numpy()))
    assert diff < 1e-12, f"Coefficient diff: {diff}"


def test_torch_vs_numpy_1d_eval():
    x = np.linspace(-1, 1, 11)
    y, _, _, _ = _tf1(x)
    np_spline = Spline1D((-1, 1), y, ("not-a-knot", "not-a-knot"), (0.0, 0.0))
    t_spline = TorchSpline1D((-1, 1), torch.tensor(y), ("not-a-knot", "not-a-knot"), (0.0, 0.0))
    x_eval = np.linspace(-1, 1, 50)
    np_vals = np_spline.eval_spline(x_eval)
    t_vals = t_spline.eval_spline(torch.tensor(x_eval))
    for np_v, t_v in zip(np_vals, t_vals):
        diff = np.max(np.abs(np.asarray(np_v) - t_v.numpy()))
        assert diff < 1e-12, f"Eval diff: {diff}"


def test_torch_vs_numpy_2d_coefficients():
    shape = (11, 13)
    x1 = np.linspace(0, 1, shape[0])
    x2 = np.linspace(0, 1, shape[1])
    y, _, _, _ = _tf1(x1, x2)
    interval = ((0, 1, shape[0]), (0, 1, shape[1]))
    bc = (("not-a-knot", "not-a-knot"), ("not-a-knot", "not-a-knot"))
    bc_val = ((0.0, 0.0), (0.0, 0.0))
    np_spline = Spline(interval, y, bc, bc_val)
    t_spline = TorchSpline(interval, torch.tensor(y), bc, bc_val)
    diff = np.max(np.abs(np_spline.coeff - t_spline.coeff.numpy()))
    assert diff < 1e-12, f"2D coefficient diff: {diff}"


def test_torch_vs_numpy_2d_eval():
    shape = (11, 13)
    x1 = np.linspace(0, 1, shape[0])
    x2 = np.linspace(0, 1, shape[1])
    y, _, _, _ = _tf1(x1, x2)
    interval = ((0, 1, shape[0]), (0, 1, shape[1]))
    bc = (("not-a-knot", "not-a-knot"), ("not-a-knot", "not-a-knot"))
    bc_val = ((0.0, 0.0), (0.0, 0.0))
    np_spline = Spline(interval, y, bc, bc_val)
    t_spline = TorchSpline(interval, torch.tensor(y), bc, bc_val)
    coords_np = np.random.RandomState(42).rand(30, 2)
    coords_t = torch.tensor(coords_np)
    f_np, grad_np, hess_np = np_spline.eval_spline(coords_np)
    f_t, grad_t, hess_t = t_spline.eval_spline(coords_t)
    assert np.max(np.abs(f_np - f_t.numpy())) < 1e-12
    assert np.max(np.abs(grad_np - grad_t.numpy())) < 1e-12
    assert np.max(np.abs(hess_np - hess_t.numpy())) < 1e-10


# ── device tests ─────────────────────────────────────────────────────


def test_device_cpu():
    x = np.linspace(0, 1, 11)
    y, _, _, _ = _tf1(x)
    yv = torch.tensor(y, device=torch.device("cpu"))
    spline = TorchSpline1D((0, 1), yv)
    assert spline.coeff.device == torch.device("cpu")
    val, _, _, _ = spline.eval_spline(torch.tensor(0.5))
    assert val.device == torch.device("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_cuda():
    x = np.linspace(0, 1, 11)
    y, _, _, _ = _tf1(x)
    yv = torch.tensor(y, device=torch.device("cuda"))
    spline = TorchSpline1D((0, 1), yv)
    assert spline.coeff.device.type == "cuda"
    val, _, _, _ = spline.eval_spline(torch.tensor(0.5, device=torch.device("cuda")))
    assert val.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_cuda_multidim():
    shape = (7, 9)
    x1 = np.linspace(0, 1, shape[0])
    x2 = np.linspace(0, 1, shape[1])
    y, _, _, _ = _tf1(x1, x2)
    interval = ((0, 1, shape[0]), (0, 1, shape[1]))
    yv = torch.tensor(y, device=torch.device("cuda"))
    spline = TorchSpline(interval, yv)
    assert spline.coeff.device.type == "cuda"
    coords = torch.rand(5, 2, device=torch.device("cuda"), dtype=torch.float64)
    f, grad, hess = spline.eval_spline(coords)
    assert f.device.type == "cuda"


# ── dtype tests ──────────────────────────────────────────────────────


def test_dtype_float32():
    x = np.linspace(0, 1, 11)
    y, _, _, _ = _tf1(x)
    bc = ("not-a-knot", "not-a-knot")
    yv = torch.tensor(y, dtype=torch.float32)
    spline = TorchSpline1D((0, 1), yv, bc, (0.0, 0.0))
    assert spline.coeff.dtype == torch.float32
    x_eval = torch.linspace(0, 1, 50, dtype=torch.float32)
    vals, _, _, _ = spline.eval_spline(x_eval)
    assert vals.dtype == torch.float32
    # Relaxed tolerance for float32
    y_ref = torch.tensor(_tf1(x_eval.numpy())[0].ravel(), dtype=torch.float32)
    assert torch.max(torch.abs(vals - y_ref)).item() < 1e-5


def test_dtype_float64():
    x = np.linspace(0, 1, 11)
    y, _, _, _ = _tf1(x)
    yv = torch.tensor(y, dtype=torch.float64)
    spline = TorchSpline1D((0, 1), yv)
    assert spline.coeff.dtype == torch.float64
    x_eval = torch.linspace(0, 1, 50, dtype=torch.float64)
    vals, _, _, _ = spline.eval_spline(x_eval)
    assert vals.dtype == torch.float64


def test_dtype_float32_multidim():
    shape = (7, 9)
    x1 = np.linspace(0, 1, shape[0])
    x2 = np.linspace(0, 1, shape[1])
    y, _, _, _ = _tf1(x1, x2)
    interval = ((0, 1, shape[0]), (0, 1, shape[1]))
    yv = torch.tensor(y, dtype=torch.float32)
    spline = TorchSpline(interval, yv)
    assert spline.coeff.dtype == torch.float32
    coords = torch.rand(10, 2, dtype=torch.float32)
    f, grad, hess = spline.eval_spline(coords)
    assert f.dtype == torch.float32


# ── input conversion ─────────────────────────────────────────────────


def test_input_conversion_numpy_array():
    x = np.linspace(0, 1, 11)
    y, _, _, _ = _tf1(x)
    # Pass numpy array directly — should auto-convert
    spline = TorchSpline1D((0, 1), y)
    assert isinstance(spline.coeff, torch.Tensor)
    assert spline.coeff.dtype == torch.float64


def test_input_conversion_list():
    y = [0.0, 1.0, 4.0, 9.0, 16.0]
    spline = TorchSpline1D((0, 1), y)
    assert isinstance(spline.coeff, torch.Tensor)


def test_input_conversion_multidim():
    shape = (7, 9)
    x1 = np.linspace(0, 1, shape[0])
    x2 = np.linspace(0, 1, shape[1])
    y, _, _, _ = _tf1(x1, x2)
    interval = ((0, 1, shape[0]), (0, 1, shape[1]))
    # Pass numpy array — should auto-convert
    spline = TorchSpline(interval, y)
    assert isinstance(spline.coeff, torch.Tensor)


# ── scalar eval ──────────────────────────────────────────────────────


def test_scalar_eval_1d():
    x = np.linspace(0, 1, 11)
    y, _, _, _ = _tf1(x)
    spline = TorchSpline1D((0, 1), torch.tensor(y))
    val, d1, d2, d3 = spline.eval_spline(torch.tensor(0.5))
    assert val.dim() == 0  # 0-dim tensor
    assert d1.dim() == 0
