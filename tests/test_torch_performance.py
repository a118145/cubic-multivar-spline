"""Performance benchmarks comparing numpy vs torch backends."""

import time
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_functions import test_function_1 as _tf1
from cubicmultispline.Spline1D import Spline1D
from cubicmultispline.Spline import Spline
from cubicmultispline.TorchSpline1D import TorchSpline1D
from cubicmultispline.TorchSpline import TorchSpline

benchmark = pytest.mark.benchmark


def _median_time(fn, runs=3):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return np.median(times)


def _print_table(title, rows):
    print(f"\n{title}")
    print(f"{'scenario':<35} {'n':>10} {'time (s)':>12} {'pts/sec':>12}")
    print("-" * 73)
    for scenario, n, t in rows:
        pts = n / t if t > 0 else float('inf')
        print(f"{scenario:<35} {n:>10} {t:>12.4f} {pts:>12.0f}")


# ── Construction benchmarks ──────────────────────────────────────────


@benchmark
def test_bench_2d_construction_comparison():
    rows = []
    for shape in [(10, 10), (20, 20), (50, 50)]:
        xs = [np.linspace(0, 1, n) for n in shape]
        y, _, _, _ = _tf1(*xs)
        interval = tuple((0, 1, n) for n in shape)
        bc = (("not-a-knot", "not-a-knot"),) * 2
        bc_val = ((0.0, 0.0),) * 2
        n_pts = shape[0] * shape[1]

        t_np = _median_time(lambda: Spline(interval, y, bc, bc_val))
        rows.append(("2D construction (numpy)", n_pts, t_np))

        yt = torch.tensor(y)
        t_torch = _median_time(lambda: TorchSpline(interval, yt, bc, bc_val))
        rows.append(("2D construction (torch)", n_pts, t_torch))

    _print_table("2D Construction: numpy vs torch", rows)


@benchmark
def test_bench_3d_construction_comparison():
    rows = []
    for shape in [(5, 5, 5), (10, 10, 10)]:
        xs = [np.linspace(0, 1, n) for n in shape]
        y, _, _, _ = _tf1(*xs)
        interval = tuple((0, 1, n) for n in shape)
        bc = (("not-a-knot", "not-a-knot"),) * 3
        bc_val = ((0.0, 0.0),) * 3
        n_pts = int(np.prod(shape))

        t_np = _median_time(lambda: Spline(interval, y, bc, bc_val))
        rows.append(("3D construction (numpy)", n_pts, t_np))

        yt = torch.tensor(y)
        t_torch = _median_time(lambda: TorchSpline(interval, yt, bc, bc_val))
        rows.append(("3D construction (torch)", n_pts, t_torch))

    _print_table("3D Construction: numpy vs torch", rows)


# ── Evaluation benchmarks ────────────────────────────────────────────


@benchmark
def test_bench_2d_evaluation_comparison():
    shape = (20, 20)
    xs = [np.linspace(0, 1, n) for n in shape]
    y, _, _, _ = _tf1(*xs)
    interval = tuple((0, 1, n) for n in shape)
    bc = (("not-a-knot", "not-a-knot"),) * 2
    bc_val = ((0.0, 0.0),) * 2
    np_spline = Spline(interval, y, bc, bc_val)
    t_spline = TorchSpline(interval, torch.tensor(y), bc, bc_val)

    rows = []
    for n_eval in [100, 1000, 5000]:
        coords_np = np.random.rand(n_eval, 2)
        coords_t = torch.tensor(coords_np)
        t_np = _median_time(lambda: np_spline.eval_spline(coords_np))
        rows.append(("2D eval (numpy)", n_eval, t_np))
        t_torch = _median_time(lambda: t_spline.eval_spline(coords_t))
        rows.append(("2D eval (torch)", n_eval, t_torch))

    _print_table("2D Evaluation: numpy vs torch", rows)


@benchmark
def test_bench_3d_evaluation_comparison():
    shape = (10, 10, 10)
    xs = [np.linspace(0, 1, n) for n in shape]
    y, _, _, _ = _tf1(*xs)
    interval = tuple((0, 1, n) for n in shape)
    bc = (("not-a-knot", "not-a-knot"),) * 3
    bc_val = ((0.0, 0.0),) * 3
    np_spline = Spline(interval, y, bc, bc_val)
    t_spline = TorchSpline(interval, torch.tensor(y), bc, bc_val)

    rows = []
    for n_eval in [100, 1000, 5000]:
        coords_np = np.random.rand(n_eval, 3)
        coords_t = torch.tensor(coords_np)
        t_np = _median_time(lambda: np_spline.eval_spline(coords_np))
        rows.append(("3D eval (numpy)", n_eval, t_np))
        t_torch = _median_time(lambda: t_spline.eval_spline(coords_t))
        rows.append(("3D eval (torch)", n_eval, t_torch))

    _print_table("3D Evaluation: numpy vs torch", rows)


# ── GPU benchmarks ───────────────────────────────────────────────────


@benchmark
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bench_2d_gpu_vs_cpu():
    shape = (20, 20)
    xs = [np.linspace(0, 1, n) for n in shape]
    y, _, _, _ = _tf1(*xs)
    interval = tuple((0, 1, n) for n in shape)
    bc = (("not-a-knot", "not-a-knot"),) * 2
    bc_val = ((0.0, 0.0),) * 2

    spline_cpu = TorchSpline(interval, torch.tensor(y, device="cpu"), bc, bc_val)
    spline_gpu = TorchSpline(interval, torch.tensor(y, device="cuda"), bc, bc_val)

    rows = []
    for n_eval in [100, 1000, 5000]:
        coords_cpu = torch.rand(n_eval, 2, dtype=torch.float64)
        coords_gpu = coords_cpu.to("cuda")
        t_cpu = _median_time(lambda: spline_cpu.eval_spline(coords_cpu))
        rows.append(("2D eval (CPU)", n_eval, t_cpu))
        # Sync for accurate GPU timing
        def gpu_eval():
            spline_gpu.eval_spline(coords_gpu)
            torch.cuda.synchronize()
        t_gpu = _median_time(gpu_eval)
        rows.append(("2D eval (GPU)", n_eval, t_gpu))

    _print_table("2D Evaluation: CPU vs GPU", rows)
