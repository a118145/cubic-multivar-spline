import time
import numpy as np
import pytest
from test_functions import test_function_1 as _tf1, test_function_2 as _tf2
from cubicmultispline.Spline1D import Spline1D
from cubicmultispline.Spline import Spline

benchmark = pytest.mark.benchmark


def _median_time(fn, runs=3):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return np.median(times)


def _print_table(title, rows):
    """Print a table with scenario, n, time, points/sec."""
    print(f"\n{title}")
    print(f"{'scenario':<30} {'n':>10} {'time (s)':>12} {'pts/sec':>12}")
    print("-" * 68)
    for scenario, n, t in rows:
        pts = n / t if t > 0 else float('inf')
        print(f"{scenario:<30} {n:>10} {t:>12.4f} {pts:>12.0f}")


# ── Timing benchmarks ──────────────────────────────────────────────


@benchmark
def test_bench_1d_construction():
    rows = []
    for n in [50, 100, 500, 1000]:
        x = np.linspace(0, 1, n)
        y, _, _, _ = _tf1(x)
        t = _median_time(lambda: Spline1D((0, 1), y, ("not-a-knot", "not-a-knot"), (0.0, 0.0)))
        rows.append(("1D construction", n, t))
    _print_table("1D Construction", rows)


@benchmark
def test_bench_1d_evaluation():
    n_pts = 100
    x = np.linspace(0, 1, n_pts)
    y, _, _, _ = _tf1(x)
    spline = Spline1D((0, 1), y, ("not-a-knot", "not-a-knot"), (0.0, 0.0))
    rows = []
    for n_eval in [100, 1000, 10000]:
        x_eval = np.linspace(0, 1, n_eval)
        t = _median_time(lambda: spline.eval_spline(x_eval))
        rows.append(("1D evaluation", n_eval, t))
    _print_table("1D Evaluation", rows)


@benchmark
def test_bench_2d_construction():
    rows = []
    for shape in [(10, 10), (20, 20), (50, 50)]:
        xs = [np.linspace(0, 1, n) for n in shape]
        y, _, _, _ = _tf1(*xs)
        interval = tuple((0, 1, n) for n in shape)
        bc = (("not-a-knot", "not-a-knot"),) * 2
        bc_value = ((0.0, 0.0),) * 2
        t = _median_time(lambda: Spline(interval, y, bc, bc_value))
        rows.append(("2D construction", shape[0] * shape[1], t))
    _print_table("2D Construction", rows)


@benchmark
def test_bench_2d_evaluation():
    shape = (20, 20)
    xs = [np.linspace(0, 1, n) for n in shape]
    y, _, _, _ = _tf1(*xs)
    interval = tuple((0, 1, n) for n in shape)
    bc = (("not-a-knot", "not-a-knot"),) * 2
    bc_value = ((0.0, 0.0),) * 2
    spline = Spline(interval, y, bc, bc_value)
    rows = []
    for n_eval in [100, 1000, 5000]:
        coords = np.random.rand(n_eval, 2)
        t = _median_time(lambda: spline.eval_spline(coords))
        rows.append(("2D evaluation", n_eval, t))
    _print_table("2D Evaluation", rows)


@benchmark
def test_bench_3d_construction():
    rows = []
    for shape in [(5, 5, 5), (10, 10, 10), (15, 15, 15)]:
        xs = [np.linspace(0, 1, n) for n in shape]
        y, _, _, _ = _tf1(*xs)
        interval = tuple((0, 1, n) for n in shape)
        bc = (("not-a-knot", "not-a-knot"),) * 3
        bc_value = ((0.0, 0.0),) * 3
        t = _median_time(lambda: Spline(interval, y, bc, bc_value))
        rows.append(("3D construction", int(np.prod(shape)), t))
    _print_table("3D Construction", rows)


@benchmark
def test_bench_3d_evaluation():
    shape = (10, 10, 10)
    xs = [np.linspace(0, 1, n) for n in shape]
    y, _, _, _ = _tf1(*xs)
    interval = tuple((0, 1, n) for n in shape)
    bc = (("not-a-knot", "not-a-knot"),) * 3
    bc_value = ((0.0, 0.0),) * 3
    spline = Spline(interval, y, bc, bc_value)
    rows = []
    for n_eval in [100, 500, 1000, 10000, 100000]:
        coords = np.random.rand(n_eval, 3)
        t = _median_time(lambda: spline.eval_spline(coords))
        rows.append(("3D evaluation", n_eval, t))
    _print_table("3D Evaluation", rows)


# ── Soft limits ─────────────────────────────────────────────────────


@benchmark
def test_1d_construction_soft_limit():
    x = np.linspace(0, 1, 1000)
    y, _, _, _ = _tf1(x)
    t = _median_time(lambda: Spline1D((0, 1), y, ("not-a-knot", "not-a-knot"), (0.0, 0.0)))
    assert t < 1.0, f"1D 1000-point construction took {t:.2f}s (limit: 1s)"


@benchmark
def test_2d_construction_soft_limit():
    shape = (50, 50)
    xs = [np.linspace(0, 1, n) for n in shape]
    y, _, _, _ = _tf1(*xs)
    interval = tuple((0, 1, n) for n in shape)
    bc = (("not-a-knot", "not-a-knot"),) * 2
    bc_value = ((0.0, 0.0),) * 2
    t = _median_time(lambda: Spline(interval, y, bc, bc_value))
    assert t < 10.0, f"2D 50x50 construction took {t:.2f}s (limit: 10s)"


@benchmark
def test_1d_evaluation_soft_limit():
    x = np.linspace(0, 1, 100)
    y, _, _, _ = _tf1(x)
    spline = Spline1D((0, 1), y, ("not-a-knot", "not-a-knot"), (0.0, 0.0))
    x_eval = np.linspace(0, 1, 10000)
    t = _median_time(lambda: spline.eval_spline(x_eval))
    assert t < 2.0, f"1D 10000-point evaluation took {t:.2f}s (limit: 2s)"


# ── Scaling verification ────────────────────────────────────────────


@benchmark
def test_scaling_dimension():
    """Construction time 1D->4D with 8 pts/dim, verify increases are bounded."""
    times = []
    for ndim in range(1, 5):
        shape = (8,) * ndim
        xs = [np.linspace(0, 1, 8) for _ in range(ndim)]
        y, _, _, _ = _tf1(*xs)
        interval = tuple((0, 1, 8) for _ in range(ndim))
        bc = (("not-a-knot", "not-a-knot"),) * ndim
        bc_value = ((0.0, 0.0),) * ndim
        t = _median_time(lambda: Spline(interval, y, bc, bc_value))
        times.append(t)
    # each added dimension should increase time by at most 100x
    for i in range(1, len(times)):
        if times[i - 1] > 1e-6:
            ratio = times[i] / times[i - 1]
            assert ratio < 100, f"Dim {i} -> {i+1} ratio {ratio:.1f} > 100"


@benchmark
def test_scaling_eval_linear():
    """Doubling eval points roughly doubles eval time (within 3x factor)."""
    shape = (20, 20)
    xs = [np.linspace(0, 1, n) for n in shape]
    y, _, _, _ = _tf1(*xs)
    interval = tuple((0, 1, n) for n in shape)
    bc = (("not-a-knot", "not-a-knot"),) * 2
    bc_value = ((0.0, 0.0),) * 2
    spline = Spline(interval, y, bc, bc_value)

    coords_500 = np.random.rand(500, 2)
    coords_1000 = np.random.rand(1000, 2)
    t1 = _median_time(lambda: spline.eval_spline(coords_500))
    t2 = _median_time(lambda: spline.eval_spline(coords_1000))
    if t1 > 1e-6:
        ratio = t2 / t1
        assert ratio < 6, f"Doubling eval points gave {ratio:.1f}x time (expected <6x)"


@benchmark
def test_convergence_rate_2d_periodic():
    """Verify O(h^4) convergence for _tf2."""
    y_min, y_max = 0.0, 2.0 * np.pi
    x_eval = np.linspace(0.05, 0.95, 20)
    y_eval = np.linspace(y_min + 0.1, y_max - 0.1, 20)
    f_ref, _, _, _ = _tf2(x_eval, y_eval, y_bounds=(y_min, y_max))
    grids = np.meshgrid(x_eval, y_eval, indexing='ij')
    coords = np.column_stack([g.ravel() for g in grids])

    errors = []
    for n2 in [10, 20, 40]:
        x = np.linspace(0, 1, 15)
        y = np.linspace(y_min, y_max, n2, endpoint=True)
        vals, _, _, _ = _tf2(x, y, y_bounds=(y_min, y_max))
        interval = ((0, 1, 15), (y_min, y_max, n2))
        bc = (("not-a-knot", "not-a-knot"), ("periodic", "periodic"))
        bc_value = ((0.0, 0.0), (0.0, 0.0))
        spline = Spline(interval, vals, bc, bc_value)
        f, _, _ = spline.eval_spline(coords)
        errors.append(np.max(np.abs(f - f_ref)))

    ratio = errors[0] / errors[1] if errors[1] > 0 else float('inf')
    assert ratio > 4, f"Convergence ratio {ratio:.1f} < 4 (expected ~16 for O(h^4))"
