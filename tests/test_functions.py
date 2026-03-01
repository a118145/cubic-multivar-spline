"""
Test functions for the multidimensional spline evaluator
==========================================================

test_function_1 : cubic tensor product polynomial, arbitrary dimension
test_function_2 : polynomial × cosine (2D, periodic in dim 2)

Both functions take one or more 1D arrays as input (one per
dimension) and return (result, grad, hess), where:

  result : (N,)       with N = n1 * n2 * ... * nndim
  grad   : (N, ndim)
  hess   : (N, ndim, ndim)

np.reshape(result, (n1, n2, ..., nndim)) returns the coordinate-indexed form.

Convention: The input arrays x1, x2, ... are spanned to the full grid
with np.meshgrid(..., indexing='ij'). Then everything is flattened to (N, ...).
"""

import numpy as np
import matplotlib.pyplot as plt


def _meshgrid_flat(*xs):
    """
    Creates the Cartesian grid from xs and returns flat arrays.

    Returns
    -------
    grids : list of (N,) arrays, one per dimension
    shape : tuple (n1, n2, ..., nd) — for np.reshape
    """
    shape = tuple(len(x) for x in xs)
    grids_nd = np.meshgrid(*xs, indexing='ij')     # each (n1, n2, ..., nd)
    grids = [g.ravel() for g in grids_nd]          # each (N,)
    return grids, shape


# ======================================================================= #
#  Test function 1: cubic tensor product polynomial                        #
#                                                                          #
#  f(x1, x2, ..., xd) = prod_{k=1}^{d}  p(xk)                           #
#  with p(x) = a3*x^3 + a2*x^2 + a1*x + a0                               #
#                                                                          #
#  Degree 3 → exactly reproducible by a cubic spline.                   #
# ======================================================================= #

# Coefficients of the 1D polynomial (freely selectable)
_P = np.array([1.0, -2.0, 0.5, 0.1])   # [a3, a2, a1, a0]


def _p(x):
    """p(x) = a3 x^3 + a2 x^2 + a1 x + a0"""
    return np.polyval(_P, x)


def _dp(x):
    """p'(x) = 3 a3 x^2 + 2 a2 x + a1"""
    return np.polyval(np.polyder(_P), x)


def _d2p(x):
    """p''(x) = 6 a3 x + 2 a2"""
    return np.polyval(np.polyder(_P, 2), x)


def test_function_1(*xs):
    """
    Cubic tensor product polynomial in arbitrary dimension.

    Parameters
    ----------
    *xs : 1D arrays, one per dimension (length n1, n2, ...)

    Returns
    -------
    result : (N,)         Function values,  N = prod(nk)
    grad   : (N, d)       Gradient
    hess   : (N, d, d)    Hessian matrix
    shape  : tuple        (n1, n2, ..., nd) for np.reshape
    """
    grids, shape = _meshgrid_flat(*xs)
    d = len(grids)
    N = int(np.prod(shape))

    # p and derivatives at each grid point
    pv  = [_p(g)   for g in grids]   # p(xk),   each (N,)
    dpv = [_dp(g)  for g in grids]   # p'(xk),  each (N,)
    d2pv = [_d2p(g) for g in grids]  # p''(xk), each (N,)

    # f = prod_k p(xk)
    result = np.ones(N)
    for k in range(d):
        result *= pv[k]

    # grad_j = p'(xj) * prod_{k≠j} p(xk) = f / p(xj) * p'(xj)
    # (safe via explicit product instead of division to avoid /0)
    grad = np.zeros((N, d))
    for j in range(d):
        g = dpv[j].copy()
        for k in range(d):
            if k != j:
                g *= pv[k]
        grad[:, j] = g

    # hess_{j,l} = p'(xj)*p'(xl) * prod_{k≠j,l} p(xk)   (j ≠ l)
    #            = p''(xj)        * prod_{k≠j}   p(xk)   (j == l)
    hess = np.zeros((N, d, d))
    for j in range(d):
        for l in range(j, d):
            h = np.ones(N)
            for k in range(d):
                if k == j and k == l:
                    h *= d2pv[k]
                elif k == j or k == l:
                    h *= dpv[k]
                else:
                    h *= pv[k]
            hess[:, j, l] = h
            if j != l:
                hess[:, l, j] = h   # Symmetry

    return result, grad, hess, shape


# ======================================================================= #
#  Test function 2: Polynomial × Cosine  (2D, periodic in dim 2)           #
#                                                                          #
#  f(x, y) = p(x) * cos(2π * (y - y_min) / (y_max - y_min))             #
#                                                                          #
#  → exactly periodic over the domain [y_min, y_max] in y                #
#  → cubic polynomial in x                                               #
# ======================================================================= #

def test_function_2(x, y, y_bounds=None):
    """
    Tensor product of cubic polynomial (dim 1) and cosine (dim 2).

    Parameters
    ----------
    x        : 1D array (n1,) — support points in dimension 1
    y        : 1D array (n2,) — support points in dimension 2 (periodic)
    y_bounds : (y_min, y_max) — period domain; default: (y[0], y[-1])

    Returns
    -------
    result : (N,)       N = n1 * n2
    grad   : (N, 2)
    hess   : (N, 2, 2)
    shape  : (n1, n2)
    """
    if y_bounds is None:
        y_bounds = (y[0], y[-1])
    y_min, y_max = y_bounds
    L = y_max - y_min                   # Period

    grids, shape = _meshgrid_flat(x, y)
    xg, yg = grids
    N = int(np.prod(shape))

    # --- 1D building blocks ---
    px  = _p(xg)
    dpx = _dp(xg)
    d2px = _d2p(xg)

    theta = 2.0 * np.pi * (yg - y_min) / L
    cy  =  np.cos(theta)
    sy  = -np.sin(theta)        # d/dy cos = -(2π/L) sin  → factor separately
    fac = 2.0 * np.pi / L

    # f = p(x) * cos(theta)
    result = px * cy

    # grad
    grad = np.zeros((N, 2))
    grad[:, 0] = dpx * cy                  # ∂f/∂x
    grad[:, 1] = px  * sy * fac            # ∂f/∂y

    # hess
    hess = np.zeros((N, 2, 2))
    hess[:, 0, 0] = d2px * cy                      # ∂²f/∂x²
    hess[:, 0, 1] = dpx  * sy * fac                # ∂²f/∂x∂y
    hess[:, 1, 0] = hess[:, 0, 1]                  # Symmetry
    hess[:, 1, 1] = -px  * cy * fac**2             # ∂²f/∂y²

    return result, grad, hess, shape


# ======================================================================= #
#  Self-test                                                               #
# ======================================================================= #

if __name__ == "__main__":

    # ---- Test function 1: 3D ----
    print("=== Test function 1 (3D) ===")
    x1 = np.linspace(0, 1, 4)
    x2 = np.linspace(0, 1, 5)
    x3 = np.linspace(0, 1, 3)
    result, grad, hess, shape = test_function_1(x1, x2, x3)
    print(f"shape  : {shape}  →  N = {result.shape[0]}")
    print(f"result : {result[:4]}  ...")
    print(f"grad[0]: {grad[0]}")
    print(f"hess[0]:\n{hess[0]}")

    # Gradient check (FD) for test function 1
    eps = 1e-6
    x1s = np.array([0.3]); x2s = np.array([0.5]); x3s = np.array([0.7])
    _, g, _, _ = test_function_1(x1s, x2s, x3s)
    f0, *_ = test_function_1(x1s, x2s, x3s)
    g_fd = []
    for k, xs in enumerate([x1s, x2s, x3s]):
        xp = xs + eps; xm = xs - eps
        args_p = [x1s, x2s, x3s]; args_p[k] = xp
        args_m = [x1s, x2s, x3s]; args_m[k] = xm
        fp, *_ = test_function_1(*args_p)
        fm, *_ = test_function_1(*args_m)
        g_fd.append((fp[0] - fm[0]) / (2*eps))
    print(f"\nGradient (analytical): {g[0]}")
    print(f"Gradient (FD):         {g_fd}")
    print(f"Max. error: {max(abs(a-b) for a,b in zip(g[0], g_fd)):.2e}")

    # ---- Test function 2: 2D ----
    print("\n=== Test function 2 (2D, periodic in y) ===")
    xv = np.linspace(0, 1, 20)
    # Support points cover [0, 2π); the period itself is 2π
    y_min, y_max = 0.0, 2.0 * np.pi
    yv = np.linspace(y_min, y_max, 28, endpoint=True)   # 28 points in [0, 2π)
    result2, grad2, hess2, shape2 = test_function_2(xv, yv, y_bounds=(y_min, y_max))
    print(f"shape  : {shape2}  →  N = {result2.shape[0]}")

    # Periodicity check: f(x, y_min) == f(x, y_max)
    xp = np.array([0.4])
    r_min, *_ = test_function_2(xp, np.array([y_min]), y_bounds=(y_min, y_max))
    r_max, *_ = test_function_2(xp, np.array([y_max]), y_bounds=(y_min, y_max))
    print(f"Periodicity f(y_min)={r_min[0]:.8f}, f(y_max)={r_max[0]:.8f}  "
          f"→ Difference: {abs(r_min[0]-r_max[0]):.2e}")

    # Gradient check FD for test function 2
    xs_ = np.array([0.3]); ys_ = np.array([1.2])
    ybounds = (y_min, y_max)
    _, g2, _, _ = test_function_2(xs_, ys_, y_bounds=ybounds)
    g2_fd = []
    for k in range(2):
        if k == 0:
            fp, *_ = test_function_2(xs_ + eps, ys_, y_bounds=ybounds)
            fm, *_ = test_function_2(xs_ - eps, ys_, y_bounds=ybounds)
        else:
            fp, *_ = test_function_2(xs_, ys_ + eps, y_bounds=ybounds)
            fm, *_ = test_function_2(xs_, ys_ - eps, y_bounds=ybounds)
        g2_fd.append((fp[0] - fm[0]) / (2*eps))
    print(f"Gradient (analytical): {g2[0]}")
    print(f"Gradient (FD):         {g2_fd}")
    print(f"Max. error: {max(abs(a-b) for a,b in zip(g2[0], g2_fd)):.2e}")

    # Plotting test_function_2
    # grid, _ = _meshgrid_flat(xv, yv)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(np.reshape(grid[0], (len(xv), len(yv))), np.reshape(grid[1], (len(xv), len(yv))), np.reshape(result2, (len(xv), len(yv))))
    # plt.show()
