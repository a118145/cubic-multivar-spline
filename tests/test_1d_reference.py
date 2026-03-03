from test_functions import test_function_1 as test_function
import numpy as np
from cubicmultispline.Spline1D import Spline1D
from cubicmultispline.Spline import Spline
import matplotlib.pyplot as plt

"""
Test 1D spline interpolation
"""
# Preparation of test data
shape = (11,)
x = np.linspace(-1, 1, shape[0])
y, _, _, _ = test_function(x)

# Preparation of inputs for spline interpolation functions
interval = ((-1, 1, shape[0]),)
bc = (("not-a-knot", "not-a-knot"),)
bc_value = ((0.0, 0.0),)

# Native 1D interpolation
spline_1D_native = Spline1D(interval[0], y, bc[0], bc_value[0])
spline_1D = Spline(interval, y, bc, bc_value)

# Evaluate splines
x_eval = np.linspace(-1, 1, 100)
y_1D_native, dy_1D_native, ddy_1D_native, _ = spline_1D_native.eval_spline(x_eval)
y_1D, dy_1D, ddy_1D = spline_1D.eval_spline(x_eval)
y_ref, dy_ref, ddy_ref, shp = test_function(x_eval)

assert np.max(np.abs(y_1D_native - y_ref.ravel())) < 1e-13, f"Native 1D spline interpolation error: {np.max(np.abs(y_1D_native - y_ref.ravel()))} > 1e-13"
assert np.max(np.abs(y_1D - y_ref)) < 1e-13, f"Multivar spline interpolation error: {np.max(np.abs(y_1D - y_ref))} > 1e-13"
assert np.max(np.abs(dy_1D_native - dy_ref.ravel())) < 1e-13, f"Native 1D spline derivative error: {np.max(np.abs(dy_1D_native - dy_ref.ravel()))} > 1e-13"
assert np.max(np.abs(dy_1D - dy_ref)) < 1e-13, f"Multivar spline derivative error: {np.max(np.abs(dy_1D - dy_ref))} > 1e-13"
assert np.max(np.abs(ddy_1D_native - ddy_ref.ravel())) < 1e-13, f"Native 1D spline second derivative error: {np.max(np.abs(ddy_1D_native - ddy_ref.ravel()))} > 1e-13"
assert np.max(np.abs(ddy_1D - ddy_ref)) < 1e-13, f"Multivar spline second derivative error: {np.max(np.abs(ddy_1D - ddy_ref))} > 1e-13"

# Compare results
if __name__ == '__main__':
    print(f"Max error native spline interpolation: {np.max(np.abs(y_1D_native - y_ref.ravel()))}")
    print(f"Max error multivar spline interpolation: {np.max(np.abs(y_1D - y_ref))}")
    print(f"Max error native spline derivative: {np.max(np.abs(dy_1D_native - dy_ref.ravel()))}")
    print(f"Max error multivar spline derivative: {np.max(np.abs(dy_1D - dy_ref))}")
    print(f"Max error native spline second derivative: {np.max(np.abs(ddy_1D_native - ddy_ref.ravel()))}")
    print(f"Max error multivar spline second derivative: {np.max(np.abs(ddy_1D - ddy_ref))}")

    # Plot results
    plt.plot(x_eval, dy_1D_native, label="Native 1D spline", marker='o')
    plt.plot(x_eval, dy_1D, label="Multivar spline", marker='x')
    plt.plot(x_eval, dy_ref, label="Reference")
    plt.legend()
    plt.show()