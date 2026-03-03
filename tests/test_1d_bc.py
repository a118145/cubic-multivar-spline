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
bc_list = [
    (("not-a-knot", "not-a-knot"),),
    (("first_derivative", "second_derivative"),),
    (("periodic", "periodic"),),
]

bc_value_list = [
    ((0.0, 0.0),),
    ((1.0, -2.0),),
    ((0.0, 0.0),),
]

for i in range(len(bc_list)):
    bc = bc_list[i]
    bc_value = bc_value_list[i]
    # Native 1D interpolation
    if bc[0][0] == "periodic":
        # make periodic in order to work
        y[-1] = y[0]

    spline_1D_native = Spline1D(interval[0], y, bc[0], bc_value[0])
    spline_1D = Spline(interval, y, bc, bc_value)

    # Evaluate splines
    x_eval = np.linspace(-1, 1, 100)
    y_1D_native, dy_1D_native, ddy_1D_native, _ = spline_1D_native.eval_spline(x_eval)
    y_1D, dy_1D, ddy_1D = spline_1D.eval_spline(x_eval)

    assert np.max(np.abs(y_1D_native - y_1D.ravel())) < 1e-10, f"Max error spline interpolation (native - multivar): {np.max(np.abs(y_1D_native - y_1D.ravel()))} > 1e-10"
    assert np.max(np.abs(dy_1D_native - dy_1D.ravel())) < 1e-10, f"Max error spline derivative (native - multivar): {np.max(np.abs(dy_1D_native - dy_1D.ravel()))} > 1e-10"
    assert np.max(np.abs(ddy_1D_native - ddy_1D.ravel())) < 1e-10, f"Max error spline second derivative (native - multivar): {np.max(np.abs(ddy_1D_native - ddy_1D.ravel()))} > 1e-10"

    if i == 1:
        assert np.abs(dy_1D_native[0] - bc_value[0][0]) < 1e-13, f"Error first derivative at start (native): {np.abs(dy_1D_native[0] - bc_value[0][0])} > 1e-13"
        assert np.abs(dy_1D.ravel()[0] - bc_value[0][0]) < 1e-13, f"Error first derivative at start (multivar): {np.abs(dy_1D.ravel()[0] - bc_value[0][0])} > 1e-13"
        assert np.abs(ddy_1D_native[-1] - bc_value[0][1]) < 1e-13, f"Error second derivative at end (native): {np.abs(ddy_1D_native[-1] - bc_value[0][1])} > 1e-13"
        assert np.abs(ddy_1D.ravel()[-1] - bc_value[0][1]) < 1e-13, f"Error second derivative at end (multivar): {np.abs(ddy_1D.ravel()[-1] - bc_value[0][1])} > 1e-13"
    if i == 2:
        assert np.abs(dy_1D_native[0] - dy_1D_native[-1]) < 1e-13, f"Error first derivative at start and end (native): {np.abs(dy_1D_native[0] - dy_1D_native[-1])} > 1e-13"
        assert np.abs(dy_1D.ravel()[0] - dy_1D.ravel()[-1]) < 1e-13, f"Error first derivative at start and end (multivar): {np.abs(dy_1D.ravel()[0] - dy_1D.ravel()[-1])} > 1e-13"
        assert np.abs(ddy_1D_native[0] - ddy_1D_native[-1]) < 1e-13, f"Error second derivative at start and end (native): {np.abs(ddy_1D_native[0] - ddy_1D_native[-1])} > 1e-13"
        assert np.abs(ddy_1D.ravel()[0] - ddy_1D.ravel()[-1]) < 1e-13, f"Error second derivative at start and end (multivar): {np.abs(ddy_1D.ravel()[0] - ddy_1D.ravel()[-1])} > 1e-13"

    # Compare results
    if __name__ == '__main__':
        print(f"Test {bc[0]} with values {bc_value[0]}")
        print(f"Max error spline interpolation (native - multivar): {np.max(np.abs(y_1D_native - y_1D.ravel()))}")
        print(f"Max error spline derivative (native - multivar): {np.max(np.abs(dy_1D_native - dy_1D.ravel()))}")
        print(f"Max error spline second derivative (native - multivar): {np.max(np.abs(ddy_1D_native - ddy_1D.ravel()))}")
        
        if i == 1:
            print("Error first derivative at start (native):", np.abs(dy_1D_native[0] - bc_value[0][0]))
            print("Error first derivative at start (multivar):", np.abs(dy_1D.ravel()[0] - bc_value[0][0]))
            print("Error second derivative at end (native):", np.abs(ddy_1D_native[-1] - bc_value[0][1]))
            print("Error second derivative at end (multivar):", np.abs(ddy_1D.ravel()[-1] - bc_value[0][1]))
        if i == 2: 
            print("Error first derivative at start and end (native):", np.abs(dy_1D_native[0] - dy_1D_native[-1]))
            print("Error first derivative at start and end (multivar):", np.abs(dy_1D.ravel()[0] - dy_1D.ravel()[-1]))
            print("Error second derivative at start and end (native):", np.abs(ddy_1D_native[0] - ddy_1D_native[-1]))
            print("Error second derivative at start and end (multivar):", np.abs(ddy_1D.ravel()[0] - ddy_1D.ravel()[-1]))
        print("")

        # Plot results
        plt.figure()
        plt.subplot(311)
        plt.plot(x_eval, y_1D_native, label="Native 1D spline", marker='o')
        plt.plot(x_eval, y_1D, label="Multivar spline", marker='x')
        plt.legend()
        plt.subplot(312)
        plt.plot(x_eval, dy_1D_native, label="Native 1D spline", marker='o')
        plt.plot(x_eval, dy_1D, label="Multivar spline", marker='x')
        plt.legend()
        plt.subplot(313)
        plt.plot(x_eval, ddy_1D_native, label="Native 1D spline", marker='o')
        plt.plot(x_eval, ddy_1D.ravel(), label="Multivar spline", marker='x')
        plt.legend()
        plt.show()