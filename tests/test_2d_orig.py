import numpy as np
import matplotlib.pyplot as plt
import cubicmultispline as spl

def test_function(x: np.array, y: np.array) -> np.array:
    tf = np.zeros(len(x)*len(y))
    coords = np.zeros((len(x)*len(y),2), dtype=float)
    for i in range(len(x)):
        for j in range(len(y)):
                pointer = j + i * len(y)
                tf[pointer] = 2*x[i]**2 + y[j]**3
                coords[pointer] = np.array([x[i], y[j]])
    return tf, coords

shape = (11, 11)
xvals = np.linspace(0, 1, shape[0])
yvals = np.linspace(0, 1, shape[1])
xgrid, ygrid = np.meshgrid(xvals, yvals, indexing='ij')
# zgrid = np.reshape(zvals, (n,n+1))
test_func, coords = test_function(xvals, yvals)
test_func_reshaped = np.reshape(test_func, shape)


spline_2d = spl.Spline(
        interval=((0, 1, shape[0]), (0, 1, shape[1])),
        yv=test_func,
        boundary_condition_type=(("first_derivative", "first_derivative"), ("not-a-knot", "not-a-knot")),
        boundary_condition_value=((0.0, 0.0), (0.0, 0.0))
    )


n_spline_eval = 20
x_spline_eval = np.linspace(0, 1, n_spline_eval)
y_spline_eval = np.linspace(0, 1, n_spline_eval)
x_spline_eval_grid, y_spline_eval_grid = np.meshgrid(x_spline_eval, y_spline_eval, indexing='ij')
z_spline_eval = np.zeros((n_spline_eval, n_spline_eval))
for i in range(n_spline_eval):
    for j in range(n_spline_eval):
        z_spline_eval[i,j] = spline_2d.eval_spline_old(x_spline_eval[i], y_spline_eval[j])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xgrid, ygrid, test_func_reshaped)
ax.plot_surface(x_spline_eval_grid, y_spline_eval_grid, z_spline_eval)
plt.show()