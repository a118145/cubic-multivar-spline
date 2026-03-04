import numpy as np
import cubicmultispline as spl
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

mpl.rcParams['axes.formatter.useoffset'] = False

""" 
Preparing dummy data 
"""

# shape of the data -> number of points in each dimension 
shape = (11, 9)
# ranges of the data -> [min, max] for each dimension       
ranges = [ 
    [0, 1],
    [0, 0.8],
    ]

# interval tuple for each dimension -> (min, max, number of points) for each dimension
interval = tuple((ranges[i][0], ranges[i][1], shape[i]) for i in range(len(shape)))

# boundary conditions for each dimension -> (first condition, second condition) for each dimension
#1s
bc = (
    ("first_derivative", "second_derivative"), 
    ("periodic", "periodic"),
    )
#1e
# boundary conditions values for each dimension -> (first condition value, second condition value) for each dimension
# for "not-a-knot" and "periodic" boundary conditions, the values are not used
#2s
bc_value = (
    (1.0, -2.0), 
    (0.0, 0.0),
    )
#2e

# create dummy data and make periodic in y
dummy_data = np.random.randn(*shape)
#3s
dummy_data[:, 0] = dummy_data[:, -1]
dummy_data = dummy_data.ravel()
#3e

spline_2d = spl.Spline(interval, dummy_data, bc, bc_value)

"""
Preparing plotting
"""

# Locations of dummy_data samples
x_sample = np.linspace(ranges[0][0], ranges[0][1], shape[0])
y_sample = np.linspace(ranges[1][0], ranges[1][1], shape[1])
x_sample, y_sample = np.meshgrid(x_sample, y_sample, indexing='ij')

# Locations of spline evaluations for smooth surface
x_spline_eval = np.linspace(ranges[0][0], ranges[0][1], shape[0]*50)
y_spline_eval = np.linspace(ranges[1][0], ranges[1][1], shape[1]*50)
x_spline_eval_grid, y_spline_eval_grid = np.meshgrid(x_spline_eval, y_spline_eval, indexing='ij')

# Spline evaluation
coords = np.concatenate((x_spline_eval_grid.reshape((-1,1)), y_spline_eval_grid.reshape((-1,1))), axis = 1)
vals, dvals, ddvals = spline_2d.eval_spline(coords)
vals = vals.reshape(x_spline_eval_grid.shape)

# Plot dummy data and spline surface
fig = plt.figure()
plt.subplots_adjust(hspace=0.5, wspace=0.4)
ax1 = fig.add_subplot(221, projection='3d')
ax1.plot_surface(x_spline_eval_grid, y_spline_eval_grid, vals, antialiased = True, alpha = 0.8, cmap = cm.Blues)
ax1.scatter(x_sample, y_sample, dummy_data.reshape(x_sample.shape), c = "red", marker = 'x')

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('data')
ax1.set_title("3D view of spline surface")
ax1.view_init(elev=30, azim = 40)

ax2 = fig.add_subplot(222)
ax2.set_ylim(0.5, 1.5)
dx_vals = dvals[:, 0].reshape(x_spline_eval_grid.shape)
ax2.plot(y_spline_eval, dx_vals[0, :])
ax2.set_xlabel('y')
ax2.set_ylabel('grad[0]')
ax2.set_title("imposed BC at x = 0")

ax3 = fig.add_subplot(223)
ax3.set_ylim(-2.5, -1.5)
ddx_vals = ddvals[:, 0, 0].reshape(x_spline_eval_grid.shape)
ax3.plot(y_spline_eval, ddx_vals[-1, :])
ax3.set_xlabel("y")
ax3.set_ylabel('hess[0,0]')
ax3.set_title("imposed BC at x = 1")

ax4 = fig.add_subplot(224)
ax4.plot(x_spline_eval[::5], vals[::5, 0], 'o-')    
ax4.plot(x_spline_eval[::5], vals[::5, -1], 'x-')
ax4.set_xlabel("x")
ax4.set_ylabel('value')
ax4.set_title("periodic edges")

#4s
# checking first and second derivative along periodic edges
dy_vals = dvals[:, 1].reshape(x_spline_eval_grid.shape)
ddy_vals = ddvals[:, 1, 1].reshape(x_spline_eval_grid.shape)

assert np.allclose(dy_vals[:, 0], dy_vals[:, -1])
assert np.allclose(ddy_vals[:, 0], ddy_vals[:, -1])
#4e
plt.savefig("./docs/_static/demo_pics/2d_spline_first-second-peri.png")
# plt.show()

