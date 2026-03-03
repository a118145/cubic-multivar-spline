import numpy as np
import cubicmultispline as spl
import matplotlib.pyplot as plt
from matplotlib import cm

""" 
Preparing dummy data 
"""

# shape of the data -> number of points in each dimension 
shape = (10, 10, 10)
# ranges of the data -> [min, max] for each dimension       
ranges = [ 
    [0, 1],
    [0, 1],
    [0, 1],
    ]

# interval tuple for each dimension -> (min, max, number of points) for each dimension
interval = tuple((ranges[i][0], ranges[i][1], shape[i]) for i in range(len(shape)))

# boundary conditions for each dimension -> (first condition, second condition) for each dimension
bc = (
    ("first_derivative", "second_derivative"), 
    ("first_derivative", "second_derivative"), 
    ("first_derivative", "second_derivative"),
    )

# boundary conditions values for each dimension -> (first condition value, second condition value) for each dimension
# for "not-a-knot" and "periodic" boundary conditions, the values are not used
bc_value = (
    (1.0, 1.0), 
    (1.0, 1.0),
    (1.0, 1.0),
    )

# create dummy data and make periodic in y
dummy_data = np.random.randn(*shape).ravel()

spline_3d = spl.Spline(interval, dummy_data, bc, bc_value)

"""
Preparing plotting
"""

# Locations of dummy_data samples
x_sample = np.linspace(ranges[0][0], ranges[0][1], shape[0])
y_sample = np.linspace(ranges[1][0], ranges[1][1], shape[1])
z_sample = np.linspace(ranges[1][0], ranges[1][1], shape[1])
x_sample, y_sample, z_sample = np.meshgrid(x_sample, y_sample, z_sample, indexing='ij')

# Locations of spline evaluations for smooth surface
x_spline_eval = np.linspace(ranges[0][0], ranges[0][1], shape[0]*1)
y_spline_eval = np.linspace(ranges[1][0], ranges[1][1], shape[1]*1)
z_spline_eval = np.linspace(ranges[1][0], ranges[1][1], shape[1]*1)
x_spline_eval_grid, y_spline_eval_grid, z_spline_eval_grid = np.meshgrid(x_spline_eval, y_spline_eval, z_spline_eval, indexing='ij')

# Spline evaluation
coords = np.concatenate((x_spline_eval_grid.reshape((-1,1)), y_spline_eval_grid.reshape((-1,1)), z_spline_eval_grid.reshape((-1,1))), axis = 1)
vals, dvals, ddvals = spline_3d.eval_spline(coords)
vals = vals.reshape(x_spline_eval_grid.shape)

# Plot dummy data and spline surface
fig = plt.figure()
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex = 'none', sharey = 'none')
# ax1 = fig.add_subplot(221, projection='3d')
# # ax1.add_subplot(221, projection='3d')
# ax1.plot_surface(x_spline_eval_grid, y_spline_eval_grid, vals, antialiased = True, alpha = 0.8, cmap = cm.Blues)
# ax1.scatter(x_sample, y_sample, dummy_data.reshape(x_sample.shape), c = "red", marker = 'x')

# ax1.set_xlabel('x')
# ax1.set_ylabel('y')
# ax1.set_zlabel('data')

ax1 = fig.add_subplot(311, projection = '3d')
dx_vals = dvals[:, 0].reshape(x_spline_eval_grid.shape)
ddx_vals = ddvals[:, 0, 0].reshape(x_spline_eval_grid.shape)
ax1.plot_surface(y_spline_eval_grid[0, :, :], z_spline_eval_grid[0, :, :], dx_vals[0, :, :])
ax1.plot_surface(y_spline_eval_grid[0, :, :], z_spline_eval_grid[0, :, :], ddx_vals[-1, :, :])

ax2 = fig.add_subplot(312, projection = '3d')
dx_vals = dvals[:, 1].reshape(x_spline_eval_grid.shape)
ddx_vals = ddvals[:, 1, 1].reshape(x_spline_eval_grid.shape)
ax2.plot_surface(x_spline_eval_grid[:, 0, :], z_spline_eval_grid[:, 0, :], dx_vals[:, 0, :])
ax2.plot_surface(x_spline_eval_grid[:, 0, :], z_spline_eval_grid[:, 0, :], ddx_vals[:, -1, :])

# ax4 = fig.add_subplot(224)
# ax4.plot(x_spline_eval[::5], vals[::5, 0], 'o-')
# ax4.plot(x_spline_eval[::5], vals[::5, -1], 'x-')

plt.savefig("./tests/demo_pics/3d_spline_first-second.png")
plt.show()

