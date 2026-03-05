import numpy as np
import cubicmultispline as spl
import matplotlib.pyplot as plt
from matplotlib import cm

""" 
Preparing dummy data 
"""

#1s
# shape of the data -> number of points in each dimension 
shape = (11, 11, 11)
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
    (-1.0, 1.0), 
    (1.0, 2.0),
    (-1.0, -3.0),
    )
#1e

# create dummy data and make periodic in y
dummy_data_orig = np.random.rand(*shape)
dummy_data = dummy_data_orig.ravel()

spline_3d = spl.Spline(interval, dummy_data, bc, bc_value)

"""
Preparing plotting
"""

# Locations of dummy_data samples
x_sample = np.linspace(ranges[0][0], ranges[0][1], shape[0])
y_sample = np.linspace(ranges[1][0], ranges[1][1], shape[1])
z_sample = np.linspace(ranges[2][0], ranges[2][1], shape[2])
x_sample, y_sample, z_sample = np.meshgrid(x_sample, y_sample, z_sample, indexing='ij')

# Locations of spline evaluations for smooth surface
x_spline_eval = np.linspace(ranges[0][0], ranges[0][1], (shape[0]-1)*10+1)
y_spline_eval = np.linspace(ranges[1][0], ranges[1][1], (shape[1]-1)*10+1)
z_spline_eval = np.linspace(ranges[2][0], ranges[2][1], (shape[2]-1)*10+1)
x_spline_eval_grid, y_spline_eval_grid, z_spline_eval_grid = np.meshgrid(x_spline_eval, y_spline_eval, z_spline_eval, indexing='ij')

# Spline evaluation
coords = np.concatenate((x_spline_eval_grid.reshape((-1,1)), y_spline_eval_grid.reshape((-1,1)), z_spline_eval_grid.reshape((-1,1))), axis = 1)
vals, dvals, ddvals = spline_3d.eval_spline(coords)
vals = vals.reshape(x_spline_eval_grid.shape)

# Plot dummy data and spline surface
fig = plt.figure(figsize=(5,5), dpi = 300)
plt.subplots_adjust(hspace=0.6, wspace=0.6)

ax1 = fig.add_subplot(221, projection = '3d')
#2s
d_vals = dvals[:, 0].reshape(x_spline_eval_grid.shape)
dd_vals = ddvals[:, 0, 0].reshape(x_spline_eval_grid.shape)
ax1.plot_surface(
    y_spline_eval_grid[0, :, :], # all y values at x = 0 (first index = 0)
    z_spline_eval_grid[0, :, :], # all z values at x = 0 (first index = 0)
    d_vals[0, :, :],             # first order partial derivative w.r.t. x at x = 0
    color = "blue")
ax1.plot_surface(
    y_spline_eval_grid[-1, :, :], # all y values at x = 1 (first index = -1)
    z_spline_eval_grid[-1, :, :], # all z values at x = 1 (first index = -1)
    dd_vals[-1, :, :],            # second order partial derivative w.r.t. x at x = 1 
    color = "red")
#2e
ax1.set_xlabel("y")
ax1.set_ylabel("z")
ax1.set_zlabel("grad[0], hess[0,0]")
ax1.set_title("x boundaries")
ax1.view_init(elev=30, azim = 220)

ax2 = fig.add_subplot(222, projection = '3d')
d_vals = dvals[:, 1].reshape(x_spline_eval_grid.shape)
dd_vals = ddvals[:, 1, 1].reshape(x_spline_eval_grid.shape)
ax2.plot_surface(x_spline_eval_grid[:, 0, :], z_spline_eval_grid[:, 0, :], d_vals[:, 0, :], color = "blue")
ax2.plot_surface(x_spline_eval_grid[:, -1, :], z_spline_eval_grid[:, -1, :], dd_vals[:, -1, :], color = "red")
ax2.set_xlabel("x")
ax2.set_ylabel("z")
ax2.set_zlabel("grad[1], hess[1,1]")
ax2.set_title("y boundaries")
ax2.view_init(elev=30, azim = 220)

ax3 = fig.add_subplot(223, projection = '3d')
d_vals = dvals[:, 2].reshape(x_spline_eval_grid.shape)
dd_vals = ddvals[:, 2, 2].reshape(x_spline_eval_grid.shape)
ax3.plot_surface(x_spline_eval_grid[:, :, 0], y_spline_eval_grid[:, :, 0], d_vals[:, :, 0], color = "blue")
ax3.plot_surface(x_spline_eval_grid[:, :, -1], y_spline_eval_grid[:, :, -1], dd_vals[:, :, -1], color = "red")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_zlabel("grad[2], hess[2,2]")
ax3.set_title("z boundaries")
ax3.view_init(elev=30, azim = 220)
plt.savefig("./docs/_static/demo_pics/3d_spline_first-second.png")
# plt.show()

step = 1
cnt = 0
for i in range(101):
    print(f"Plotting {i}")
    fig2 = plt.figure(figsize=(5,4), dpi = 300)
    ax = fig2.add_subplot(111, projection = '3d')
    ax.view_init(elev=30, azim = 220)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("val ∈ [0, 1)")
    ax.set_title(f"Slice at z = {i/100:.2f}")
    ax.set_zlim(-2,3)
    ax.plot_surface(x_spline_eval_grid[:, :, i*step], y_spline_eval_grid[:, :, i*step], vals[:, :, i*step], cmap = "Blues") 
    #3s
    if not i % 10:
        assert(np.allclose(vals[::10, ::10, i*step], dummy_data_orig[:,:,cnt]))
        cnt += 1
    #3e
    # plt.show()
    if i == 0 and False:
        plt.axis('off')
        plt.savefig(f"./docs/_static/3d_spline_first-second_logo.png", transparent=True)
        plt.axis('on')
    # plt.savefig(f"./docs/_static/demo_gifs/3d_spline_first-second_{i}.png", transparent=False)
    plt.close()


