#1s
import numpy as np
import cubicmultispline as spl
import matplotlib.pyplot as plt
from matplotlib import cm

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
bc = (
    ("not-a-knot", "not-a-knot"), 
    ("not-a-knot", "not-a-knot"),
    )

# boundary conditions values for each dimension -> (first condition value, second condition value) for each dimension
# for "not-a-knot" and "periodic" boundary conditions, the values are not used
bc_value = (
    (0.0, 0.0), 
    (0.0, 0.0),
    )

# generation of random dummy data
dummy_data = np.random.randn(*shape).ravel()
#1e
#2s
# generation of spline surface
spline_2d = spl.Spline(interval, dummy_data, bc, bc_value)
#2e
#3s
"""
Preparing plotting
"""

# Locations of dummy_data samples
x_sample = np.linspace(ranges[0][0], ranges[0][1], shape[0])
y_sample = np.linspace(ranges[1][0], ranges[1][1], shape[1])
x_sample, y_sample = np.meshgrid(x_sample, y_sample, indexing='ij')
#3e
#4s
# Locations of spline evaluations for smooth surface
x_spline_eval = np.linspace(ranges[0][0], ranges[0][1], shape[0]*50)
y_spline_eval = np.linspace(ranges[1][0], ranges[1][1], shape[1]*50)
x_spline_eval, y_spline_eval = np.meshgrid(x_spline_eval, y_spline_eval, indexing='ij')
#4e
#5s
# Spline evaluation
coords = np.concatenate((
    x_spline_eval.reshape((-1,1)), 
    y_spline_eval.reshape((-1,1))
    ), axis = 1)
vals, dvals, ddvals = spline_2d.eval_spline(coords)
vals = vals.reshape(x_spline_eval.shape)
#5e
#6s
# Plot dummy data and spline surface
fig = plt.figure()
plt.subplots_adjust(hspace=0.5, wspace=0.4)
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(x_spline_eval, y_spline_eval, vals, antialiased = True, alpha = 0.8, cmap = cm.Blues)
ax.scatter(x_sample, y_sample, dummy_data.reshape(x_sample.shape), c = "red", marker = 'x')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('data')
#6e
plt.savefig("./docs/_static/demo_pics/2d_spline_not-a-knot.png")
plt.show()

