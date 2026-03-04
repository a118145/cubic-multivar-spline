Boundary conditions
======================


The library provides two additional examples in the ``tests`` directory:

1. ``demo_first-second-peri.py``: 2D spline with clamped and natural boundary conditions in the x-direction and periodicity in the y-direction
2. ``demo_first-second-3d.py``: 3D spline with first and second order boundary conditions

In the following, the 2-dimensional case is discussed in detail. The data is prepared in the same manner as before. However, the boundary conditions are different. In this case, we use clamped and natural boundary conditions in the x-direction and periodicity in the y-direction: 

```python
bc = (
    ("first_derivative", "second_derivative"), 
    ("periodic", "periodic"),
    )
```

The following values are used:

```python
bc_value = (
    (1.0, -2.0), 
    (0.0, 0.0),
    )
```

Note, that in case of a periodic boundary constraint, both edges of the domain are periodic. This is checked inside `Spline1D` and corrected if necessary, i.e., all values inside the tuple of the corresponding dimension are set to `"periodic"`. The boundary condition values are not of any meaning in this case.

In addition to the differing boundary conditions, the dummy data has to be periodic in the dimension where periodicity is imposed. This is done by setting the first and last value of the dummy data to be equal:

```python
dummy_data[:, 0] = dummy_data[:, -1]
```

The :code:`Spline1D` class raises an error if the dummy data is not periodic. The resulting spline surface should look similar to the following:
![2d_spline_first-second-peri](./tests/demo_pics/2d_spline_first-second-peri.png)

In addition to the 3-dimensional view of the spline surface, the boundary conditions are checked by inspecting the gradient and hessian at the edges of the domain. The partial derivative w.r.t. the x-axis at the left edge should be equal to the first boundary condition value, and the second order partial derivative w.r.t. the x-axis (twice) at the right edge should be equal to the second boundary condition value. This is indeed the case. The periodic edges are inspected by means of the values at the left and right edge of the domain - they are equal as required by periodicity. The first and second derivative along the y-direction (normal to the periodic edges) are checked separately: 

```python
# checking first and second derivative along periodic edges
dy_vals = dvals[:, 1].reshape(x_spline_eval_grid.shape)
ddy_vals = ddvals[:, 1, 1].reshape(x_spline_eval_grid.shape)

assert np.allclose(dy_vals[:, 0], dy_vals[:, -1])
assert np.allclose(ddy_vals[:, 0], ddy_vals[:, -1]) 
```
