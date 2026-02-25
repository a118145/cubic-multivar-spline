import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from cubic_multivar_spline.Spline1D import Spline1D

class Spline2D:

    def __init__(
        self, 
        interval: Tuple[Tuple[float, float, int]], 
        yv: np.ndarray, 
        boundary_condition_type: Tuple[Tuple[str, str]] = None, 
        boundary_condition_value: Tuple[Tuple[float, float]] = None 
        ) -> None:

        num_vals = 1
        for i in range(len(interval)):
            if len(interval[i]) != 3:
                raise ValueError(f"Interval {i} must be a tuple of tuples of length 3.")
            num_vals *= interval[i][2]
        if num_vals != len(yv):
            raise ValueError(f"Number of sample values {len(yv)} does not match number of sample points defined by product of third value inside intervals.")

        self._ndim = len(interval)
        self._interval = interval
        self._yv = yv
        if boundary_condition_type is None:
            boundary_condition_type = tuple(("not-a-knot", "not-a-knot") for _ in range(self._ndim))
        if boundary_condition_value is None:
            boundary_condition_value = tuple((0.0, 0.0) for _ in range(self._ndim))
        self._boundary_condition_type = boundary_condition_type
        self._boundary_condition_value = boundary_condition_value
        # print()
        _tmp_coeff = Spline2D.recursive_spline(
            interval = self._interval, 
            yv = self._yv, 
            boundary_condition_type = self._boundary_condition_type, 
            boundary_condition_value = self._boundary_condition_value)
        _coeff_shape = tuple([inter[2] + 2 for inter in interval])
        self._coeff = np.reshape(_tmp_coeff, _coeff_shape)
        print(self._coeff)

    @staticmethod
    def recursive_spline_test(
        interval: Tuple[Tuple[float, float, int]], 
        yv: np.ndarray, 
        boundary_condition_type: Tuple[Tuple[str, str]], 
        boundary_condition_value: Tuple[Tuple[float, float]]
        ) -> np.ndarray:
        ci_star = np.zeros((13,11))
        for i in range(interval[0][2]):
            tmp_spline = Spline1D(interval[1], yv[i*interval[1][2]:(i+1)*interval[1][2]], boundary_condition_type[1], boundary_condition_value[1])
            # print(len(tmp_spline.coeff))
            ci_star[:,i] = tmp_spline.coeff
        ci = np.zeros((13,13))
        for i in range(np.size(ci_star, 0)):
            tmp_spline = Spline1D(interval[0], ci_star[i,:], boundary_condition_type[0], boundary_condition_value[0])
            ci[i,:] = tmp_spline.coeff
        return ci

    @staticmethod
    def recursive_spline(
        interval: Tuple[Tuple[float, float, int]], 
        yv: np.ndarray, 
        boundary_condition_type: Tuple[Tuple[str, str]], 
        boundary_condition_value: Tuple[Tuple[float, float]]
        ) -> np.ndarray:
        print(interval)
        if len(interval) is 1:
            print("1D spline")
            # lowest level of recursion reached -> 1D spline
            return Spline1D(interval[0], yv, boundary_condition_type[0], boundary_condition_value[0]).coeff
        
        ci_star_size = 1
        ci_star_len = 1
        ci_size = 1
        ci_len = 1
        for i in range(len(interval)):
            if i != 0:
                ci_star_size *= (interval[i][2] + 2)
                ci_star_len *= (interval[i][2] + 2)
                ci_len *= (interval[i][2] + 2)
            else:
                ci_star_size *= interval[i][2]
            ci_size *= (interval[i][2] + 2)
        ci_star = np.zeros(ci_star_size)
        for i in range(interval[0][2]):
            ci_star[i*ci_star_len:(i+1)*ci_star_len] = Spline2D.recursive_spline(
                interval[1:], 
                yv[i*interval[0][2]:(i+1)*interval[0][2]],
                boundary_condition_type[1:], 
                boundary_condition_value[1:]
            )
        ci = np.zeros(ci_size)
        for i in range(ci_size//ci_star_len):
            tmp_spline = Spline1D(interval[0], ci_star[i::ci_star_len], boundary_condition_type[0], boundary_condition_value[0])
            ci[i*ci_len:(i+1)*ci_len] = tmp_spline.coeff
        return ci
        
    @property
    def coeff(self) -> np.ndarray:
        return self._coeff

    def eval_spline(self, x: float, y: float) -> Tuple[float, float, float, float]:
        hi = (self._interval[0][1] - self._interval[0][0]) / (self._interval[0][2]-1)
        hj = (self._interval[1][1] - self._interval[1][0]) / (self._interval[1][2]-1)
        res = 0
        for i in range(13):
            for j in range(13):
                ti = (x - self._interval[0][0]) / hi - i + 1
                tj = (y - self._interval[1][0]) / hj - j + 1
                res += self._coeff[i, j] * Spline1D._phi(ti) * Spline1D._phi(tj)
        return res





if __name__ == "__main__":

    def test_function(x: np.array, y: np.array) -> np.array:
        tf = np.zeros(len(x)*len(y))
        coords = np.zeros((len(x)*len(y),2), dtype=float)
        for i in range(len(x)):
            for j in range(len(y)):
                pointer = len(y)*i + j
                tf[pointer] = 2*x[j]**2 + y[i]**3
                coords[pointer] = np.array([x[i], y[j]])
        return tf, coords

    n = 11 # number of points in each direction
    xvals = np.linspace(0, 1, n)
    yvals = np.linspace(0, 1, n)
    xgrid, ygrid = np.meshgrid(xvals, yvals)
    zvals, coords = test_function(xvals, yvals)
    zgrid = np.reshape(zvals, (n,n))

    # print(len(test_function(np.linspace(0, 1, n), np.linspace(0, 1, n))))

    
    spline = Spline2D(
        interval=((0, 1, n), (0, 1, n)),
        yv=zvals,
        boundary_condition_type=(("first_derivative", "first_derivative"), ("not-a-knot", "not-a-knot")),
        boundary_condition_value=((0.0, 0.0), (0.0, 0.0))
    )
    # print(spline.coeff)
    print(spline.eval_spline(0.5, 0.59))
    print(test_function(np.array([0.5]), np.array([0.59])))
    # print(coords)
    n_spline_eval = 10
    x_spline_eval = np.linspace(0, 1, n_spline_eval)
    y_spline_eval = np.linspace(0, 1, n_spline_eval)
    x_spline_eval_grid, y_spline_eval_grid = np.meshgrid(x_spline_eval, y_spline_eval)
    z_spline_eval = np.zeros((n_spline_eval, n_spline_eval))
    for i in range(n_spline_eval):
        for j in range(n_spline_eval):
            z_spline_eval[j,i] = spline.eval_spline(x_spline_eval[i], y_spline_eval[j])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xgrid, ygrid, zgrid)
    ax.plot_surface(x_spline_eval_grid, y_spline_eval_grid, z_spline_eval)
    plt.show()