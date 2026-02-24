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
        self._coeff = Spline2D.recursive_spline(
            interval = self._interval, 
            yv = self._yv, 
            boundary_condition_type = self._boundary_condition_type, 
            boundary_condition_value = self._boundary_condition_value)

    @staticmethod
    def recursive_spline(
        interval: Tuple[Tuple[float, float, int]], 
        yv: np.ndarray, 
        boundary_condition_type: Tuple[Tuple[str, str]], 
        boundary_condition_value: Tuple[Tuple[float, float]]
        ) -> np.ndarray:
        ci_star = np.zeros((13,11))
        for i in range(interval[1][2]):
            tmp_spline = Spline1D(interval[0], yv[i::interval[0][2]], boundary_condition_type[0], boundary_condition_value[0])
            # print(len(tmp_spline.coeff))
            ci_star[:,i] = tmp_spline.coeff
        ci = np.zeros((13,13))
        for i in range(np.size(ci_star, 0)):
            tmp_spline = Spline1D(interval[1], ci_star[i,:], boundary_condition_type[1], boundary_condition_value[1])
            ci[i,:] = tmp_spline.coeff
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
        for i in range(len(x)):
            for j in range(len(y)):
                pointer = len(x)*i + j
                tf[pointer] = x[i]**2 + y[j]**3
        return tf

    n = 11 # number of points in each direction

    print(len(test_function(np.linspace(0, 1, n), np.linspace(0, 1, n))))

    
    spline = Spline2D(
        interval=((0, 1, n), (0, 1, n)),
        yv=test_function(np.linspace(0, 1, n), np.linspace(0, 1, n)),
        boundary_condition_type=None,
        boundary_condition_value=None
    )
    print(spline.coeff)
    print(spline.eval_spline(0.5, 0.59))
    print(test_function(np.array([0.5]), np.array([0.59])))