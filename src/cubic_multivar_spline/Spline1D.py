import numpy as np
from typing import Tuple


SPLINE_BOUNDARY_CONDITIONS = ("not-a-knot", "periodic", "second_derivative", "first_derivative")

class Spline1D:

    def __init__(self, interval: Tuple[float, float], yv: np.ndarray, boundary_condition_type: Tuple[str, str] = ("second_derivative", "second_derivative"), boundary_condition_value: Tuple[float, float] = (0.0, 0.0) ) -> None:
        self._interval = interval
        self._yv = yv
        self._boundary_condition_type = boundary_condition_type
        self._boundary_condition_value = boundary_condition_value
        self._n = len(yv) - 1
        self._h = (interval[1]-interval[0])/(self._n)
        self._a = interval[0]
        self._b = interval[1]


    def _tridiag(self) -> np.ndarray:
        return np.diag(4*np.ones(self._n), 0) + np.diag(np.ones(self._n-1), 1) + np.diag(np.ones(self._n-1), -1)

if __name__ == '__main__':
    interval = (0.0, 1.0)
    yv = np.array([0.0, 1.0, 4.0, 9.0, 7.0])
    bc = ("second_derivative", "second_derivative")
    bc_value = (0.0, 0.0)
    spline = Spline1D(interval, yv, bc, bc_value)
    print(spline._tridiag())