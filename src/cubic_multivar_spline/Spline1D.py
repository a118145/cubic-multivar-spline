import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt


SPLINE_BOUNDARY_CONDITIONS = ("not-a-knot", "periodic", "second_derivative", "first_derivative")

class Spline1D:

    def __init__(self, interval: Tuple[float, float], yv: np.ndarray, boundary_condition_type: Tuple[str, str] = ("second_derivative", "second_derivative"), boundary_condition_value: Tuple[float, float] = (0.0, 0.0) ) -> None:
        self._interval = interval
        self._yv = yv
        if "periodic" in boundary_condition_type:
            self._boundary_condition_type = ("periodic", "periodic")
        else:
            self._boundary_condition_type = boundary_condition_type
        self._boundary_condition_value = boundary_condition_value
        self._n = len(yv) - 1
        self._h = (interval[1]-interval[0])/(self._n)
        self._a = interval[0]
        self._b = interval[1]

        self._matrix = self._base_diag()
        self._rhs = np.zeros(self._n+3)
        self._rhs[:-2] = self._yv
        self._add_bc_to_matrix_rhs()

        self._coeff = np.linalg.solve(self._matrix, self._rhs)


    def _tridiag(self) -> np.ndarray:
        return np.diag(4*np.ones(self._n), 0) + np.diag(np.ones(self._n-1), 1) + np.diag(np.ones(self._n-1), -1)

    
    def _base_diag(self) -> np.ndarray:
        tmp_diag = np.diag(np.ones(self._n+3), 0) + np.diag(4*np.ones(self._n+2), 1) + np.diag(np.ones(self._n+1), 2)
        if self._boundary_condition_type[0] == "periodic":
            tmp_diag[-3:, :] = 0
        else:
            tmp_diag[-2:, :] = 0
        return tmp_diag

    def _add_bc_to_matrix_rhs(self) -> None:
        if self._boundary_condition_type[0] == "periodic":
            self._matrix[-3, 0] = 1
            self._matrix[-3, -3] = -1
            self._matrix[-2, 1] = 1
            self._matrix[-2, -2] = -1
            self._matrix[-1, 2] = 1
            self._matrix[-1, -1] = -1
        else:
            for i, val in enumerate(self._boundary_condition_type): 
                if val == "first_derivative":
                    col = i * self._n
                    self._matrix[-2+i, col:col+3] = np.array([-1, 0, 1])
                    self._rhs[-2+i] = self._boundary_condition_value[i]*self._h/3
                if val == "second_derivative":
                    # print(i)
                    # print(self._matrix[-2, -3:])
                    col = i * self._n
                    self._matrix[-2+i, col:col+3] = np.array([1, -2, 1])
                    self._rhs[-2+i] = self._boundary_condition_value[i]*self._h**2/6
                if val == "not-a-knot":
                    col = i * (self._n-2)
                    self._matrix[-2+i, col:col+5] = np.array([1, -4, 6, -4, 1])
                    
    @staticmethod
    def _phi(t: float | np.ndarray) -> float | np.ndarray:
        t_abs = np.abs(t)
        result = np.zeros_like(t, dtype=float)
        
        # Case 1: 1 < |t| < 2
        mask1 = (t_abs > 1) & (t_abs <= 2)
        result[mask1] = (2 - t_abs[mask1])**3
        
        # Case 2: |t| < 1
        mask2 = t_abs <= 1
        result[mask2] = 4 - 6 * t_abs[mask2]**2 + 3 * t_abs[mask2]**3
        
        # Case 3: otherwise (already 0 from initialization)
        
        return result if isinstance(t, np.ndarray) else float(result)
    
    @staticmethod
    def _dphi_dt(t: float | np.ndarray) -> float | np.ndarray:
        t_abs = np.abs(t)
        t_sign = np.sign(t)
        result = np.zeros_like(t, dtype=float)
        
        # Case 1: 1 < |t| < 2
        mask1 = (t_abs > 1) & (t_abs <= 2)
        result[mask1] = -t_sign[mask1]*3*(2 - t_abs[mask1])**2
        
        # Case 2: |t| < 1
        mask2 = t_abs <= 1
        result[mask2] =  - t_sign[mask2]*12 * t_abs[mask2] + t_sign[mask2]*9 * t_abs[mask2]**2
        
        # Case 3: otherwise (already 0 from initialization)
        
        return result if isinstance(t, np.ndarray) else float(result)
    
    @staticmethod
    def _d2phi_dt2(t: float | np.ndarray) -> float | np.ndarray:
        t_abs = np.abs(t)
        result = np.zeros_like(t, dtype=float)
        
        # Case 1: 1 < |t| < 2
        mask1 = (t_abs > 1) & (t_abs <= 2)
        result[mask1] = 6*(2 - t_abs[mask1])
        
        # Case 2: |t| < 1
        mask2 = t_abs <= 1
        result[mask2] =  - 12 + 18 * t_abs[mask2]
        
        # Case 3: otherwise (already 0 from initialization)
        
        return result if isinstance(t, np.ndarray) else float(result)
    
    @staticmethod
    def _d3phi_dt3(t: float | np.ndarray) -> float | np.ndarray:
        t_abs = np.abs(t)
        t_sign = np.sign(t)
        result = np.zeros_like(t, dtype=float)
        
        # Case 1: 1 < |t| < 2
        mask1 = (t_abs > 1) & (t_abs <= 2)
        result[mask1] = -6*t_sign[mask1]
        
        # Case 2: |t| < 1
        mask2 = t_abs <= 1
        result[mask2] =  t_sign[mask2]* 18 
        
        # Case 3: otherwise (already 0 from initialization)
        
        return result if isinstance(t, np.ndarray) else float(result)

    def eval_spline_old(self, x: float | np.ndarray) -> Tuple[float, float, float, float] | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(x)
        l = np.floor((x-self._a)/self._h).astype(int)
        # print('l', l)
        m = np.min([l+4, self._n+3])
        # print('m', m)
        t = ((x-self._a)/self._h-(np.arange(l+1,m+1)-2))
        # print('t', t)
        return np.sum(self._coeff[l:m]*Spline1D._phi(t)), np.sum(self._coeff[l:m]*Spline1D._dphi_dt(t)/self._h), np.sum(self._coeff[l:m]*Spline1D._d2phi_dt2(t)/self._h**2), np.sum(self._coeff[l:m]*Spline1D._d3phi_dt3(t)/self._h**3)

    def eval_spline(self, x: float | np.ndarray) -> Tuple[float, float, float, float] | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = np.atleast_1d(x)
        l = np.floor((x-self._a)/self._h).astype(int)
        
        # Initialize results arrays
        spline_val = np.zeros_like(x, dtype=float)
        spline_d1 = np.zeros_like(x, dtype=float)
        spline_d2 = np.zeros_like(x, dtype=float)
        spline_d3 = np.zeros_like(x, dtype=float)
        
        # Process each possible basis function index
        for j in range(4):  # At most 4 basis functions contribute
            # Get the coefficient index for each point
            coeff_idx = l + j
            # Only process valid indices
            valid_mask = (coeff_idx >= 0) & (coeff_idx <= self._n + 2)
            
            if np.any(valid_mask):
                # Compute t values for this basis function
                t_vals = ((x[valid_mask]-self._a)/self._h - coeff_idx[valid_mask] + 1)
                
                # Get coefficients
                coeffs = self._coeff[coeff_idx[valid_mask]]
                
                # Add contributions
                spline_val[valid_mask] += coeffs * Spline1D._phi(t_vals)
                spline_d1[valid_mask] += coeffs * Spline1D._dphi_dt(t_vals) / self._h
                spline_d2[valid_mask] += coeffs * Spline1D._d2phi_dt2(t_vals) / self._h**2
                spline_d3[valid_mask] += coeffs * Spline1D._d3phi_dt3(t_vals) / self._h**3
        
        # Return appropriate type based on input
        if len(x) == 1:
            return float(spline_val[0]), float(spline_d1[0]), float(spline_d2[0]), float(spline_d3[0])
        else:
            return spline_val, spline_d1, spline_d2, spline_d3

    @property
    def coeff(self):
        return self._coeff

if __name__ == '__main__':
    interval = (0.0, 1.0)
    yv = np.array([0.0, 1.0, 4.0, 9.0, 0.0])
    bc = ("periodic", "periodic")
    bc_value = (10.0, 0.0)
    spline = Spline1D(interval, yv, bc, bc_value)
    print(spline._tridiag())
    print(spline._matrix)
    print(spline._rhs)
    print(spline._coeff)

    step = 0.01
    xv = np.arange(0, 1+step/2, step)
    # yvals = np.zeros(len(xv))
    # dyvals = np.zeros(len(xv))
    # ddyvals = np.zeros(len(xv))
    # dddyvals = np.zeros(len(xv))
    # for i in range(len(xv)):
    #     # print('################')
    #     # print(i)
    #     # print(spline.eval_spline(xv[i]))
    #     yvals[i], dyvals[i], ddyvals[i], dddyvals[i] = spline.eval_spline_old(xv[i])
    yvals_new, dyvals_new, ddyvals_new, dddyvals_new = spline.eval_spline(xv)

    plt.plot(xv, yvals_new, 'x-')
    # plt.plot(xv, dyvals_new, 'o-')
    # plt.plot(xv, yvals_new, 'd-')
    # plt.plot(xv, dyvals_new, '<-')
    # plt.plot(xv, ddyvals, '.-')
    # plt.plot(xv, dddyvals, 'd-')
    # plt.plot(xv+1, yvals, 'x-')
    plt.show()