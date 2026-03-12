import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt


SPLINE_BOUNDARY_CONDITIONS = ("not-a-knot", "periodic", "second_derivative", "first_derivative")
"""
Supported boundary conditions for cubic spline interpolation.

Types:
    not-a-knot: Third derivative is continuous at the second and second-to-last knots
    periodic: Function values and derivatives are equal at the boundaries
    second_derivative: Second derivative is specified at boundaries
    first_derivative: First derivative is specified at boundaries
"""

class Spline1D:
    """
    One-dimensional cubic spline interpolation using B-spline basis functions.
    
    This class implements cubic spline interpolation with various boundary conditions.
    The spline is represented using B-spline basis functions and supports evaluation
    of the function value and its first three derivatives.
    
    Attributes:
        _interval: Tuple of (start, end) values for the interpolation domain
        _boundary_condition_type: Tuple of boundary condition types for start and end
        _boundary_condition_value: Tuple of boundary condition values for start and end
        _coeff: Array of spline coefficients
        _h: Step size between knots
        _a: Start value of the interval
    """

    def __init__(self, interval: Tuple[float, float], yv: np.ndarray, boundary_condition_type: Tuple[str, str] = ("not-a-knot", "not-a-knot"), boundary_condition_value: Tuple[float, float] = (0.0, 0.0) ) -> None:
        """
        Initialize a 1D cubic spline interpolator.
        
        Parameters
        ----------
        interval : Tuple[float, float]
            (start, end) values defining the interpolation domain
        yv : np.ndarray
            Array of function values at the interpolation points
        boundary_condition_type : Tuple[str, str], optional
            Boundary conditions for (start, end) of the interval.
            Options: "not-a-knot", "periodic", "first_derivative", "second_derivative"
            Default: ("not-a-knot", "not-a-knot")
        boundary_condition_value : Tuple[float, float], optional
            Values for the boundary conditions. For derivative conditions,
            these are the derivative values at the boundaries.
            Default: (0.0, 0.0)
            
        Raises
        ------
        ValueError
            If unsupported boundary condition is specified or if periodic
            boundary conditions are used with mismatching endpoint values.
        """
        self._interval = interval
        # print("Spline1D interval:", interval)
        # self._yv = yv
        # checking for unsupported boundary condition
        for bc in boundary_condition_type:
            if bc not in SPLINE_BOUNDARY_CONDITIONS:
                raise ValueError(f"Unsupported boundary condition: {bc}. The following conditions are supported: {SPLINE_BOUNDARY_CONDITIONS}")

        if "periodic" in boundary_condition_type:
            self._boundary_condition_type = ("periodic", "periodic")
            if yv[0] != yv[-1]:
                raise ValueError("For periodic boundary conditions, the first and last values must be equal.")
        else:
            self._boundary_condition_type = boundary_condition_type
        self._boundary_condition_value = boundary_condition_value
        self._n = len(yv) - 1
        self._h = (interval[1]-interval[0])/(self._n)
        self._a = interval[0]

        self._matrix = self._base_diag()
        self._rhs = np.zeros(self._n+3)
        self._rhs[:-2] = yv
        self._add_bc_to_matrix_rhs()

        self._coeff = np.linalg.solve(self._matrix, self._rhs)
    
    def _base_diag(self) -> np.ndarray:
        """
        Create the base diagonal matrix for the spline coefficient system.
        
        Constructs the banded matrix representing the continuity conditions
        for the cubic spline. The matrix has 1s on the main diagonal,
        4s on the first superdiagonal, and 1s on the second superdiagonal.
        
        Returns
        -------
        np.ndarray
            The base diagonal matrix of shape (n+3, n+3) where n is the number
            of intervals (len(yv) - 1)
        """
        tmp_diag = np.diag(np.ones(self._n+3), 0) + np.diag(4*np.ones(self._n+2), 1) + np.diag(np.ones(self._n+1), 2)
        if self._boundary_condition_type[0] == "periodic":
            tmp_diag[-3:, :] = 0
        else:
            tmp_diag[-2:, :] = 0
        return tmp_diag

    def _add_bc_to_matrix_rhs(self) -> None:
        """
        Apply boundary conditions to the coefficient matrix and right-hand side.
        
        Modifies the system matrix and right-hand side vector to incorporate
        the specified boundary conditions. Different boundary conditions
        require different modifications to the last two rows of the system.
        
        Boundary Conditions Implemented:
        - periodic: Enforces equality of function values and derivatives at boundaries
        - first_derivative: Sets specified first derivative values at boundaries
        - second_derivative: Sets specified second derivative values at boundaries
        - not-a-knot: Enforces continuity of third derivative at interior knots
        """
        if self._boundary_condition_type[0] == "periodic":
            self._matrix[-3, 0] = 1
            self._matrix[-3, -3] = -1
            self._matrix[-2, 1] = 1
            self._matrix[-2, -2] = -1
            self._matrix[-1, 2] = 1
            self._matrix[-1, -1] = -1
            self._rhs[-3:] = 0
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
        """
        Cubic B-spline basis function.
        
        Evaluates the cubic B-spline basis function φ(t) which is piecewise
        defined and has compact support on [-2, 2]. This is the fundamental
        building block for the cubic spline interpolation.
        
        The function is defined as:
        - φ(t) = (2 - |t|)³ for 1 < |t| ≤ 2
        - φ(t) = 4 - 6t² + 3|t|³ for |t| ≤ 1
        - φ(t) = 0 otherwise
        
        Parameters
        ----------
        t : float or np.ndarray
            Parameter value(s) at which to evaluate the basis function
            
        Returns
        -------
        float or np.ndarray
            Basis function value(s) at the given parameter(s)
        """
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
        """
        First derivative of the cubic B-spline basis function.
        
        Evaluates dφ/dt, the first derivative of the cubic B-spline basis
        function with respect to the parameter t. This is used for computing
        the first derivative of the interpolated spline.
        
        Parameters
        ----------
        t : float or np.ndarray
            Parameter value(s) at which to evaluate the derivative
            
        Returns
        -------
        float or np.ndarray
            First derivative value(s) at the given parameter(s)
        """
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
        """
        Second derivative of the cubic B-spline basis function.
        
        Evaluates d²φ/dt², the second derivative of the cubic B-spline basis
        function with respect to the parameter t. This is used for computing
        the second derivative of the interpolated spline.
        
        Parameters
        ----------
        t : float or np.ndarray
            Parameter value(s) at which to evaluate the second derivative
            
        Returns
        -------
        float or np.ndarray
            Second derivative value(s) at the given parameter(s)
        """
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
        """
        Third derivative of the cubic B-spline basis function.
        
        Evaluates d³φ/dt³, the third derivative of the cubic B-spline basis
        function with respect to the parameter t. This is used for computing
        the third derivative of the interpolated spline and for implementing
        the not-a-knot boundary condition.
        
        Parameters
        ----------
        t : float or np.ndarray
            Parameter value(s) at which to evaluate the third derivative
            
        Returns
        -------
        float or np.ndarray
            Third derivative value(s) at the given parameter(s)
        """
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

    def eval_spline(self, x: float | np.ndarray) -> Tuple[float, float, float, float] | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate the spline and its derivatives at given point(s).
        
        Computes the spline value and its first three derivatives at the
        specified evaluation point(s). The evaluation is performed using
        a vectorized approach for efficiency.
        
        Parameters
        ----------
        x : float or np.ndarray
            Point(s) at which to evaluate the spline. Can be a single value
            or an array of values.
            
        Returns
        -------
        Tuple containing four elements:
        - spline value(s)
        - first derivative(s)
        - second derivative(s) 
        - third derivative(s)
        
        For single point input: returns (float, float, float, float)
        For array input: returns (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        """
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
    def coeff(self) -> np.ndarray:
        """
        Get the spline coefficients.
        
        Returns
        -------
        np.ndarray
            Array of spline coefficients of length n+3, where n is the
            number of intervals (len(yv) - 1)
        """
        return self._coeff

if __name__ == '__main__':
    interval = (0.0, 1.0)
    yv = np.array([0.0, 1.0, 4.0, 9.0, 0.0])
    bc = ("periodic", "periodic")
    bc_value = (10.0, 0.0)
    spline = Spline1D(interval, yv, bc, bc_value)
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