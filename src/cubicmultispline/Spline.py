import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from Spline1D import Spline1D
from itertools import product as iproduct

class Spline:
    """
    Multidimensional cubic spline interpolation using tensor product B-splines.
    
    This class implements multidimensional cubic spline interpolation by constructing
    tensor products of 1D B-spline basis functions. The interpolation
    supports various boundary conditions and can evaluate function values,
    gradients, and Hessians.
    
    The implementation uses a recursive approach to compute coefficients
    and a vectorized evaluation method for efficiency.
    
    Attributes:
        _ndim: Number of dimensions
        _interval: List of (start, end, n_points) tuples for each dimension
        _boundary_condition_type: Boundary conditions for each dimension
        _boundary_condition_value: Boundary condition values for each dimension
        _coeff: Multidimensional array of spline coefficients
        _knots: List of knot vectors for each dimension
    """

    def __init__(
        self, 
        interval: Tuple[Tuple[float, float, int]], 
        yv: np.ndarray, 
        boundary_condition_type: Tuple[Tuple[str, str]] = None, 
        boundary_condition_value: Tuple[Tuple[float, float]] = None 
        ) -> None:
        """
        Initialize a multidimensional cubic spline interpolator.
        
        Parameters
        ----------
        interval : Tuple[Tuple[float, float, int]]
            For each dimension: (start, end, n_points) defining the domain
            and number of interpolation points
        yv : np.ndarray
            Flattened array of function values at all grid points.
            Length must equal product of n_points from all dimensions.
        boundary_condition_type : Tuple[Tuple[str, str]], optional
            Boundary conditions for each dimension. Each tuple contains
            conditions for (start, end) of that dimension.
            Default: all dimensions use "not-a-knot"
        boundary_condition_value : Tuple[Tuple[float, float]], optional
            Values for boundary conditions. Default: all zeros.
            
        Raises
        ------
        ValueError
            If interval format is incorrect or if number of values
            doesn't match the grid specification.
        """

        num_vals = 1
        for i in range(len(interval)):
            if len(interval[i]) != 3:
                raise ValueError(f"Interval {i} must be a tuple of tuples of length 3.")
            num_vals *= interval[i][2]
        if num_vals != len(yv):
            raise ValueError(f"Number of sample values {len(yv)} does not match number of sample points defined by product of third value inside intervals.")

        self._ndim = len(interval)
        self._interval = interval
        # self._h = np.zeros(self._ndim)
        # self._a = np.zeros(self._ndim)
        # self._n_elems = np.zeros(self._ndim)
        self._knots = [np.linspace(interval[k][0], interval[k][1], interval[k][2]) for k in range(self._ndim)]
        # for i in range(self._ndim):
            # self._h[i] = (interval[i][1] - interval[i][0]) / (interval[i][2]-1)
            # self._a[i] = interval[i][0]
            # self._n_elems[i] = interval[i][2] - 1
        
        # self._yv = yv
        if boundary_condition_type is None:
            boundary_condition_type = tuple(("not-a-knot", "not-a-knot") for _ in range(self._ndim))
        if boundary_condition_value is None:
            boundary_condition_value = tuple((0.0, 0.0) for _ in range(self._ndim))
        self._boundary_condition_type = boundary_condition_type
        self._boundary_condition_value = boundary_condition_value
        
        _tmp_coeff = Spline.recursive_spline(
            interval = self._interval, 
            yv = yv, 
            boundary_condition_type = self._boundary_condition_type, 
            boundary_condition_value = self._boundary_condition_value)
        # _tmp_coeff = Spline.recursive_spline_test(
        #     interval = self._interval, 
        #     yv = yv, 
        #     boundary_condition_type = self._boundary_condition_type, 
        #     boundary_condition_value = self._boundary_condition_value)
        self._coeff_shape = tuple([inter[2] + 2 for inter in interval])
        # print(self._coeff_shape)
        # print(_tmp_coeff.size)
        # self._coeff_orig = _tmp_coeff
        self._coeff = np.reshape(_tmp_coeff, self._coeff_shape, order = 'C')
        # self._str_coeff = _str_coeff
        # print(self._coeff.size)

    # @staticmethod
    # def recursive_spline_test(
    #     interval: Tuple[Tuple[float, float, int]], 
    #     yv: np.ndarray, 
    #     boundary_condition_type: Tuple[Tuple[str, str]], 
    #     boundary_condition_value: Tuple[Tuple[float, float]]
    #     ) -> np.ndarray:
    #     ci_star = np.zeros((13,11))
    #     for i in range(interval[0][2]):
    #         tmp_spline = Spline1D(interval[1], yv[i*interval[1][2]:(i+1)*interval[1][2]], boundary_condition_type[1], boundary_condition_value[1])
    #         # print(len(tmp_spline.coeff))
    #         ci_star[:,i] = tmp_spline.coeff
    #     ci = np.zeros((13,13))
    #     for i in range(np.size(ci_star, 0)):
    #         tmp_spline = Spline1D(interval[0], ci_star[i,:], boundary_condition_type[0], boundary_condition_value[0])
    #         ci[i::13,:] = tmp_spline.coeff
    #     return ci

    @staticmethod
    def recursive_spline(
        interval: Tuple[Tuple[float, float, int]], 
        yv: np.ndarray, 
        boundary_condition_type: Tuple[Tuple[str, str]], 
        boundary_condition_value: Tuple[Tuple[float, float]]
        ) -> np.ndarray:
        """
        Recursively compute multidimensional spline coefficients.
        
        Uses a recursive approach to compute tensor product spline coefficients.
        For each dimension, 1D splines are computed along the other
        dimensions, then combined using another 1D spline.
        
        Parameters
        ----------
        interval : Tuple[Tuple[float, float, int]]
            Intervals for remaining dimensions
        yv : np.ndarray
            Function values for remaining dimensions
        boundary_condition_type : Tuple[Tuple[str, str]]
            Boundary conditions for remaining dimensions
        boundary_condition_value : Tuple[Tuple[float, float]]
            Boundary condition values for remaining dimensions
            
        Returns
        -------
        np.ndarray
            Flattened array of multidimensional spline coefficients
        """
        # print('##########')
        # print(f"Level {len(interval)}")
        # print(interval)
        if len(interval) is 1:
            # lowest level of recursion reached -> 1D spline
            coeff = Spline1D(interval[0], yv, boundary_condition_type[0], boundary_condition_value[0]).coeff 
            return coeff

        ci_star_size = 1
        ci_star_len = 1
        ci_size = 1
        ci_len = 1
        num_recur_coords = 1
        for i in range(len(interval)):
            if i != 0:
                ci_star_size *= (interval[i][2] + 2)
                ci_star_len *= (interval[i][2] + 2)
                num_recur_coords *= interval[i][2]
            else:
                ci_star_size *= interval[i][2]
            ci_len *= (interval[i][2] + 2)
            ci_size *= (interval[i][2] + 2)
        ci_len = interval[0][2] + 2
        
        # print('ci_star_size', ci_star_size)
        # print('ci_star_len', ci_star_len)
        # print('ci_size', ci_size)
        # print('ci_len', ci_len)
        # print('num_recur_coords', num_recur_coords)
        # print('ci_size // ci_star_len', ci_size // ci_star_len)

        ci_star = np.zeros(ci_star_size)
        for i in range(interval[0][2]):
            ci_star[i*ci_star_len:(i+1)*ci_star_len] = Spline.recursive_spline(
                interval[1:], 
                yv[i*num_recur_coords:(i+1)*num_recur_coords],
                boundary_condition_type[1:], 
                boundary_condition_value[1:]
            )
        # print('##########')
        # print(f"Level {len(interval)}")
        ci = np.zeros(ci_size)
        for i in range(ci_star_len): #range(ci_size//ci_star_len):
            # scaling of boundary condition values needed due to more basis functions per level
            factor = 1/(6**(len(interval)-1))
            new_bc_vals = tuple([boundary_condition_value[0][0]*factor, boundary_condition_value[0][1]*factor])
            tmp_spline = Spline1D(interval[0], ci_star[i::ci_star_len], boundary_condition_type[0], new_bc_vals)
            # print(tmp_spline.coeff.size)
            # print(ci.size, i, ci_len)
            # ci[i*ci_len:(i+1)*ci_len] = tmp_spline.coeff
            ci[i::ci_star_len] = tmp_spline.coeff
        # print('############')
        return ci
        
    @property
    def coeff(self) -> np.ndarray:
        """
        Get multidimensional spline coefficients.
        
        Returns
        -------
        np.ndarray
            Multidimensional array of spline coefficients with shape
            (n1+2, n2+2, ..., nd+2) where ni is the number
            of points in dimension i
        """
        return self._coeff
  
    def _find_span_and_local(self, x_1d, knots_1d):
        """
        Find knot span index and local parameter for 1D evaluation.
        
        For each evaluation point, determines which knot span it falls in
        and computes the local parameter t in [0, 1) within that span.
        
        Parameters
        ----------
        x_1d : np.ndarray
            1D array of evaluation points
        knots_1d : np.ndarray
            Knot vector for the dimension
            
        Returns
        -------
        idx : np.ndarray
            Index of left knot in span for each point of shape (N,)
        t : np.ndarray
            Local parameter in [0, 1) for each point of shape (N,)
        h : np.ndarray
            Knot spacing for each span of shape (N,)
        """
        # Clamp points to the valid range
        knots_1d = np.asarray(knots_1d)
        x_c = np.clip(x_1d, knots_1d[0], knots_1d[-1])

        # Left knot index (max. len-2 so right boundary is correct)
        idx = np.searchsorted(knots_1d, x_c, side='right') - 1
        idx = np.clip(idx, 0, len(knots_1d) - 2)

        h = knots_1d[idx + 1] - knots_1d[idx]
        t = (x_c - knots_1d[idx]) / h
        return idx, t, h


    def eval_spline(self, x):
        """
        Evaluate multidimensional spline and its derivatives.
        
        Computes spline values, gradients, and Hessians at specified evaluation
        points using a vectorized tensor product approach. Based on the
        methodology from DOI 10.1007/s10614-007-9092-4, Section 3.2.
        
        Parameters
        ----------
        x : np.ndarray
            Evaluation points of shape (N, d) where N is number of points
            and d is number of dimensions
            
        Returns
        -------
        f : np.ndarray
            Function values at evaluation points, shape (N,)
        grad : np.ndarray
            Gradient vectors at evaluation points, shape (N, d)
        hess : np.ndarray
            Hessian matrices at evaluation points, shape (N, d, d)
            
        Notes
        -----
        The evaluation uses a vectorized approach that efficiently computes
        contributions from all active basis functions (up to 4^d per point).
        """
        x = np.atleast_2d(x)
        d = self._ndim
        x = np.reshape(x, (-1, d))
        N, _ = x.shape
        C = self._coeff

        # ------------------------------------------------------------------ #
        # 1) For each dimension: active indices + phi, dphi, d2phi evaluation  #
        # ------------------------------------------------------------------ #
        # active_idx[k]  : (N, 4) int   – up to 4 active basis indices in Dim k
        # phi_vals[k]    : (N, 4) float
        # dphi_vals[k]   : (N, 4) float  (derivative w.r.t. t)
        # d2phi_vals[k]  : (N, 4) float

        active_idx = []
        phi_vals   = []
        dphi_vals  = []
        d2phi_vals = []
        h_arr      = []   # Knot distance (for chain rule)

        for k in range(d):
            knots_k = np.asarray(self._knots[k])
            span_idx, t_k, h_k = self._find_span_and_local(x[:, k], knots_k)

            # n control points → n+2 coefficients (one extra on each side)
            # Coefficient index for span i: shifted by +1 compared to span_idx
            # → active indices: span_idx+0, span_idx+1, span_idx+2, span_idx+3
            max_idx = self._coeff_shape[k] - 1
            offsets = np.array([0, 1, 2, 3])             # (4,)
            idx_mat = span_idx[:, None] + offsets[None, :]
            idx_mat = np.clip(idx_mat, 0, max_idx)

            # t_shifted: local parameter relative to the respective basis function center
            # Center at coefficient index span_idx+o → t = t_k - (o - 1) = t_k + 1 - o
            # (o=0 → t_k+1, o=1 → t_k, o=2 → t_k-1, o=3 → t_k-2)
            t_shifted = t_k[:, None] - (offsets[None, :] - 1)  # (N, 4)

            phi_mat   = Spline1D._phi(t_shifted)        # (N, 4)
            dphi_mat  = Spline1D._dphi_dt(t_shifted)    # (N, 4)
            d2phi_mat = Spline1D._d2phi_dt2(t_shifted)  # (N, 4)

            active_idx.append(idx_mat)          # (N, 4)
            phi_vals.append(phi_mat)
            dphi_vals.append(dphi_mat / h_k[:, None])   # Chain rule: dt/dx = 1/h
            d2phi_vals.append(d2phi_mat / h_k[:, None]**2)
            h_arr.append(h_k)

        # ------------------------------------------------------------------ #
        # 2) Sum over all 4^d combinations                                    #
        # ------------------------------------------------------------------ #
        f    = np.zeros(N)
        grad = np.zeros((N, d))
        hess = np.zeros((N, d, d))

        offsets_per_dim = [range(4)] * d   # 4 entries per dimension

        for combo in iproduct(*offsets_per_dim):
            # combo = (o0, o1, ..., o_{d-1}), one local index 0..3 per dimension

            # Coefficient indices for all N points
            c_idx = tuple(active_idx[k][:, combo[k]] for k in range(d))  # d x (N,)
            c_vals = C[c_idx]   # (N,)

            # Product of basis functions (all dimensions)
            phi_prod = np.ones(N)
            for k in range(d):
                phi_prod *= phi_vals[k][:, combo[k]]

            f += c_vals * phi_prod

            # Gradient: Product rule – one dimension derived, rest phi
            for j in range(d):
                dphi_prod = np.ones(N)
                for k in range(d):
                    if k == j:
                        dphi_prod *= dphi_vals[k][:, combo[k]]
                    else:
                        dphi_prod *= phi_vals[k][:, combo[k]]
                grad[:, j] += c_vals * dphi_prod

            # Hessian: two dimensions derived
            for j in range(d):
                for l in range(j, d):
                    d2phi_prod = np.ones(N)
                    for k in range(d):
                        if k == j and k == l:          # j == l: 2. derivative
                            d2phi_prod *= d2phi_vals[k][:, combo[k]]
                        elif k == j or k == l:         # mixed derivative
                            d2phi_prod *= dphi_vals[k][:, combo[k]]
                        else:
                            d2phi_prod *= phi_vals[k][:, combo[k]]
                    contrib = c_vals * d2phi_prod
                    hess[:, j, l] += contrib
                    if j != l:
                        hess[:, l, j] += contrib       # Symmetry

        return f, grad, hess



if __name__ == "__main__":

    def test_function(x: np.array, y: np.array, z: np.array) -> np.array:
        tf = np.zeros(len(x)*len(y)*len(z))
        coords = np.zeros((len(x)*len(y)*len(z),3), dtype=float)
        for i in range(len(x)):
            for j in range(len(y)):
                for k in range(len(z)):
                    pointer = k + j * len(z) + i * len(y)*len(z)
                    tf[pointer] = 2*x[i]**2 + y[j]**3 + z[k]**4
                    coords[pointer] = np.array([x[i], y[j], z[k]])
        return tf, coords

    
    def test_function2d(x: np.array, y: np.array) -> np.array:
        tf = np.zeros(len(x)*len(y))
        coords = np.zeros((len(x)*len(y),2), dtype=float)
        for i in range(len(x)):
            for j in range(len(y)):
                    pointer = j + i * len(y)
                    tf[pointer] = 2*x[i]**2 + y[j]**3
                    coords[pointer] = np.array([x[i], y[j]])
        return tf, coords

    n = 5 # number of points in each direction
    xvals = np.linspace(0, 1, n)
    yvals = np.linspace(0, 1, n+1)
    zvals = np.linspace(0, 1, n)
    xgrid, ygrid = np.meshgrid(xvals, yvals, indexing='ij')
    # zgrid = np.reshape(zvals, (n,n+1))
    test_func, coords = test_function(xvals, yvals, zvals)
    test_func2d, coords = test_function2d(xvals, yvals)
    test_func2d_reshaped = np.reshape(test_func2d, (n,n+1))
    # print(coords)


    # print(len(test_function(np.linspace(0, 1, n), np.linspace(0, 1, n))))

    
    spline = Spline(
        interval=((0, 1, n), (0, 1, n+1), (0, 1, n)),
        yv=test_func,
        boundary_condition_type=(("not-a-knot", "not-a-knot"), ("not-a-knot", "not-a-knot"), ("not-a-knot", "not-a-knot")),
        boundary_condition_value=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    )


    n_nodes = [n, n+1]
    shape = tuple(n + 2 for n in n_nodes)            # (13, 13)
    # ci = np.random.randn(np.prod(shape))
    knots = [np.linspace(0, 1, n_nodes[k]) for k in range(2)]

    spline_2d = Spline(
        interval=((0, 1, n), (0, 1, n+1)),
        yv=test_func2d,
        boundary_condition_type=(("first_derivative", "first_derivative"), ("first_derivative", "first_derivative")),
        boundary_condition_value=((0.0, 0.0), (0.0, 0.0))
    )
    # print(spline.coeff)
    test_range = np.arange(n**2*(n+1))
    test_range_reshape = np.reshape(test_range, (n,n+1,n))

    coeff_reshape = np.reshape(spline.coeff, (n+2,n+3,n+2))

    # print(spline.eval_spline_windsurf([[0,0,0], [0.7,0.8,0.3]]))
    print(spline_2d.eval_spline_windsurf([[0,0], [0.7,0.8]]))
    print("Old function:")
    print(spline_2d.eval_spline_old(0, 0))
    print(spline_2d.eval_spline_old(0.7, 0.8))
    print("Expected:")
    print("f(0,0) =", 2*0**2 + 0**3)
    print("f(0.7,0.8) =", 2*0.7**2 + 0.8**3)

    # for i in range(coeff_reshape.shape[0]):
    #     for j in range(coeff_reshape.shape[1]):
    #         for k in range(coeff_reshape.shape[2]):
    #             pointer = k + j * coeff_reshape.shape[2] + i * coeff_reshape.shape[1]*coeff_reshape.shape[2]
    #             # print('#######')
    #             # print(pointer)
    #             # print(test_range_reshape[i,j,k])
    #             if spline.coeff_orig[pointer] != coeff_reshape[i,j,k]:
    #                 print(pointer)
    #                 print(spline.coeff_orig[pointer],coeff_reshape[i,j,k])
    #                 print('########')
                # coords[pointer] = np.array([x[i], y[j], z[k]])
    
    # print(spline.eval_spline(0.5, 0.59, 0.5))
    # print(test_function(np.array([0.5]), np.array([0.59]), np.array([0.5])))
    n_spline_eval = 200
    x_spline_eval = np.linspace(0, 1, n_spline_eval)
    y_spline_eval = np.linspace(0, 1, n_spline_eval)
    x_spline_eval_grid, y_spline_eval_grid = np.meshgrid(x_spline_eval, y_spline_eval, indexing='ij')
    z_spline_eval = np.zeros((n_spline_eval, n_spline_eval))
    # for i in range(n_spline_eval):
    #     for j in range(n_spline_eval):
    #         z_spline_eval[i,j] = spline_2d.eval_spline_old(x_spline_eval[i], y_spline_eval[j])
    z_spline_eval_ai, _, _ = spline_2d.eval_spline(np.concatenate([np.reshape(x_spline_eval_grid, (-1, 1)), np.reshape(y_spline_eval_grid, (-1, 1))], axis=1))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xgrid, ygrid, test_func2d_reshaped)
    # ax.plot_surface(x_spline_eval_grid, y_spline_eval_grid, z_spline_eval)
    ax.plot_surface(x_spline_eval_grid, y_spline_eval_grid, np.reshape(z_spline_eval_ai, (n_spline_eval, n_spline_eval)))
    plt.show()