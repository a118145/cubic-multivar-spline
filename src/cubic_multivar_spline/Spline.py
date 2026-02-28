import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from cubic_multivar_spline.Spline1D import Spline1D

class Spline:

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
        self._h = np.zeros(self._ndim)
        self._a = np.zeros(self._ndim)
        self._n_elems = np.zeros(self._ndim)
        for i in range(self._ndim):
            self._h[i] = (interval[i][1] - interval[i][0]) / (interval[i][2]-1)
            self._a[i] = interval[i][0]
            self._n_elems[i] = interval[i][2] - 1
        
        self._yv = yv
        if boundary_condition_type is None:
            boundary_condition_type = tuple(("not-a-knot", "not-a-knot") for _ in range(self._ndim))
        if boundary_condition_value is None:
            boundary_condition_value = tuple((0.0, 0.0) for _ in range(self._ndim))
        self._boundary_condition_type = boundary_condition_type
        self._boundary_condition_value = boundary_condition_value
        
        _tmp_coeff, _str_coeff = Spline.recursive_spline(
            interval = self._interval, 
            yv = self._yv, 
            boundary_condition_type = self._boundary_condition_type, 
            boundary_condition_value = self._boundary_condition_value)
        _coeff_shape = tuple([inter[2] + 2 for inter in interval])
        print(_coeff_shape)
        print(_tmp_coeff.size)
        self._coeff_orig = _tmp_coeff
        self._coeff = np.reshape(_tmp_coeff, _coeff_shape, order = 'C')
        self._str_coeff = _str_coeff
        print(self._coeff.size)

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
        print('##########')
        print(f"Level {len(interval)}")
        print(interval)
        if len(interval) is 1:
            print("1D spline")
            # lowest level of recursion reached -> 1D spline
            # print(boundary_condition_type[0])
            # print(boundary_condition_value[0])
            coeff = Spline1D(interval[0], yv, boundary_condition_type[0], boundary_condition_value[0]).coeff 
            str_coeff = ['1' for i in range(len(coeff))]
            return coeff, str_coeff

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
        
        print('ci_star_size', ci_star_size)
        print('ci_star_len', ci_star_len)
        print('ci_size', ci_size)
        print('ci_len', ci_len)
        print('num_recur_coords', num_recur_coords)
        print('ci_size // ci_star_len', ci_size // ci_star_len)

        str_ci_star = [] * ci_star_size
        ci_star = np.zeros(ci_star_size)
        for i in range(interval[0][2]):
            ci_star[i*ci_star_len:(i+1)*ci_star_len], str_ci_star[i*ci_star_len:(i+1)*ci_star_len] = Spline.recursive_spline(
                interval[1:], 
                yv[i*num_recur_coords:(i+1)*num_recur_coords],
                boundary_condition_type[1:], 
                boundary_condition_value[1:]
            )
        print('##########')
        print(f"Level {len(interval)}")
        ci = np.zeros(ci_size)
        str_ci = [] * ci_size
        tmp_str = ['2' for _ in range(ci_star_len)]
        for i in range(ci_star_len): #range(ci_size//ci_star_len):
            tmp_spline = Spline1D(interval[0], ci_star[i::ci_star_len], boundary_condition_type[0], boundary_condition_value[0])
            print(tmp_spline.coeff.size)
            print(ci.size, i, ci_len)
            ci[i*ci_len:(i+1)*ci_len] = tmp_spline.coeff
            # ci[i::ci_star_len] = tmp_spline.coeff
            # str_ci[i*ci_len:(i+1)*ci_len] = [tmp_str[i] + s for s in str_ci_star[i::ci_star_len]]
        print('############')
        return ci, str_ci
        
    @property
    def coeff(self) -> np.ndarray:
        return self._coeff
        
    @property
    def coeff_orig(self) -> np.ndarray:
        return self._coeff_orig

    def eval_spline_old(self, x: float, y: float) -> Tuple[float, float, float, float]:
        hi = (self._interval[0][1] - self._interval[0][0]) / (self._interval[0][2]-1)
        hj = (self._interval[1][1] - self._interval[1][0]) / (self._interval[1][2]-1)
        res = 0
        for i in range(7):
            for j in range(8):
                ti = (x - self._interval[0][0]) / hi - i + 1
                tj = (y - self._interval[1][0]) / hj - j + 1
                res += self._coeff[i, j] * Spline1D._phi(ti) * Spline1D._phi(tj)
        return res

    def eval_spline_windsurf(self, coords: np.ndarray) -> Tuple[float, np.ndarray] | Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the spline and its gradient at given coordinates with vectorized operations.
        
        Args:
            coords: Array of shape (n_points, ndim) or (ndim,) for single point
            
        Returns:
            Tuple of (spline_value, gradient)
            For single point: returns (scalar, 1D array), for multiple points: returns (1D array, 2D array)
        """
        coords = np.atleast_2d(coords)
        if coords.shape[1] != self._ndim:
            raise ValueError(f"Number of coordinates {coords.shape[1]} does not match number of dimensions {self._ndim}.")
        
        n_points = coords.shape[0]
        
        # Initialize results
        spline_val = np.zeros(n_points, dtype=float)
        gradient = np.zeros((n_points, self._ndim), dtype=float)
        
        # For each point, compute the spline value and gradient using tensor product basis functions
        for point_idx in range(n_points):
            point = coords[point_idx]
            
            # Initialize contribution accumulator
            point_value = 0.0
            point_gradient = np.zeros(self._ndim, dtype=float)
            
            # Iterate through all coefficient indices
            for coeff_indices in np.ndindex(*self._coeff.shape):
                # Compute t values and phi values for each dimension
                t_values = []
                phi_values = []
                dphi_values = []
                
                for dim in range(self._ndim):
                    # Use the same formula as eval_spline_old: (x - a)/h - i + 1
                    t = (point[dim] - self._a[dim]) / self._h[dim] - coeff_indices[dim] + 1
                    t_values.append(t)
                    phi_values.append(Spline1D._phi(t))
                    dphi_values.append(Spline1D._dphi_dt(t))
                
                # Get coefficient and add contribution (tensor product)
                coeff = self._coeff[coeff_indices]
                phi_product = np.prod(phi_values)
                
                # Add to spline value
                point_value += coeff * phi_product
                
                # Add to gradient (chain rule for each dimension)
                for dim in range(self._ndim):
                    # Replace phi in dimension 'dim' with its derivative
                    phi_with_derivative = phi_values.copy()
                    phi_with_derivative[dim] = dphi_values[dim]
                    
                    # Chain rule: d/dx_dim = (1/h_dim) * dphi/dt
                    gradient_contribution = coeff * np.prod(phi_with_derivative) / self._h[dim]
                    point_gradient[dim] += gradient_contribution
            
            spline_val[point_idx] = point_value
            gradient[point_idx] = point_gradient
        
        # Return appropriate type based on input
        if n_points == 1:
            return float(spline_val[0]), gradient[0]
        else:
            return spline_val, gradient



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
    spline_2d = Spline(
        interval=((0, 1, n), (0, 1, n+1)),
        yv=test_func2d,
        boundary_condition_type=(("not-a-knot", "not-a-knot"), ("not-a-knot", "not-a-knot")),
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

    for i in range(coeff_reshape.shape[0]):
        for j in range(coeff_reshape.shape[1]):
            for k in range(coeff_reshape.shape[2]):
                pointer = k + j * coeff_reshape.shape[2] + i * coeff_reshape.shape[1]*coeff_reshape.shape[2]
                # print('#######')
                # print(pointer)
                # print(test_range_reshape[i,j,k])
                if spline.coeff_orig[pointer] != coeff_reshape[i,j,k]:
                    print(pointer)
                    print(spline.coeff_orig[pointer],coeff_reshape[i,j,k])
                    print('########')
                # coords[pointer] = np.array([x[i], y[j], z[k]])
    
    # print(spline.eval_spline(0.5, 0.59, 0.5))
    # print(test_function(np.array([0.5]), np.array([0.59]), np.array([0.5])))
    n_spline_eval = 5
    x_spline_eval = np.linspace(0, 1, n_spline_eval)
    y_spline_eval = np.linspace(0, 1, n_spline_eval)
    x_spline_eval_grid, y_spline_eval_grid = np.meshgrid(x_spline_eval, y_spline_eval)
    z_spline_eval = np.zeros((n_spline_eval, n_spline_eval))
    for i in range(n_spline_eval):
        for j in range(n_spline_eval):
            z_spline_eval[j,i] = spline_2d.eval_spline_old(x_spline_eval[i], y_spline_eval[j])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xgrid, ygrid, test_func2d_reshaped)
    ax.plot_surface(x_spline_eval_grid, y_spline_eval_grid, z_spline_eval)
    plt.show()