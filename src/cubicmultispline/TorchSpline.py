import torch
from typing import Tuple
from .TorchSpline1D import TorchSpline1D
from itertools import product as iproduct


class TorchSpline:
    """
    Multidimensional cubic spline interpolation using tensor product B-splines (PyTorch backend).

    Drop-in replacement for Spline that operates on torch.Tensor inputs and keeps
    all internal data on the same device/dtype as the input.
    """

    def __init__(
        self,
        interval: Tuple[Tuple[float, float, int]],
        yv,
        boundary_condition_type: Tuple[Tuple[str, str]] = None,
        boundary_condition_value: Tuple[Tuple[float, float]] = None
    ) -> None:
        if not isinstance(yv, torch.Tensor):
            yv = torch.as_tensor(yv, dtype=torch.float64)
        self._device = yv.device
        self._dtype = yv.dtype

        num_vals = 1
        for i in range(len(interval)):
            if len(interval[i]) != 3:
                raise ValueError(f"Interval {i} must be a tuple of tuples of length 3.")
            num_vals *= interval[i][2]
        if num_vals != len(yv):
            raise ValueError(f"Number of sample values {len(yv)} does not match number of sample points defined by product of third value inside intervals.")

        self._ndim = len(interval)
        self._interval = interval
        self._knots = [torch.linspace(interval[k][0], interval[k][1], interval[k][2], device=self._device, dtype=self._dtype) for k in range(self._ndim)]

        if boundary_condition_type is None:
            boundary_condition_type = tuple(("not-a-knot", "not-a-knot") for _ in range(self._ndim))
        if boundary_condition_value is None:
            boundary_condition_value = tuple((0.0, 0.0) for _ in range(self._ndim))
        self._boundary_condition_type = boundary_condition_type
        self._boundary_condition_value = boundary_condition_value

        _tmp_coeff = TorchSpline.recursive_spline(
            interval=self._interval,
            yv=yv,
            boundary_condition_type=self._boundary_condition_type,
            boundary_condition_value=self._boundary_condition_value,
            device=self._device,
            dtype=self._dtype
        )
        self._coeff_shape = tuple([inter[2] + 2 for inter in interval])
        self._coeff = _tmp_coeff.reshape(self._coeff_shape)

    @staticmethod
    def recursive_spline(
        interval: Tuple[Tuple[float, float, int]],
        yv: torch.Tensor,
        boundary_condition_type: Tuple[Tuple[str, str]],
        boundary_condition_value: Tuple[Tuple[float, float]],
        device=None,
        dtype=None
    ) -> torch.Tensor:
        if len(interval) == 1:
            coeff = TorchSpline1D(interval[0], yv, boundary_condition_type[0], boundary_condition_value[0]).coeff
            return coeff

        ci_star_size = 1
        ci_star_len = 1
        ci_size = 1
        num_recur_coords = 1
        for i in range(len(interval)):
            if i != 0:
                ci_star_size *= (interval[i][2] + 2)
                ci_star_len *= (interval[i][2] + 2)
                num_recur_coords *= interval[i][2]
            else:
                ci_star_size *= interval[i][2]
            ci_size *= (interval[i][2] + 2)
        ci_len = interval[0][2] + 2

        ci_star = torch.zeros(ci_star_size, device=device, dtype=dtype)
        for i in range(interval[0][2]):
            ci_star[i * ci_star_len:(i + 1) * ci_star_len] = TorchSpline.recursive_spline(
                interval[1:],
                yv[i * num_recur_coords:(i + 1) * num_recur_coords],
                boundary_condition_type[1:],
                boundary_condition_value[1:],
                device=device,
                dtype=dtype
            )

        ci = torch.zeros(ci_size, device=device, dtype=dtype)
        for i in range(ci_star_len):
            factor = 1 / (6 ** (len(interval) - 1))
            new_bc_vals = (boundary_condition_value[0][0] * factor, boundary_condition_value[0][1] * factor)
            tmp_spline = TorchSpline1D(interval[0], ci_star[i::ci_star_len], boundary_condition_type[0], new_bc_vals)
            ci[i::ci_star_len] = tmp_spline.coeff

        return ci

    @property
    def coeff(self) -> torch.Tensor:
        return self._coeff

    def _find_span_and_local(self, x_1d: torch.Tensor, knots_1d: torch.Tensor):
        x_c = torch.clamp(x_1d, knots_1d[0].item(), knots_1d[-1].item())

        idx = torch.searchsorted(knots_1d, x_c, side='right') - 1
        idx = torch.clamp(idx, 0, len(knots_1d) - 2)

        h = knots_1d[idx + 1] - knots_1d[idx]
        t = (x_c - knots_1d[idx]) / h
        return idx, t, h

    def eval_spline(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, device=self._device, dtype=self._dtype)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        d = self._ndim
        x = x.reshape(-1, d)
        N = x.shape[0]
        C = self._coeff

        active_idx = []
        phi_vals = []
        dphi_vals = []
        d2phi_vals = []
        h_arr = []

        for k in range(d):
            knots_k = self._knots[k]
            span_idx, t_k, h_k = self._find_span_and_local(x[:, k], knots_k)

            max_idx = self._coeff_shape[k] - 1
            offsets = torch.arange(4, device=self._device, dtype=torch.long)
            idx_mat = span_idx.unsqueeze(1) + offsets.unsqueeze(0)
            idx_mat = torch.clamp(idx_mat, 0, max_idx)

            t_shifted = t_k.unsqueeze(1) - (offsets.unsqueeze(0).to(self._dtype) - 1)

            phi_mat = TorchSpline1D._phi(t_shifted)
            dphi_mat = TorchSpline1D._dphi_dt(t_shifted)
            d2phi_mat = TorchSpline1D._d2phi_dt2(t_shifted)

            active_idx.append(idx_mat)
            phi_vals.append(phi_mat)
            dphi_vals.append(dphi_mat / h_k.unsqueeze(1))
            d2phi_vals.append(d2phi_mat / h_k.unsqueeze(1) ** 2)
            h_arr.append(h_k)

        f = torch.zeros(N, device=self._device, dtype=self._dtype)
        grad = torch.zeros((N, d), device=self._device, dtype=self._dtype)
        hess = torch.zeros((N, d, d), device=self._device, dtype=self._dtype)

        offsets_per_dim = [range(4)] * d

        for combo in iproduct(*offsets_per_dim):
            c_idx = tuple(active_idx[k][:, combo[k]] for k in range(d))
            c_vals = C[c_idx]

            phi_prod = torch.ones(N, device=self._device, dtype=self._dtype)
            for k in range(d):
                phi_prod *= phi_vals[k][:, combo[k]]

            f += c_vals * phi_prod

            for j in range(d):
                dphi_prod = torch.ones(N, device=self._device, dtype=self._dtype)
                for k in range(d):
                    if k == j:
                        dphi_prod *= dphi_vals[k][:, combo[k]]
                    else:
                        dphi_prod *= phi_vals[k][:, combo[k]]
                grad[:, j] += c_vals * dphi_prod

            for j in range(d):
                for l in range(j, d):
                    d2phi_prod = torch.ones(N, device=self._device, dtype=self._dtype)
                    for k in range(d):
                        if k == j and k == l:
                            d2phi_prod *= d2phi_vals[k][:, combo[k]]
                        elif k == j or k == l:
                            d2phi_prod *= dphi_vals[k][:, combo[k]]
                        else:
                            d2phi_prod *= phi_vals[k][:, combo[k]]
                    contrib = c_vals * d2phi_prod
                    hess[:, j, l] += contrib
                    if j != l:
                        hess[:, l, j] += contrib

        return f, grad, hess
