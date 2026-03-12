import torch
from typing import Tuple


SPLINE_BOUNDARY_CONDITIONS = ("not-a-knot", "periodic", "second_derivative", "first_derivative")


class TorchSpline1D:
    """
    One-dimensional cubic spline interpolation using B-spline basis functions (PyTorch backend).

    Drop-in replacement for Spline1D that operates on torch.Tensor inputs and keeps
    all internal data on the same device/dtype as the input.
    """

    def __init__(self, interval: Tuple[float, float], yv, boundary_condition_type: Tuple[str, str] = ("not-a-knot", "not-a-knot"), boundary_condition_value: Tuple[float, float] = (0.0, 0.0)) -> None:
        if not isinstance(yv, torch.Tensor):
            yv = torch.as_tensor(yv, dtype=torch.float64)
        self._device = yv.device
        self._dtype = yv.dtype
        self._interval = interval

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
        self._h = (interval[1] - interval[0]) / self._n
        self._a = interval[0]

        self._matrix = self._base_diag()
        self._rhs = torch.zeros(self._n + 3, device=self._device, dtype=self._dtype)
        self._rhs[:-2] = yv
        self._add_bc_to_matrix_rhs()

        self._coeff = torch.linalg.solve(self._matrix, self._rhs)

    def _base_diag(self) -> torch.Tensor:
        n3 = self._n + 3
        tmp_diag = (torch.diag(torch.ones(n3, device=self._device, dtype=self._dtype), 0)
                    + torch.diag(4 * torch.ones(n3 - 1, device=self._device, dtype=self._dtype), 1)
                    + torch.diag(torch.ones(n3 - 2, device=self._device, dtype=self._dtype), 2))
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
            self._rhs[-3:] = 0
        else:
            for i, val in enumerate(self._boundary_condition_type):
                if val == "first_derivative":
                    col = i * self._n
                    self._matrix[-2 + i, col:col + 3] = torch.tensor([-1, 0, 1], device=self._device, dtype=self._dtype)
                    self._rhs[-2 + i] = self._boundary_condition_value[i] * self._h / 3
                if val == "second_derivative":
                    col = i * self._n
                    self._matrix[-2 + i, col:col + 3] = torch.tensor([1, -2, 1], device=self._device, dtype=self._dtype)
                    self._rhs[-2 + i] = self._boundary_condition_value[i] * self._h ** 2 / 6
                if val == "not-a-knot":
                    col = i * (self._n - 2)
                    self._matrix[-2 + i, col:col + 5] = torch.tensor([1, -4, 6, -4, 1], device=self._device, dtype=self._dtype)

    @staticmethod
    def _phi(t: torch.Tensor) -> torch.Tensor:
        t_abs = torch.abs(t)
        result = torch.zeros_like(t)

        mask1 = (t_abs > 1) & (t_abs <= 2)
        mask2 = t_abs <= 1

        result = torch.where(mask1, (2 - t_abs) ** 3, result)
        result = torch.where(mask2, 4 - 6 * t_abs ** 2 + 3 * t_abs ** 3, result)
        return result

    @staticmethod
    def _dphi_dt(t: torch.Tensor) -> torch.Tensor:
        t_abs = torch.abs(t)
        t_sign = torch.sign(t)
        result = torch.zeros_like(t)

        mask1 = (t_abs > 1) & (t_abs <= 2)
        mask2 = t_abs <= 1

        result = torch.where(mask1, -t_sign * 3 * (2 - t_abs) ** 2, result)
        result = torch.where(mask2, -t_sign * 12 * t_abs + t_sign * 9 * t_abs ** 2, result)
        return result

    @staticmethod
    def _d2phi_dt2(t: torch.Tensor) -> torch.Tensor:
        t_abs = torch.abs(t)
        result = torch.zeros_like(t)

        mask1 = (t_abs > 1) & (t_abs <= 2)
        mask2 = t_abs <= 1

        result = torch.where(mask1, 6 * (2 - t_abs), result)
        result = torch.where(mask2, -12 + 18 * t_abs, result)
        return result

    @staticmethod
    def _d3phi_dt3(t: torch.Tensor) -> torch.Tensor:
        t_abs = torch.abs(t)
        t_sign = torch.sign(t)
        result = torch.zeros_like(t)

        mask1 = (t_abs > 1) & (t_abs <= 2)
        mask2 = t_abs <= 1

        result = torch.where(mask1, -6 * t_sign, result)
        result = torch.where(mask2, t_sign * 18, result)
        return result

    def eval_spline(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, device=self._device, dtype=self._dtype)
        scalar_input = x.dim() == 0
        if scalar_input:
            x = x.unsqueeze(0)

        l = torch.floor((x - self._a) / self._h).to(torch.long)

        spline_val = torch.zeros_like(x)
        spline_d1 = torch.zeros_like(x)
        spline_d2 = torch.zeros_like(x)
        spline_d3 = torch.zeros_like(x)

        for j in range(4):
            coeff_idx = l + j
            valid_mask = (coeff_idx >= 0) & (coeff_idx <= self._n + 2)

            if torch.any(valid_mask):
                t_vals = ((x[valid_mask] - self._a) / self._h - coeff_idx[valid_mask].to(self._dtype) + 1)
                coeffs = self._coeff[coeff_idx[valid_mask]]

                spline_val[valid_mask] += coeffs * TorchSpline1D._phi(t_vals)
                spline_d1[valid_mask] += coeffs * TorchSpline1D._dphi_dt(t_vals) / self._h
                spline_d2[valid_mask] += coeffs * TorchSpline1D._d2phi_dt2(t_vals) / self._h ** 2
                spline_d3[valid_mask] += coeffs * TorchSpline1D._d3phi_dt3(t_vals) / self._h ** 3

        if scalar_input:
            return spline_val.squeeze(0), spline_d1.squeeze(0), spline_d2.squeeze(0), spline_d3.squeeze(0)
        return spline_val, spline_d1, spline_d2, spline_d3

    @property
    def coeff(self) -> torch.Tensor:
        return self._coeff
