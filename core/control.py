from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class BoxTanhControl:
    """Differentiable box constraints via tanh parameterization.

    Decision variable: v ∈ R^nu (unconstrained)
    Control: u = u_max * tanh(v)  (for symmetric bounds [-u_max, u_max])
    or u = u_min + (u_max-u_min)*(tanh(v)+1)/2 for general box.
    """

    u_min: Tensor  # [nu]
    u_max: Tensor  # [nu]

    def u(self, v: Tensor) -> Tensor:
        # Supports shape [..., nu]
        u_min = self.u_min.to(device=v.device, dtype=v.dtype)
        u_max = self.u_max.to(device=v.device, dtype=v.dtype)
        # Map tanh(v) ∈ (-1,1) to [u_min,u_max]
        return u_min + (u_max - u_min) * (torch.tanh(v) + 1.0) * 0.5

    def du_dv_diag(self, v: Tensor) -> Tensor:
        """Elementwise derivative du/dv (diagonal Jacobian) for the tanh box map."""
        u_min = self.u_min.to(device=v.device, dtype=v.dtype)
        u_max = self.u_max.to(device=v.device, dtype=v.dtype)
        scale = (u_max - u_min) * 0.5
        sech2 = 1.0 - torch.tanh(v) ** 2
        return scale * sech2


@dataclass(frozen=True)
class BoxClampControl:
    """Hard box constraints u ∈ [u_min, u_max] with active-set helpers.

    This matches the paper's control-limited DDP / active-set discussion:
      - Solve with u as the decision variable (constrained)
      - For DOC, treat dimensions at bounds as active, enforcing δu_i = 0
    
    Paper (Appendix G, Section "Incorporating Control Constraints"):
      "This is equivalent to the condition δu_k^(i) = 0 if u_k^(i) ∈ {u_min, u_max}"
      The active-set method partitions controls into free and active sets,
      solving reduced KKT conditions only for free variables.
    
    Implementation:
      - iLQR solver: forward pass clamps u, backward pass computes full gain K
      - DOC sensitivity: backward pass uses _solve_reduced (active dims zeroed),
        forward pass enforces δu[active] = 0
    """

    u_min: Tensor  # [nu]
    u_max: Tensor  # [nu]
    active_tol: float = 1e-8  # Numerical tolerance for "at boundary" detection

    def clamp(self, u: Tensor) -> Tensor:
        u_min = self.u_min.to(device=u.device, dtype=u.dtype)
        u_max = self.u_max.to(device=u.device, dtype=u.dtype)
        return torch.clamp(u, min=u_min, max=u_max)

    def active_mask(self, u: Tensor) -> Tensor:
        """Return boolean mask [...,nu] where component is (numerically) active."""
        u_min = self.u_min.to(device=u.device, dtype=u.dtype)
        u_max = self.u_max.to(device=u.device, dtype=u.dtype)
        return (u <= (u_min + self.active_tol)) | (u >= (u_max - self.active_tol))

