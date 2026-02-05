from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

from .control import BoxTanhControl


def _diag(v: Tensor) -> Tensor:
    return torch.diag(v)


def _d2u_dv2_diag(ctrl: BoxTanhControl, v: Tensor) -> Tensor:
    """Second derivative d²u/dv² (diagonal) for BoxTanhControl."""
    u_min = ctrl.u_min.to(device=v.device, dtype=v.dtype)
    u_max = ctrl.u_max.to(device=v.device, dtype=v.dtype)
    scale = (u_max - u_min) * 0.5
    th = torch.tanh(v)
    sech2 = 1.0 - th**2
    # d/dv sech2 = -2*tanh(v)*sech2
    return scale * (-2.0 * th * sech2)


def nominal_cost_derivs(
    *,
    x_hat: Tensor,   # [4] = [x(3), b]
    v: Tensor,       # [2]
    target: Tensor,  # [3]
    Q: Tensor,       # [3]
    R: Tensor,       # [2]
    qb: Tensor,      # []
    ctrl: BoxTanhControl,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Exact derivatives for nominal stage cost with u=tanh-box(v)."""
    x = x_hat[:-1]
    b = x_hat[-1]
    dx = x - target

    u = ctrl.u(v)
    du_dv = ctrl.du_dv_diag(v)
    d2u_dv2 = _d2u_dv2_diag(ctrl, v)

    # Gradients
    l_x = torch.cat([2.0 * Q * dx, (2.0 * qb * b).view(1)], dim=0)
    l_v = 2.0 * R * u * du_dv

    # Hessians
    l_xx = torch.diag(torch.cat([2.0 * Q, (2.0 * qb).view(1)], dim=0))
    l_vv_diag = 2.0 * R * ((du_dv**2) + u * d2u_dv2)
    l_vv = torch.diag(l_vv_diag)
    l_vx = torch.zeros(v.numel(), x_hat.numel(), device=x_hat.device, dtype=x_hat.dtype)
    return l_x, l_v, l_xx, l_vv, l_vx


def nominal_cost_derivs_u(
    *,
    x_hat: Tensor,   # [4] = [x(3), b]
    u: Tensor,       # [2] (already within bounds)
    target: Tensor,  # [3]
    Q: Tensor,       # [3]
    R: Tensor,       # [2]
    qb: Tensor,      # []
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Exact derivatives for nominal stage cost with decision variable u (paper's box-constrained form)."""
    x = x_hat[:-1]
    b = x_hat[-1]
    dx = x - target
    l_x = torch.cat([2.0 * Q * dx, (2.0 * qb * b).view(1)], dim=0)
    l_u = 2.0 * R * u
    l_xx = torch.diag(torch.cat([2.0 * Q, (2.0 * qb).view(1)], dim=0))
    l_uu = torch.diag(2.0 * R)
    l_ux = torch.zeros(u.numel(), x_hat.numel(), device=x_hat.device, dtype=x_hat.dtype)
    return l_x, l_u, l_xx, l_uu, l_ux


def auxiliary_cost_derivs(
    *,
    x_hat: Tensor,    # [4]
    v: Tensor,        # [2]
    x_ref: Tensor,    # [3]
    u_ref: Tensor,    # [2]
    Q: Tensor,        # [3]
    R: Tensor,        # [2]
    qb: Tensor,       # []
    ctrl: BoxTanhControl,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Exact derivatives for auxiliary tracking stage cost with u=tanh-box(v)."""
    x = x_hat[:-1]
    b = x_hat[-1]
    dx = x - x_ref

    u = ctrl.u(v)
    du = u - u_ref
    du_dv = ctrl.du_dv_diag(v)
    d2u_dv2 = _d2u_dv2_diag(ctrl, v)

    l_x = torch.cat([2.0 * Q * dx, (2.0 * qb * b).view(1)], dim=0)
    l_v = 2.0 * R * du * du_dv

    l_xx = torch.diag(torch.cat([2.0 * Q, (2.0 * qb).view(1)], dim=0))
    l_vv_diag = 2.0 * R * ((du_dv**2) + du * d2u_dv2)
    l_vv = torch.diag(l_vv_diag)
    l_vx = torch.zeros(v.numel(), x_hat.numel(), device=x_hat.device, dtype=x_hat.dtype)
    return l_x, l_v, l_xx, l_vv, l_vx


def auxiliary_cost_derivs_u(
    *,
    x_hat: Tensor,    # [4]
    u: Tensor,        # [2] (already within bounds)
    x_ref: Tensor,    # [3]
    u_ref: Tensor,    # [2]
    Q: Tensor,        # [3]
    R: Tensor,        # [2]
    qb: Tensor,       # []
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Exact derivatives for auxiliary tracking stage cost with decision variable u (paper's box-constrained form)."""
    x = x_hat[:-1]
    b = x_hat[-1]
    dx = x - x_ref
    du = u - u_ref
    l_x = torch.cat([2.0 * Q * dx, (2.0 * qb * b).view(1)], dim=0)
    l_u = 2.0 * R * du
    l_xx = torch.diag(torch.cat([2.0 * Q, (2.0 * qb).view(1)], dim=0))
    l_uu = torch.diag(2.0 * R)
    l_ux = torch.zeros(u.numel(), x_hat.numel(), device=x_hat.device, dtype=x_hat.dtype)
    return l_x, l_u, l_xx, l_uu, l_ux


def nominal_terminal_derivs(*, x_hat_N: Tensor, target: Tensor, Qf: Tensor) -> tuple[Tensor, Tensor]:
    xN = x_hat_N[:-1]
    dxN = xN - target
    phi_x = torch.cat([2.0 * Qf * dxN, torch.zeros(1, device=x_hat_N.device, dtype=x_hat_N.dtype)], dim=0)
    phi_xx = torch.diag(torch.cat([2.0 * Qf, torch.zeros(1, device=x_hat_N.device, dtype=x_hat_N.dtype)], dim=0))
    return phi_x, phi_xx


def auxiliary_terminal_derivs(*, x_hat_N: Tensor, x_ref_N: Tensor, Qf: Tensor) -> tuple[Tensor, Tensor]:
    xN = x_hat_N[:-1]
    dxN = xN - x_ref_N
    phi_x = torch.cat([2.0 * Qf * dxN, torch.zeros(1, device=x_hat_N.device, dtype=x_hat_N.dtype)], dim=0)
    phi_xx = torch.diag(torch.cat([2.0 * Qf, torch.zeros(1, device=x_hat_N.device, dtype=x_hat_N.dtype)], dim=0))
    return phi_x, phi_xx

