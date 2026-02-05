from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class NominalWeights:
    """Nominal MPC weights (diagonal quadratic + barrier penalty)."""

    Q_diag: Tensor   # [nx]
    R_diag: Tensor   # [nu]
    Qf_diag: Tensor  # [nx]
    q_b: Tensor      # scalar


@dataclass
class AuxiliaryWeights:
    """Auxiliary MPC tracking weights (diagonal quadratic + barrier penalty)."""

    Q_diag: Tensor   # [nx]
    R_diag: Tensor   # [nu]
    Qf_diag: Tensor  # [nx]
    q_b: Tensor      # scalar


def nominal_stage_quadratic(
    x_hat: Tensor,
    u: Tensor,
    *,
    target_x: Tensor,
    w: NominalWeights,
) -> Tensor:
    """Nominal stage cost:
      ℓ̄ = ||x - x_target||_{Q}^2 + ||u||_{R}^2 + q_b * b^2
    where x_hat = [x, b].
    """
    x = x_hat[..., :-1]
    b = x_hat[..., -1]
    dx = x - target_x
    return (w.Q_diag * dx * dx).sum(dim=-1) + (w.R_diag * u * u).sum(dim=-1) + w.q_b * (b * b)


def nominal_terminal_quadratic(x_hat_N: Tensor, *, target_x: Tensor, w: NominalWeights) -> Tensor:
    xN = x_hat_N[..., :-1]
    dxN = xN - target_x
    return (w.Qf_diag * dxN * dxN).sum(dim=-1)


def auxiliary_stage_quadratic(
    x_hat: Tensor,
    u: Tensor,
    *,
    x_ref: Tensor,
    u_ref: Tensor,
    w: AuxiliaryWeights,
) -> Tensor:
    """Auxiliary tracking stage cost:
      ℓ = ||x - x̄||_Q^2 + ||u - ū||_R^2 + q_b * b^2
    where x_hat = [x, b].
    """
    x = x_hat[..., :-1]
    b = x_hat[..., -1]
    dx = x - x_ref
    du = u - u_ref
    return (w.Q_diag * dx * dx).sum(dim=-1) + (w.R_diag * du * du).sum(dim=-1) + w.q_b * (b * b)


def auxiliary_terminal_quadratic(x_hat_N: Tensor, *, x_ref_N: Tensor, w: AuxiliaryWeights) -> Tensor:
    xN = x_hat_N[..., :-1]
    dxN = xN - x_ref_N
    return (w.Qf_diag * dxN * dxN).sum(dim=-1)


def quadratic_derivatives_nominal(
    x_hat: Tensor, u: Tensor, *, target_x: Tensor, w: NominalWeights
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return (l_x, l_u, l_xx, l_uu, l_ux) for one time step."""
    x = x_hat[:-1]
    b = x_hat[-1]
    dx = x - target_x

    l_x_state = 2.0 * w.Q_diag * dx
    l_x_b = 2.0 * w.q_b * b
    l_x = torch.cat([l_x_state, l_x_b.view(1)], dim=0)
    l_u = 2.0 * w.R_diag * u

    l_xx = torch.diag(torch.cat([2.0 * w.Q_diag, (2.0 * w.q_b).view(1)], dim=0))
    l_uu = torch.diag(2.0 * w.R_diag)
    l_ux = torch.zeros(u.numel(), x_hat.numel(), device=x_hat.device, dtype=x_hat.dtype)
    return l_x, l_u, l_xx, l_uu, l_ux


def quadratic_derivatives_terminal_nominal(
    x_hat_N: Tensor, *, target_x: Tensor, w: NominalWeights
) -> tuple[Tensor, Tensor]:
    xN = x_hat_N[:-1]
    dxN = xN - target_x
    phi_x_state = 2.0 * w.Qf_diag * dxN
    phi_x = torch.cat([phi_x_state, torch.zeros(1, device=x_hat_N.device, dtype=x_hat_N.dtype)], dim=0)
    phi_xx = torch.diag(torch.cat([2.0 * w.Qf_diag, torch.zeros(1, device=x_hat_N.device, dtype=x_hat_N.dtype)], dim=0))
    return phi_x, phi_xx


def quadratic_derivatives_auxiliary(
    x_hat: Tensor,
    u: Tensor,
    *,
    x_ref: Tensor,
    u_ref: Tensor,
    w: AuxiliaryWeights,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    x = x_hat[:-1]
    b = x_hat[-1]
    dx = x - x_ref
    du = u - u_ref

    l_x_state = 2.0 * w.Q_diag * dx
    l_x_b = 2.0 * w.q_b * b
    l_x = torch.cat([l_x_state, l_x_b.view(1)], dim=0)
    l_u = 2.0 * w.R_diag * du

    l_xx = torch.diag(torch.cat([2.0 * w.Q_diag, (2.0 * w.q_b).view(1)], dim=0))
    l_uu = torch.diag(2.0 * w.R_diag)
    l_ux = torch.zeros(u.numel(), x_hat.numel(), device=x_hat.device, dtype=x_hat.dtype)
    return l_x, l_u, l_xx, l_uu, l_ux


def quadratic_derivatives_terminal_auxiliary(
    x_hat_N: Tensor, *, x_ref_N: Tensor, w: AuxiliaryWeights
) -> tuple[Tensor, Tensor]:
    xN = x_hat_N[:-1]
    dxN = xN - x_ref_N
    phi_x_state = 2.0 * w.Qf_diag * dxN
    phi_x = torch.cat([phi_x_state, torch.zeros(1, device=x_hat_N.device, dtype=x_hat_N.dtype)], dim=0)
    phi_xx = torch.diag(torch.cat([2.0 * w.Qf_diag, torch.zeros(1, device=x_hat_N.device, dtype=x_hat_N.dtype)], dim=0))
    return phi_x, phi_xx

