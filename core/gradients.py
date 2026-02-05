from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .costs import AuxiliaryWeights, NominalWeights


@dataclass
class NominalGrads:
    Q_diag: Tensor
    R_diag: Tensor
    Qf_diag: Tensor
    q_b: Tensor


@dataclass
class AuxiliaryGrads:
    Q_diag: Tensor
    R_diag: Tensor
    Qf_diag: Tensor
    q_b: Tensor


def grads_nominal_from_deltas(
    *,
    X_hat: Tensor,   # [N+1, nx+1]
    U: Tensor,       # [N, nu]
    dX: Tensor,      # [N+1, nx+1]
    dU: Tensor,      # [N, nu]
    target_x: Tensor,  # [nx]
) -> NominalGrads:
    """Compute ∇_{θ̄} L via the provided accumulation form for weight-parameterized quadratic costs.

    This matches the user's final '梯度累积' structure specialized to:
      ℓ̄ = Σ_i Q_i (x_i-x*_i)^2 + Σ_j R_j u_j^2 + q_b b^2
      φ̄ = Σ_i Qf_i (xN_i-x*_i)^2
    """
    x = X_hat[:, :-1]
    b = X_hat[:, -1]
    dx = x - target_x.view(1, -1)

    d_x = dX[:, :-1]
    d_b = dX[:, -1]

    # Stage contributions
    grad_Q = (2.0 * dx[:-1] * d_x[:-1]).sum(dim=0)
    grad_R = (2.0 * U * dU).sum(dim=0)
    grad_qb = (2.0 * b[:-1] * d_b[:-1]).sum()

    # Terminal contributions
    dxN = dx[-1]
    d_xN = d_x[-1]
    grad_Qf = 2.0 * dxN * d_xN

    return NominalGrads(Q_diag=grad_Q, R_diag=grad_R, Qf_diag=grad_Qf, q_b=grad_qb)


def grads_auxiliary_from_deltas(
    *,
    X_hat: Tensor,     # [N+1, nx+1]
    U: Tensor,         # [N, nu]
    dX: Tensor,        # [N+1, nx+1]
    dU: Tensor,        # [N, nu]
    X_ref: Tensor,     # [N+1, nx] (nominal x trajectory)
    U_ref: Tensor,     # [N, nu]   (nominal u trajectory)
) -> AuxiliaryGrads:
    """Compute ∇_{θ} L for tracking quadratic costs."""
    x = X_hat[:, :-1]
    b = X_hat[:, -1]
    dx = x - X_ref
    du = U - U_ref

    d_x = dX[:, :-1]
    d_b = dX[:, -1]

    grad_Q = (2.0 * dx[:-1] * d_x[:-1]).sum(dim=0)
    grad_R = (2.0 * du * dU).sum(dim=0)
    grad_qb = (2.0 * b[:-1] * d_b[:-1]).sum()

    dxN = dx[-1]
    d_xN = d_x[-1]
    grad_Qf = 2.0 * dxN * d_xN

    return AuxiliaryGrads(Q_diag=grad_Q, R_diag=grad_R, Qf_diag=grad_Qf, q_b=grad_qb)


@torch.no_grad()
def apply_nominal_update(w: NominalWeights, g: NominalGrads, *, lr: float) -> NominalWeights:
    return NominalWeights(
        Q_diag=w.Q_diag - lr * g.Q_diag,
        R_diag=w.R_diag - lr * g.R_diag,
        Qf_diag=w.Qf_diag - lr * g.Qf_diag,
        q_b=w.q_b - lr * g.q_b,
    )


@torch.no_grad()
def apply_auxiliary_update(w: AuxiliaryWeights, g: AuxiliaryGrads, *, lr: float) -> AuxiliaryWeights:
    return AuxiliaryWeights(
        Q_diag=w.Q_diag - lr * g.Q_diag,
        R_diag=w.R_diag - lr * g.R_diag,
        Qf_diag=w.Qf_diag - lr * g.Qf_diag,
        q_b=w.q_b - lr * g.q_b,
    )

