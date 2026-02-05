from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import torch
from torch import Tensor


class Dynamics(Protocol):
    """Discrete-time dynamics x_{k+1}=f(x_k,u_k)."""

    def __call__(self, x: Tensor, u: Tensor) -> Tensor: ...


class StageCost(Protocol):
    """Stage cost ℓ(x_k,u_k,theta) (or tracking ℓ(x,u,x̄,ū,theta))."""

    def __call__(self, x: Tensor, u: Tensor, **kwargs) -> Tensor: ...


class TerminalCost(Protocol):
    """Terminal cost φ(x_N,theta) (or tracking φ(x_N,x̄_N,theta))."""

    def __call__(self, xN: Tensor, **kwargs) -> Tensor: ...


@dataclass(frozen=True)
class OCPConfig:
    horizon: int
    nx: int
    nu: int


def rollout_dynamics(x0: Tensor, U: Tensor, *, f: Dynamics) -> Tensor:
    """Rollout equality dynamics for a control sequence.

    Args:
      x0: [B,nx] or [nx]
      U:  [B,N,nu] or [N,nu]
    Returns:
      X:  [B,N+1,nx] or [N+1,nx] matching batch mode of inputs
    """
    batched = x0.ndim == 2
    if not batched:
        x0_b = x0.unsqueeze(0)
        U_b = U.unsqueeze(0)
    else:
        x0_b = x0
        U_b = U

    B, N, nx = U_b.shape[0], U_b.shape[1], x0_b.shape[-1]
    X = torch.empty(B, N + 1, nx, device=x0_b.device, dtype=x0_b.dtype)
    X[:, 0] = x0_b
    x = x0_b
    for k in range(N):
        x = f(x, U_b[:, k])
        X[:, k + 1] = x
    return X if batched else X.squeeze(0)


def total_cost(
    *,
    X: Tensor,
    U: Tensor,
    stage_cost: Callable[..., Tensor],
    terminal_cost: Callable[..., Tensor],
    stage_kwargs: dict,
    terminal_kwargs: dict,
) -> Tensor:
    """Compute sum_{k=0}^{N-1} ℓ_k + φ_N (batched)."""
    batched = X.ndim == 3
    if not batched:
        Xb = X.unsqueeze(0)
        Ub = U.unsqueeze(0)
    else:
        Xb = X
        Ub = U

    B, N = Ub.shape[0], Ub.shape[1]
    J = torch.zeros(B, device=Xb.device, dtype=Xb.dtype)
    for k in range(N):
        J = J + stage_cost(Xb[:, k], Ub[:, k], **stage_kwargs)
    J = J + terminal_cost(Xb[:, N], **terminal_kwargs)
    return J if batched else J.squeeze(0)

