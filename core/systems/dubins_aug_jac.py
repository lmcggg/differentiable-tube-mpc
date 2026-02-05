from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

import torch
from torch import Tensor

from ..barrier import DBaSConfig, relaxed_inverse_barrier_B_alpha
from .dubins import DubinsConfig
from .dubins_obstacles import (
    CircleObstacle,
    grad_h_circle_obstacle,
    grad_h_min_circle_obstacles,
    grad_h_multi_circle_obstacles,
    h_circle_obstacle,
    h_min_circle_obstacles,
    h_multi_circle_obstacles,
)


def _B_inv(z: Tensor, eps: float) -> Tensor:
    return 1.0 / torch.clamp(z, min=eps)


def _dB_inv_dz(z: Tensor, eps: float) -> Tensor:
    zc = torch.clamp(z, min=eps)
    return -1.0 / (zc * zc)


def _dB_relaxed_inv_dz(z: Tensor, *, alpha: Tensor, eps: float) -> Tensor:
    """Derivative of relaxed inverse barrier B_alpha(z) with alpha_eff=max(alpha,eps)."""
    alpha_eff = torch.maximum(alpha, torch.tensor(eps, device=z.device, dtype=z.dtype))
    safe = z >= alpha_eff
    # safe branch: d/dz (1/z) = -1/z^2
    d_safe = _dB_inv_dz(z, eps)
    # unsafe branch: B = 1/a - (z-a)/a^2 + (z-a)^2/a^3
    diff = z - alpha_eff
    d_unsafe = -(1.0 / (alpha_eff**2)) + (2.0 * diff) / (alpha_eff**3)
    return torch.where(safe, d_safe, d_unsafe)

def dubins_f_jac(x: Tensor, u: Tensor, *, cfg: DubinsConfig) -> tuple[Tensor, Tensor]:
    """Analytic Jacobians for Dubins discrete dynamics x_{k+1}=f(x_k,u_k)."""
    dt = cfg.dt
    th = x[2]
    v = u[0]
    c = torch.cos(th)
    s = torch.sin(th)

    A = torch.eye(3, device=x.device, dtype=x.dtype)
    A[0, 2] = -dt * v * s
    A[1, 2] = dt * v * c

    B = torch.zeros(3, 2, device=x.device, dtype=x.dtype)
    B[0, 0] = dt * c
    B[1, 0] = dt * s
    B[2, 1] = dt
    return A, B


def dubins_augmented_jacobian(
    x_hat: Tensor,
    u: Tensor,
    *,
    cfg: DubinsConfig,
    obs: Union[CircleObstacle, list[CircleObstacle]],
    db_cfg: DBaSConfig,
    obs_beta: float = 20.0,
    obs_agg: str = "min",
) -> tuple[Tensor, Tensor]:
    """Analytic Jacobians for safety-embedded Dubins with DBaS.

    State x_hat = [x(3), b]
    Control u = [v, omega]   (note: this is after applying bounds)
    """
    eps = db_cfg.eps
    x = x_hat[:-1]
    b = x_hat[-1]

    # Base dynamics Jacobians
    A3, B3 = dubins_f_jac(x, u, cfg=cfg)

    # Forward state
    x_next = torch.stack(
        [
            x[0] + cfg.dt * u[0] * torch.cos(x[2]),
            x[1] + cfg.dt * u[0] * torch.sin(x[2]),
            x[2] + cfg.dt * u[1],
        ],
        dim=0,
    )

    # Safety function h and gradients (single obstacle or multi-obstacle aggregation)
    if isinstance(obs, list):
        if obs_agg == "smoothmin":
            h_curr = h_multi_circle_obstacles(x, obstacles=obs, beta=obs_beta)
            h_next = h_multi_circle_obstacles(x_next, obstacles=obs, beta=obs_beta)
            dh_curr = grad_h_multi_circle_obstacles(x, obstacles=obs, beta=obs_beta)       # [3]
            dh_next = grad_h_multi_circle_obstacles(x_next, obstacles=obs, beta=obs_beta)  # [3]
        else:
            # Default: exact min + argmin gradient (piecewise smooth)
            h_curr = h_min_circle_obstacles(x, obstacles=obs)
            h_next = h_min_circle_obstacles(x_next, obstacles=obs)
            dh_curr = grad_h_min_circle_obstacles(x, obstacles=obs)       # [3]
            dh_next = grad_h_min_circle_obstacles(x_next, obstacles=obs)  # [3]
    else:
        h_curr = h_circle_obstacle(x, obs=obs)
        h_next = h_circle_obstacle(x_next, obs=obs)
        dh_curr = grad_h_circle_obstacle(x, obs=obs)         # [3]
        dh_next = grad_h_circle_obstacle(x_next, obs=obs)    # [3]

    # Barrier and derivative (inverse barrier, relaxed with alpha possibly 0)
    # Paper uses inverse barrier with alpha=0 and gamma=0 for Dubins.
    alpha = db_cfg.alpha
    gamma = db_cfg.gamma if isinstance(db_cfg.gamma, Tensor) else torch.tensor(db_cfg.gamma, device=x.device, dtype=x.dtype)

    # B(h) and B'(h)
    alpha_t = alpha if isinstance(alpha, Tensor) else torch.tensor(alpha, device=x.device, dtype=x.dtype)
    B_curr = relaxed_inverse_barrier_B_alpha(h_curr, alpha=alpha_t, eps=eps)
    B_next = relaxed_inverse_barrier_B_alpha(h_next, alpha=alpha_t, eps=eps)
    dB_curr = _dB_relaxed_inv_dz(h_curr, alpha=alpha_t, eps=eps)
    dB_next = _dB_relaxed_inv_dz(h_next, alpha=alpha_t, eps=eps)

    # b_next = B_next - gamma*(B_curr - b)  -> ∂b/∂b = gamma
    # ∂b/∂x = dB_next * dh_next^T * ∂x_next/∂x - gamma*dB_curr*dh_curr^T
    # ∂b/∂u = dB_next * dh_next^T * ∂x_next/∂u
    # where dh_next is w.r.t x_next, so chain by A3/B3.
    row_x = (dB_next * dh_next).unsqueeze(0) @ A3 - (gamma * dB_curr * dh_curr).unsqueeze(0)
    row_u = (dB_next * dh_next).unsqueeze(0) @ B3

    # Assemble A,B for x_hat
    A = torch.zeros(4, 4, device=x.device, dtype=x.dtype)
    B = torch.zeros(4, 2, device=x.device, dtype=x.dtype)
    A[:3, :3] = A3
    B[:3, :] = B3
    A[3, :3] = row_x.squeeze(0)
    A[3, 3] = gamma
    B[3, :] = row_u.squeeze(0)
    return A, B

