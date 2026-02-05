from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class DubinsConfig:
    dt: float = 0.01
    v_max: float = 10.0
    omega_max: float = float(torch.pi)

    # Disturbance bounds (uniform) for [x,y,theta]
    w_low: Tuple[float, float, float] = (-0.05, -0.05, -0.05)
    w_high: Tuple[float, float, float] = (0.05, 0.05, 0.05)

    # Target
    x_target: Tuple[float, float, float] = (10.0, 10.0, float(torch.pi / 4))


def dubins_step(x: Tensor, u: Tensor, *, cfg: DubinsConfig) -> Tensor:
    """Discrete Dubins vehicle dynamics.

    State:  [x, y, theta]
    Input:  [v, omega]
    """
    unbatched = x.ndim == 1
    if unbatched:
        x = x.unsqueeze(0)
    if u.ndim == 1:
        u = u.unsqueeze(0)

    dt = cfg.dt
    px, py, th = x[:, 0], x[:, 1], x[:, 2]
    v, om = u[:, 0], u[:, 1]
    px_n = px + dt * v * torch.cos(th)
    py_n = py + dt * v * torch.sin(th)
    th_n = th + dt * om
    out = torch.stack([px_n, py_n, th_n], dim=-1)
    return out.squeeze(0) if unbatched else out


def clamp_control(u: Tensor, *, cfg: DubinsConfig) -> Tensor:
    """Hard box constraints matching the table."""
    unbatched = u.ndim == 1
    if unbatched:
        u = u.unsqueeze(0)
    u0 = torch.clamp(u[:, 0], -cfg.v_max, cfg.v_max)
    u1 = torch.clamp(u[:, 1], -cfg.omega_max, cfg.omega_max)
    out = torch.stack([u0, u1], dim=-1)
    return out.squeeze(0) if unbatched else out


def sample_disturbance(x: Tensor, *, cfg: DubinsConfig) -> Tensor:
    """Uniform disturbance w_t in W (broadcast to batch)."""
    unbatched = x.ndim == 1
    if unbatched:
        x = x.unsqueeze(0)
    low = torch.tensor(cfg.w_low, device=x.device, dtype=x.dtype)
    high = torch.tensor(cfg.w_high, device=x.device, dtype=x.dtype)
    w = low + (high - low) * torch.rand_like(x)
    return w.squeeze(0) if unbatched else w


def default_safe_h_no_obstacles(x: Tensor) -> Tensor:
    """Placeholder safety function h(x)>0.

    For Dubins in the user's table, explicit obstacles are not specified, so h(x) is left to config.
    This returns a positive constant (always safe) unless overridden by experiment config.
    """
    if x.ndim == 1:
        return torch.ones((), device=x.device, dtype=x.dtype)
    return torch.ones(x.shape[0], device=x.device, dtype=x.dtype)

