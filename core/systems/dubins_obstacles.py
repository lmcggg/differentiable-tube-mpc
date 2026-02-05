from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class CircleObstacle:
    center: Tuple[float, float]
    radius: float


def h_circle_obstacle(x: Tensor, *, obs: CircleObstacle) -> Tensor:
    """Continuously differentiable safety function for a circular obstacle.

    Safe set: h(x) > 0 where
      h(x) = ||p - c||^2 - r^2
    with p=(x,y).
    """
    if x.ndim == 1:
        px, py = x[0], x[1]
    else:
        px, py = x[:, 0], x[:, 1]
    cx, cy = obs.center
    dx = px - cx
    dy = py - cy
    return dx * dx + dy * dy - (obs.radius ** 2)


def grad_h_circle_obstacle(x: Tensor, *, obs: CircleObstacle) -> Tensor:
    """Gradient of h wrt state x=[x,y,theta] (theta derivative is 0)."""
    if x.ndim != 1:
        raise ValueError("Expected unbatched state for analytic gradient.")
    cx, cy = obs.center
    return torch.tensor([2.0 * (x[0] - cx), 2.0 * (x[1] - cy), 0.0], device=x.device, dtype=x.dtype)


def h_multi_circle_obstacles(
    x: Tensor,
    *,
    obstacles: list[CircleObstacle],
    beta: float = 20.0,
) -> Tensor:
    """Differentiable aggregation of multiple circle obstacle constraints.

    Each obstacle induces h_i(x) = ||p-c_i||^2 - r_i^2, safe if h_i(x) > 0.
    The safe set is the intersection over obstacles. We need a single smooth h(x),
    so we use a smooth-min approximation:

      h(x) ≈ min_i h_i(x)  via  h = -(1/beta) * log sum_i exp(-beta * h_i)

    Larger beta -> closer to min, but less smooth.
    """
    if len(obstacles) == 0:
        # No obstacles => always safe.
        if x.ndim == 1:
            return torch.ones((), device=x.device, dtype=x.dtype)
        return torch.ones(x.shape[0], device=x.device, dtype=x.dtype)

    hs = [h_circle_obstacle(x, obs=o) for o in obstacles]  # each scalar or [B]
    H = torch.stack(hs, dim=0)  # [M] or [M,B]
    # stable log-sum-exp
    z = (-beta) * H
    z_max = torch.max(z, dim=0).values
    lse = z_max + torch.log(torch.sum(torch.exp(z - z_max), dim=0))
    return -(1.0 / beta) * lse


def grad_h_multi_circle_obstacles(
    x: Tensor,
    *,
    obstacles: list[CircleObstacle],
    beta: float = 20.0,
) -> Tensor:
    """Gradient of smooth-min multi-obstacle h(x) for unbatched x=[x,y,theta]."""
    if x.ndim != 1:
        raise ValueError("Expected unbatched state for analytic gradient.")
    if len(obstacles) == 0:
        return torch.zeros(3, device=x.device, dtype=x.dtype)

    hs = torch.stack([h_circle_obstacle(x, obs=o) for o in obstacles], dim=0)  # [M]
    grads = torch.stack([grad_h_circle_obstacle(x, obs=o) for o in obstacles], dim=0)  # [M,3]

    # weights for smooth-min: w_i = softmax(-beta * h_i)
    z = (-beta) * hs
    z = z - torch.max(z)
    w = torch.exp(z)
    w = w / torch.sum(w)
    return torch.sum(w.view(-1, 1) * grads, dim=0)


def h_min_circle_obstacles(x: Tensor, *, obstacles: list[CircleObstacle]) -> Tensor:
    """Non-smooth (exact) aggregation: h(x) = min_i h_i(x).

    This avoids the 'merged obstacles' effect that smooth-min can introduce.
    """
    if len(obstacles) == 0:
        if x.ndim == 1:
            return torch.ones((), device=x.device, dtype=x.dtype)
        return torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
    hs = [h_circle_obstacle(x, obs=o) for o in obstacles]
    H = torch.stack(hs, dim=0)
    return torch.min(H, dim=0).values


def grad_h_min_circle_obstacles(x: Tensor, *, obstacles: list[CircleObstacle]) -> Tensor:
    """Subgradient of h(x)=min_i h_i(x) using the argmin obstacle's gradient."""
    if x.ndim != 1:
        raise ValueError("Expected unbatched state for analytic gradient.")
    if len(obstacles) == 0:
        return torch.zeros(3, device=x.device, dtype=x.dtype)
    hs = torch.stack([h_circle_obstacle(x, obs=o) for o in obstacles], dim=0)  # [M]
    i = int(torch.argmin(hs).item())
    return grad_h_circle_obstacle(x, obs=obstacles[i])

