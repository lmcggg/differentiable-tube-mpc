from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Union

import torch
from torch import Tensor


BarrierType = Literal["inverse", "log"]


ScalarLike = Union[float, Tensor]


@dataclass(frozen=True)
class DBaSConfig:
    """Discrete Barrier State (DBaS) configuration.

    Matches the user's formulas:
      - b_{k+1} = B(h(f(x_k,u_k))) - gamma * (B(h(x_k)) - b_k)
      - gamma ∈ [-1, 1]
      - relaxed inverse barrier B_alpha
    """

    barrier_type: BarrierType = "inverse"
    alpha: ScalarLike = 0.1
    gamma: ScalarLike = 0.0

    # Numerical safety for log/inverse near zero.
    # NOTE: Very small eps (e.g. 1e-12) can overflow in float32 when used inside
    # inverse-barrier derivatives (1/eps^2) and subsequent quadratic forms.
    eps: float = 1e-6


def relaxed_inverse_barrier_B_alpha(zeta: Tensor, *, alpha: ScalarLike, eps: float = 1e-12) -> Tensor:
    """Relaxed inverse barrier B_alpha(zeta) exactly per the provided piecewise definition.

    B_alpha(zeta) =
        1/zeta                                    if zeta >= alpha
        1/alpha - (zeta-alpha)/alpha^2 + (zeta-alpha)^2/alpha^3   otherwise
    """
    if isinstance(alpha, (float, int)) and alpha < 0:
        raise ValueError("alpha must be >= 0")

    # IMPORTANT:
    # The paper's relaxed inverse barrier is defined for alpha>0 and provides a smooth extension
    # outside the constraint set (zeta < alpha). In experiments they sometimes set alpha=0 to denote
    # the unrelaxed barrier, but numerically we still need a smooth extension to avoid zero
    # gradients from clamping when zeta <= 0. We therefore use alpha_eff = max(alpha, eps).
    alpha_t = alpha if isinstance(alpha, Tensor) else torch.tensor(alpha, device=zeta.device, dtype=zeta.dtype)
    alpha_eff = torch.maximum(alpha_t, torch.tensor(eps, device=zeta.device, dtype=zeta.dtype))

    safe = zeta >= alpha_eff
    B_safe = 1.0 / torch.clamp(zeta, min=eps)

    diff = zeta - alpha_eff
    B_unsafe = (1.0 / alpha_eff) - diff / (alpha_eff**2) + (diff**2) / (alpha_eff**3)
    return torch.where(safe, B_safe, B_unsafe)


def barrier_B(zeta: Tensor, *, barrier_type: BarrierType, eps: float = 1e-12) -> Tensor:
    """Barrier function B(zeta).

    - inverse: B(zeta)=1/zeta
    - log:     B(zeta)=-log(zeta)
    """
    if barrier_type == "inverse":
        return 1.0 / torch.clamp(zeta, min=eps)
    if barrier_type == "log":
        return -torch.log(torch.clamp(zeta, min=eps))
    raise ValueError(f"Unknown barrier_type: {barrier_type}")


def dbas_step(
    *,
    x_k: Tensor,
    u_k: Tensor,
    b_k: Tensor,
    f: Callable[[Tensor, Tensor], Tensor],
    h: Callable[[Tensor], Tensor],
    cfg: DBaSConfig,
) -> tuple[Tensor, Tensor]:
    """One DBaS-augmented step.

    Returns:
      - x_{k+1} = f(x_k, u_k)
      - b_{k+1} defined by user's formula
    """
    if isinstance(cfg.gamma, (float, int)) and not (-1.0 <= cfg.gamma <= 1.0):
        raise ValueError("gamma must be in [-1, 1]")

    x_next = f(x_k, u_k)
    h_next = h(x_next)
    h_curr = h(x_k)

    if cfg.barrier_type == "inverse":
        B_next = relaxed_inverse_barrier_B_alpha(h_next, alpha=cfg.alpha, eps=cfg.eps)
        B_curr = relaxed_inverse_barrier_B_alpha(h_curr, alpha=cfg.alpha, eps=cfg.eps)
    else:
        # The paper text also mentions log barrier; relaxed form is only specified for inverse,
        # so for log we use the exact log barrier B(zeta)=-log(zeta).
        B_next = barrier_B(h_next, barrier_type="log", eps=cfg.eps)
        B_curr = barrier_B(h_curr, barrier_type="log", eps=cfg.eps)

    gamma_t = cfg.gamma if isinstance(cfg.gamma, Tensor) else torch.tensor(cfg.gamma, device=B_next.device, dtype=B_next.dtype)
    b_next = B_next - gamma_t * (B_curr - b_k)
    return x_next, b_next


def dbas_init_b0(x0: Tensor, *, h: Callable[[Tensor], Tensor], cfg: DBaSConfig) -> Tensor:
    """Initialize b_0 from x_0.

    For inverse barrier, we use B_alpha(h(x0)) (consistent with the augmentation idea).
    For log barrier, we use B(h(x0))=-log(h(x0)).
    """
    h0 = h(x0)
    if cfg.barrier_type == "inverse":
        return relaxed_inverse_barrier_B_alpha(h0, alpha=cfg.alpha, eps=cfg.eps)
    return barrier_B(h0, barrier_type="log", eps=cfg.eps)

