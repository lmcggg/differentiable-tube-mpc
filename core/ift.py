from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class IFTInputs:
    """All quantities needed for the Theorem-5 style gradient accumulation."""

    # Optimal trajectory of the lower-level problem
    X: Tensor              # [N+1, nx]
    V: Tensor              # [N, nu] (decision variable in our implementation)
    delta_X: Tensor        # [N+1, nx]
    delta_V: Tensor        # [N, nu]
    delta_lambda: Tensor   # [N+1, nx]


def _grad_wrt_tensors(scalar: Tensor, tensors: Sequence[Tensor]) -> List[Optional[Tensor]]:
    """Compute gradients of scalar w.r.t. tensors that require grad."""
    idx = [i for i, t in enumerate(tensors) if t.requires_grad]
    if not idx:
        return [None for _ in tensors]
    inputs = [tensors[i] for i in idx]
    grads_sub = torch.autograd.grad(scalar, inputs, retain_graph=True, allow_unused=True)
    out: List[Optional[Tensor]] = [None for _ in tensors]
    for i, g in zip(idx, grads_sub):
        out[i] = g
    return out


def ift_gradient(
    *,
    inputs: IFTInputs,
    theta_tensors: Sequence[Tensor],
    xi_fn: Callable[[], Tensor],
    f_fn: Callable[[Tensor, Tensor], Tensor],
    stage_cost_fn: Callable[[Tensor, Tensor, int], Tensor],
    terminal_cost_fn: Callable[[Tensor], Tensor],
) -> List[Optional[Tensor]]:
    """Compute ∇_θ L using the DDP-structured δz and the user's accumulation formula:

      ∇_θ L = ξ_θ^T δλ_0
            + Σ_{k=0}^{N-1} ( ℒ_{θx}^{(k)} δx_k + ℒ_{θu}^{(k)} δu_k + f_{θ_k}^T δλ_{k+1} )
            + φ_{θx} δx_N

    Implemented via autograd VJP tricks without explicitly forming mixed Hessians:
      - ℒ_{θx} δx is computed as ∂/∂θ [ (∂ℓ/∂x)·δx ]
      - ℒ_{θu} δu is computed as ∂/∂θ [ (∂ℓ/∂u)·δu ]
      - f_θ^T δλ is computed as ∂/∂θ [ δλ^T f(x,u,θ) ]
      - ξ_θ^T δλ0 is computed as ∂/∂θ [ δλ0^T ξ(θ) ] where ξ is provided by xi_fn().
      - φ_{θx} δx_N as ∂/∂θ [ (∂φ/∂x)·δx_N ].
    """
    X, V = inputs.X, inputs.V
    dX, dV, dlmb = inputs.delta_X, inputs.delta_V, inputs.delta_lambda
    N = V.shape[0]
    device, dtype = X.device, X.dtype

    total = torch.zeros((), device=device, dtype=dtype)

    # ξ_θ^T δλ0  -> grad wrt theta of (δλ0^T ξ(θ))
    xi = xi_fn()
    total = total + (dlmb[0] * xi).sum()

    for k in range(N):
        # Clone to avoid any view/in-place versioning issues from AsStrided.
        xk_req = X[k].detach().clone().requires_grad_(True)
        vk_req = V[k].detach().clone().requires_grad_(True)

        # ℒ_{θx}^{(k)} δx_k term
        l = stage_cost_fn(xk_req, vk_req, k)
        l_x = torch.autograd.grad(l, xk_req, create_graph=True)[0]
        total = total + (l_x * dX[k]).sum()

        # ℒ_{θu}^{(k)} δu_k term (here u is v)
        l_u = torch.autograd.grad(l, vk_req, create_graph=True)[0]
        total = total + (l_u * dV[k]).sum()

        # f_{θ_k}^T δλ_{k+1} term
        x_next = f_fn(xk_req, vk_req)
        total = total + (dlmb[k + 1] * x_next).sum()

    # φ_{θx} δx_N term
    xN = X[N].detach().clone().requires_grad_(True)
    phi = terminal_cost_fn(xN)
    phi_x = torch.autograd.grad(phi, xN, create_graph=True)[0]
    total = total + (phi_x * dX[N]).sum()

    return _grad_wrt_tensors(total, theta_tensors)
