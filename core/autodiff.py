from __future__ import annotations

from typing import Callable, Tuple

import torch
from torch import Tensor


def grad_hess_xu(
    cost_fn: Callable[[Tensor, Tensor, int], Tensor],
    x: Tensor,
    u: Tensor,
    k: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Compute (l_x, l_u, l_xx, l_uu, l_ux) for a scalar cost l(x,u,k) using PyTorch autograd.

    This is an exact differentiation routine (no finite differences).
    
    NOTE: This is a FALLBACK function. For best performance, use analytical 
    derivatives in cost_derivs.py which are ~40x faster.
    """
    x_req = x.detach().requires_grad_(True)
    u_req = u.detach().requires_grad_(True)
    nx = x_req.numel()
    nu = u_req.numel()

    def l_of_z(z: Tensor) -> Tensor:
        xx = z[:nx]
        uu = z[nx:]
        return cost_fn(xx, uu, k)

    z = torch.cat([x_req, u_req], dim=0)
    l = l_of_z(z)
    g = torch.autograd.grad(l, z, create_graph=True)[0]
    H = torch.autograd.functional.hessian(l_of_z, z, create_graph=True)

    l_x = g[:nx]
    l_u = g[nx:]
    l_xx = H[:nx, :nx]
    l_uu = H[nx:, nx:]
    l_ux = H[nx:, :nx]
    return l_x, l_u, l_xx, l_uu, l_ux


def grad_hess_x(
    term_cost_fn: Callable[[Tensor], Tensor],
    xN: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute (phi_x, phi_xx) for a scalar terminal cost phi(xN).
    
    NOTE: This is a FALLBACK function. For best performance, use analytical 
    derivatives in cost_derivs.py which are ~40x faster.
    """
    x_req = xN.detach().requires_grad_(True)

    def phi(xx: Tensor) -> Tensor:
        return term_cost_fn(xx)

    val = phi(x_req)
    g = torch.autograd.grad(val, x_req, create_graph=True)[0]
    H = torch.autograd.functional.hessian(phi, x_req, create_graph=True)
    return g, H


def compute_jacobian(
    f: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    u: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute Jacobians df/dx and df/du using autograd.
    
    NOTE: This is a FALLBACK function. For best performance, provide 
    analytical Jacobians (f_jac parameter) which are ~30x faster.
    """
    x_req = x.detach().clone().requires_grad_(True)
    u_req = u.detach().clone().requires_grad_(True)
    
    A = torch.autograd.functional.jacobian(lambda xx: f(xx, u_req), x_req)
    B = torch.autograd.functional.jacobian(lambda uu: f(x_req, uu), u_req)
    return A, B
