"""Utility functions for efficient tensor operations.

This module provides helper functions for numerical utilities
commonly used in MPC and optimal control computations.
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
from torch import Tensor


def solve_psd(A: Tensor, b: Tensor, reg: float = 1e-6) -> Tensor:
    """Solve Ax = b where A is positive semi-definite.
    
    Uses Cholesky decomposition for efficiency (~1.2x faster than LU),
    with regularization fallback for numerical stability.
    
    Args:
        A: [n, n] PSD matrix
        b: [n] or [n, m] RHS
        reg: Regularization factor if Cholesky fails
    Returns:
        x: solution to Ax = b
    """
    try:
        L = torch.linalg.cholesky(A)
        if b.ndim == 1:
            return torch.cholesky_solve(b.unsqueeze(-1), L).squeeze(-1)
        return torch.cholesky_solve(b, L)
    except RuntimeError:
        # Add regularization and retry with LU
        n = A.shape[0]
        A_reg = A + reg * torch.eye(n, device=A.device, dtype=A.dtype)
        if b.ndim == 1:
            return torch.linalg.solve(A_reg, b.unsqueeze(-1)).squeeze(-1)
        return torch.linalg.solve(A_reg, b)


def regularize_matrix(H: Tensor, reg: float = 1e-6) -> Tensor:
    """Add regularization to a matrix diagonal.
    
    Args:
        H: [n, n] matrix
        reg: Regularization factor
    Returns:
        [n, n] regularized matrix H + reg * I
    """
    n = H.shape[-1]
    return H + reg * torch.eye(n, device=H.device, dtype=H.dtype)


def quadratic_cost_derivs_diagonal(
    x: Tensor,
    u: Tensor,
    Q: Tensor,
    R: Tensor,
    x_ref: Optional[Tensor] = None,
    u_ref: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Compute derivatives of quadratic cost with diagonal weight matrices.
    
    This is an optimized analytical computation (no autodiff needed).
    For cost: ||x - x_ref||_Q^2 + ||u - u_ref||_R^2
    
    Args:
        x: [nx] state
        u: [nu] control
        Q: [nx] diagonal state cost weights
        R: [nu] diagonal control cost weights
        x_ref: Optional reference state (defaults to 0)
        u_ref: Optional reference control (defaults to 0)
    Returns:
        l_x: [nx] gradient w.r.t. x
        l_u: [nu] gradient w.r.t. u
        l_xx: [nx, nx] Hessian w.r.t. x (diagonal)
        l_uu: [nu, nu] Hessian w.r.t. u (diagonal)
        l_ux: [nu, nx] cross Hessian (zeros for separable cost)
    """
    dx = x if x_ref is None else (x - x_ref)
    du = u if u_ref is None else (u - u_ref)
    
    l_x = 2.0 * Q * dx
    l_u = 2.0 * R * du
    l_xx = torch.diag(2.0 * Q)
    l_uu = torch.diag(2.0 * R)
    l_ux = torch.zeros(u.numel(), x.numel(), device=x.device, dtype=x.dtype)
    
    return l_x, l_u, l_xx, l_uu, l_ux
