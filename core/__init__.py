"""Strict differentiable Tube MPC (PyTorch-only) core package.

PERFORMANCE NOTES:
- Always provide analytical Jacobians (f_jac) for ~30x speedup
- Always use analytical cost derivatives (stage_derivs/terminal_derivs) for ~40x speedup
- The autodiff functions are FALLBACKS only
"""

from .ddp import (
    ILQRConfig,
    SensitivityResult,
    ilqr_solve,
    ddp_sensitivity,
    rollout,
)

from .ift import (
    IFTInputs,
    ift_gradient,
)

from .autodiff import (
    grad_hess_xu,
    grad_hess_x,
    compute_jacobian,
)

from .utils import (
    solve_psd,
    regularize_matrix,
    quadratic_cost_derivs_diagonal,
)

__all__ = [
    # DDP / iLQR
    "ILQRConfig",
    "SensitivityResult",
    "ilqr_solve",
    "ddp_sensitivity",
    "rollout",
    # IFT
    "IFTInputs",
    "ift_gradient",
    # Autodiff (fallbacks)
    "grad_hess_xu",
    "grad_hess_x",
    "compute_jacobian",
    # Utils
    "solve_psd",
    "regularize_matrix",
    "quadratic_cost_derivs_diagonal",
]
