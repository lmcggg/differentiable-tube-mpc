from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from .control import BoxClampControl


@dataclass(frozen=True)
class ILQRConfig:
    horizon: int
    nx: int
    nu: int
    max_iter: int = 30
    tol: float = 1e-6
    reg: float = 1e-6
    line_search_alphas: Tuple[float, ...] = (1.0, 0.5, 0.25, 0.1)


def _solve_reduced(
    *,
    A: Tensor,
    B: Tensor,
    active: Tensor,
) -> Tensor:
    """Solve A x = B with active-set elimination (paper Appendix G).

    For control-constrained problems, the KKT conditions partition into:
      A_ff x_f + A_fa x_a = B_f  (free variables)
      x_a = 0                     (active constraints: δu_i = 0 at bounds)
    
    This function solves only the reduced system A_ff x_f = B_f.
    
    Paper reference (Appendix G, "Incorporating Control Constraints"):
      "the control constraints can be partitioned into an active set and 
       an inactive set... This is equivalent to the condition δu_k^(i) = 0
       if u_k^(i) ∈ {u_min, u_max}"

    Args:
      A: [nu,nu] (symmetric PD after regularization) - full Hessian Q_uu
      B: [nu,m] - RHS vectors (Q_ux for K gain, tilde_Q_u for feedforward)
      active: [nu] boolean - True if control dimension is at constraint boundary
    Returns:
      X: [nu,m] with X[active]=0, X[free]=A_ff^{-1} B_f
    """
    nu = A.shape[0]
    if active.dtype != torch.bool:
        active = active.to(torch.bool)
    free_idx = torch.where(~active)[0]
    X = torch.zeros((nu, B.shape[1]), device=A.device, dtype=A.dtype)
    if free_idx.numel() == 0:
        return X
    Aff = A[free_idx][:, free_idx]
    Bf = B[free_idx]
    Xf = torch.linalg.solve(Aff, Bf)
    X[free_idx] = Xf
    return X


def _linearize_autograd(
    f: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    u: Tensor,
    *,
    create_graph: bool = False,
) -> tuple[Tensor, Tensor]:
    """Compute A=df/dx, B=df/du at a single point (no batching).
    
    Note: This is a FALLBACK for when analytical Jacobian is not provided.
    For best performance, always provide f_jac to ilqr_solve().
    """
    x = x.detach().clone().requires_grad_(True)
    u = u.detach().clone().requires_grad_(True)

    def fx(xx: Tensor) -> Tensor:
        return f(xx, u)

    def fu(uu: Tensor) -> Tensor:
        return f(x, uu)

    A = torch.autograd.functional.jacobian(fx, x, create_graph=create_graph)
    B = torch.autograd.functional.jacobian(fu, u, create_graph=create_graph)
    return A, B


def rollout(x0: Tensor, V: Tensor, *, f: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
    """Rollout for single trajectory: x0 [nx], V [N,nu] -> X [N+1,nx]."""
    N = V.shape[0]
    nx = x0.shape[0]
    X = torch.empty(N + 1, nx, device=x0.device, dtype=x0.dtype)
    X[0] = x0
    x = x0
    for k in range(N):
        x = f(x, V[k])
        X[k + 1] = x
    return X


def ilqr_solve(
    *,
    x0: Tensor,
    V_init: Tensor,
    cfg: ILQRConfig,
    f: Callable[[Tensor, Tensor], Tensor],
    f_jac: Optional[Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]] = None,
    ctrl: Optional[BoxClampControl] = None,
    stage_cost: Callable[[Tensor, Tensor, int], Tensor],
    terminal_cost: Callable[[Tensor], Tensor],
    stage_derivs: Callable[[Tensor, Tensor, int], tuple[Tensor, Tensor, Tensor, Tensor, Tensor]],
    terminal_derivs: Callable[[Tensor], tuple[Tensor, Tensor]],
    feasible_fn: Optional[Callable[[Tensor, int], bool]] = None,
    debug: bool = False,
    debug_name: str = "ilqr",
) -> tuple[Tensor, Tensor]:
    """Solve equality-constrained OCP with iLQR (DDP-style).

    PERFORMANCE NOTE:
    - Always provide f_jac (analytical Jacobian) for ~30x speedup
    - stage_derivs/terminal_derivs should be analytical for ~40x speedup
    - If f_jac is None, falls back to slow autograd (~30x slower)
    """
    N = cfg.horizon
    nu = cfg.nu
    V = V_init.clone()
    if ctrl is not None:
        V = ctrl.clamp(V)

    X = rollout(x0, V, f=f)
    prev_cost: Optional[Tensor] = None
    
    device, dtype = x0.device, x0.dtype
    # Pre-allocate regularization matrix (reused every iteration)
    reg_eye = cfg.reg * torch.eye(nu, device=device, dtype=dtype)

    def _ensure_finite(t: Tensor, name: str, *, it: int | None = None, k: int | None = None) -> None:
        if torch.isfinite(t).all():
            return
        if not debug:
            raise FloatingPointError(f"{debug_name}: non-finite detected in {name}")
        where = []
        if it is not None:
            where.append(f"it={it}")
        if k is not None:
            where.append(f"k={k}")
        where_s = (", ".join(where)) if where else "-"
        finite = torch.isfinite(t)
        bad = (~finite).sum().item()
        if t.numel() and finite.any():
            vals = t[finite]
            t_min = vals.min().item()
            t_max = vals.max().item()
        else:
            t_min = float("nan")
            t_max = float("nan")
        print(f"[NUMERIC-FAIL] {debug_name} {where_s}: {name} has {bad} non-finite entries (min={t_min}, max={t_max})", flush=True)
        raise FloatingPointError(f"{debug_name}: non-finite detected in {name} ({where_s})")

    for it in range(cfg.max_iter):
        # Derivatives along trajectory
        A_seq = []
        B_seq = []
        l_x = []
        l_u = []
        l_xx = []
        l_uu = []
        l_ux = []
        cost = torch.zeros((), device=device, dtype=dtype)

        for k in range(N):
            _ensure_finite(X[k], "X[k]", it=it, k=k)
            _ensure_finite(V[k], "V[k]", it=it, k=k)
            
            # Use analytical Jacobian if provided (30x faster!)
            if f_jac is not None:
                Ak, Bk = f_jac(X[k], V[k])
            else:
                Ak, Bk = _linearize_autograd(f, X[k], V[k], create_graph=False)
            _ensure_finite(Ak, "A_k", it=it, k=k)
            _ensure_finite(Bk, "B_k", it=it, k=k)
            
            # Use analytical cost derivatives (40x faster than autograd!)
            lx, lu, lxx, luu, lux = stage_derivs(X[k], V[k], k)
            _ensure_finite(lx, "l_x", it=it, k=k)
            _ensure_finite(lu, "l_u", it=it, k=k)
            _ensure_finite(lxx, "l_xx", it=it, k=k)
            _ensure_finite(luu, "l_uu", it=it, k=k)
            _ensure_finite(lux, "l_ux", it=it, k=k)
            A_seq.append(Ak)
            B_seq.append(Bk)
            l_x.append(lx)
            l_u.append(lu)
            l_xx.append(lxx)
            l_uu.append(luu)
            l_ux.append(lux)
            cost = cost + stage_cost(X[k], V[k], k)
            _ensure_finite(cost, "cost_partial", it=it, k=k)

        phi_x, phi_xx = terminal_derivs(X[N])
        _ensure_finite(phi_x, "phi_x", it=it, k=N)
        _ensure_finite(phi_xx, "phi_xx", it=it, k=N)
        cost = cost + terminal_cost(X[N])
        _ensure_finite(cost, "cost_total", it=it, k=N)

        # Backward pass (standard iLQR)
        V_x = phi_x
        V_xx = phi_xx
        K_seq = [None] * N
        k_seq = [None] * N

        for k in reversed(range(N)):
            A = A_seq[k]
            B = B_seq[k]

            Q_x = l_x[k] + A.T @ V_x
            Q_u = l_u[k] + B.T @ V_x
            Q_xx = l_xx[k] + A.T @ V_xx @ A
            Q_ux = l_ux[k] + B.T @ V_xx @ A
            Q_xu = Q_ux.T
            BtVxxB = B.T @ V_xx @ B
            Q_uu = l_uu[k] + BtVxxB
            if debug and (not torch.isfinite(Q_uu).all()):
                def _s(t: Tensor) -> str:
                    finite = torch.isfinite(t)
                    bad = (~finite).sum().item()
                    mx = (t[finite].abs().max().item() if t.numel() and finite.any() else float("nan"))
                    return f"shape={tuple(t.shape)} bad={int(bad)} max_abs={mx}"
                print(f"[NUMERIC-DETAIL] {debug_name} it={it} k={k}", flush=True)
                print(f"  l_uu:   {_s(l_uu[k])}", flush=True)
                print(f"  B:      {_s(B)}", flush=True)
                print(f"  V_xx:   {_s(V_xx)}", flush=True)
                print(f"  BtVxxB: {_s(BtVxxB)}", flush=True)
                print(f"  Q_uu finite mask:\n{torch.isfinite(Q_uu)}", flush=True)
            _ensure_finite(Q_uu, "Q_uu", it=it, k=k)

            # Regularization for numerical stability
            Q_uu_reg = Q_uu + reg_eye
            _ensure_finite(Q_uu_reg, "Q_uu_reg", it=it, k=k)

            # Solve for gains (linalg.solve handles small matrices efficiently)
            K = -torch.linalg.solve(Q_uu_reg, Q_ux)
            kff = -torch.linalg.solve(Q_uu_reg, Q_u)
            _ensure_finite(K, "K", it=it, k=k)
            _ensure_finite(kff, "kff", it=it, k=k)

            K_seq[k] = K
            k_seq[k] = kff

            V_x = Q_x + K.T @ Q_uu @ kff + K.T @ Q_u + Q_xu @ kff
            V_xx = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_xu @ K
            _ensure_finite(V_x, "V_x", it=it, k=k)
            _ensure_finite(V_xx, "V_xx", it=it, k=k)

        # Forward pass with line search
        best_cost: Optional[Tensor] = None
        best_X: Optional[Tensor] = None
        best_U: Optional[Tensor] = None

        for alpha in cfg.line_search_alphas:
            X_new = torch.empty_like(X)
            V_new = torch.empty_like(V)
            X_new[0] = x0
            x = x0
            for k in range(N):
                dx = x - X[k]
                du = k_seq[k] + K_seq[k] @ dx
                u = V[k] + alpha * du
                if ctrl is not None:
                    u = ctrl.clamp(u)
                V_new[k] = u
                x = f(x, u)
                X_new[k + 1] = x
            _ensure_finite(X_new, "X_new", it=it)
            _ensure_finite(V_new, "V_new", it=it)

            if feasible_fn is not None:
                ok = True
                for k in range(N + 1):
                    if not feasible_fn(X_new[k], k):
                        ok = False
                        break
                if not ok:
                    continue

            cand_cost = torch.zeros((), device=device, dtype=dtype)
            for k in range(N):
                cand_cost = cand_cost + stage_cost(X_new[k], V_new[k], k)
            cand_cost = cand_cost + terminal_cost(X_new[N])
            _ensure_finite(cand_cost, "cand_cost", it=it)

            if best_cost is None or cand_cost < best_cost:
                best_cost = cand_cost
                best_X = X_new
                best_U = V_new

        if best_X is None or best_U is None or best_cost is None:
            raise RuntimeError("iLQR line search failed to produce a candidate.")

        X, V = best_X, best_U

        if prev_cost is not None and torch.abs(prev_cost - best_cost) < cfg.tol:
            break
        prev_cost = best_cost

    return X, V


@dataclass(frozen=True)
class SensitivityResult:
    delta_X: Tensor  # [N+1, nx]
    delta_V: Tensor  # [N, nu]
    delta_lambda: Tensor  # [N+1, nx]


def ddp_sensitivity(
    *,
    X: Tensor,
    V: Tensor,
    f: Callable[[Tensor, Tensor], Tensor],
    f_jac: Optional[Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]] = None,
    ctrl: Optional[BoxClampControl] = None,
    stage_hess: Callable[[Tensor, Tensor, int], tuple[Tensor, Tensor, Tensor]],
    terminal_hess: Callable[[Tensor], Tensor],
    upper_grad_x: Callable[[Tensor, int], Tensor],
    upper_grad_u: Callable[[Tensor, int], Tensor],
    upper_grad_xN: Callable[[Tensor], Tensor],
) -> SensitivityResult:
    """Compute δz by solving Lzz δz = -∇z L_upper using the DDP-structured recursions provided.

    Implements the user's equations (Backward Pass + Forward Pass).
    """
    N = V.shape[0]
    nx = X.shape[1]
    nu = V.shape[1]
    device, dtype = X.device, X.dtype

    A_seq = []
    B_seq = []
    l_xx = []
    l_uu = []
    l_ux = []

    for k in range(N):
        if f_jac is not None:
            Ak, Bk = f_jac(X[k], V[k])
        else:
            Ak, Bk = _linearize_autograd(f, X[k], V[k], create_graph=False)
        L_xx_k, L_uu_k, L_ux_k = stage_hess(X[k], V[k], k)
        A_seq.append(Ak)
        B_seq.append(Bk)
        l_xx.append(L_xx_k)
        l_uu.append(L_uu_k)
        l_ux.append(L_ux_k)

    phi_xx = terminal_hess(X[N])

    # Everything in δz solve is purely numerical; keep it out of autograd graph.
    with torch.no_grad():
        # Backward pass
        V_xx_next = phi_xx
        tilde_V_x_next = upper_grad_xN(X[N])

        K_seq = [None] * N
        k_seq = [None] * N
        active_seq = [None] * N
        V_xx_seq = [None] * (N + 1)
        tilde_V_x_seq = [None] * (N + 1)
        V_xx_seq[N] = V_xx_next
        tilde_V_x_seq[N] = tilde_V_x_next

        reg_eye = 1e-9 * torch.eye(nu, device=device, dtype=dtype)

        for k in reversed(range(N)):
            A = A_seq[k]
            B = B_seq[k]

            Q_xx = l_xx[k] + A.T @ V_xx_next @ A
            Q_xu = l_ux[k].T + A.T @ V_xx_next @ B
            Q_ux = l_ux[k] + B.T @ V_xx_next @ A
            Q_uu = l_uu[k] + B.T @ V_xx_next @ B

            g_u = upper_grad_u(V[k], k)
            g_x = upper_grad_x(X[k], k)

            tilde_Q_u = g_u + B.T @ tilde_V_x_next
            tilde_Q_x = g_x + A.T @ tilde_V_x_next

            Q_uu_reg = Q_uu + reg_eye
            if ctrl is not None:
                active = ctrl.active_mask(V[k]).detach()
            else:
                active = torch.zeros((nu,), device=device, dtype=torch.bool)
            active_seq[k] = active

            K = -_solve_reduced(A=Q_uu_reg, B=Q_ux, active=active)
            kff = -_solve_reduced(A=Q_uu_reg, B=tilde_Q_u.view(-1, 1), active=active).view(-1)

            K_seq[k] = K
            k_seq[k] = kff

            tilde_V_x = tilde_Q_x + Q_xu @ kff
            V_xx = Q_xx + Q_xu @ K

            V_xx_seq[k] = V_xx
            tilde_V_x_seq[k] = tilde_V_x

            V_xx_next = V_xx
            tilde_V_x_next = tilde_V_x

        # Forward pass
        delta_X = torch.zeros_like(X)
        delta_V = torch.zeros_like(V)
        delta_lambda = torch.zeros_like(X)

        for k in range(N):
            delta_V[k] = k_seq[k] + K_seq[k] @ delta_X[k]
            if ctrl is not None:
                active = active_seq[k]
                if active is not None and torch.any(active):
                    delta_V[k][active] = 0.0
            delta_X[k + 1] = A_seq[k] @ delta_X[k] + B_seq[k] @ delta_V[k]
            delta_lambda[k] = tilde_V_x_seq[k] + V_xx_seq[k] @ delta_X[k]
        delta_lambda[N] = tilde_V_x_seq[N] + V_xx_seq[N] @ delta_X[N]

    return SensitivityResult(delta_X=delta_X, delta_V=delta_V, delta_lambda=delta_lambda)
