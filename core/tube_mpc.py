from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import numpy as np
import torch
from torch import Tensor

from .barrier import DBaSConfig, dbas_init_b0, dbas_step
from .control import BoxClampControl
from .cost_derivs import (
    auxiliary_cost_derivs_u,
    auxiliary_terminal_derivs,
    nominal_cost_derivs_u,
    nominal_terminal_derivs,
)
from .ddp import ILQRConfig, ddp_sensitivity, ilqr_solve
from .ift import IFTInputs, ift_gradient
from .params import AuxiliaryTheta, NominalTheta
from .systems.dubins import DubinsConfig, dubins_step, sample_disturbance, default_safe_h_no_obstacles
from .systems.dubins_aug_jac import dubins_augmented_jacobian
from .systems.dubins_obstacles import CircleObstacle, h_circle_obstacle, h_min_circle_obstacles, h_multi_circle_obstacles


@dataclass
class ExperimentTrajectories:
    x_real: list[np.ndarray]
    u_real: list[np.ndarray]
    x_bar: list[np.ndarray]
    u_bar: list[np.ndarray]
    loss: list[float]
    b_real: list[np.ndarray]
    Qa_history: list[np.ndarray]  # Track Qa parameter evolution
    Ra_history: list[np.ndarray]  # Track Ra parameter evolution
    qba_history: list[float]  # Track qba parameter evolution


def run_closed_loop_experiment(cfg: Dict[str, Any], *, device: torch.device, run_dir: str) -> Dict[str, Any]:
    system_cfg = cfg["system"]
    if system_cfg["name"] != "dubins":
        raise NotImplementedError("Only dubins is wired in the skeleton; other systems are added next.")

    # Paper-aligned Dubins setting: nominal fixed, ancillary adapts (Appendix:experiments).
    # We switch to a much faster, fully-analytic DOC gradient for ancillary weights (Q,R,q_b),
    # which matches Algorithm 1/DT-MPC in the paper but avoids heavy autograd.
    if bool(cfg.get("paper_dubins_mode", True)) and (not bool(cfg.get("adaptation", {}).get("adapt_nominal", True))):
        return _run_dubins_paper(cfg, device=device, run_dir=run_dir)

    dtype = torch.float64 if bool(cfg.get("use_float64", False)) else torch.float32
    dt = float(system_cfg["dt"])
    N = int(system_cfg["horizon_N"])
    H = int(system_cfg["task_horizon_H"])

    dub_cfg = DubinsConfig(
        dt=dt,
        v_max=float(system_cfg["control_bounds"]["v_max"]),
        omega_max=float(system_cfg["control_bounds"]["omega_max"]),
        w_low=tuple(system_cfg["disturbance"]["w_low"]),
        w_high=tuple(system_cfg["disturbance"]["w_high"]),
        x_target=tuple(system_cfg["target"]),
    )

    # Safety function h(x): single obstacle or multi-obstacle aggregation
    env_cfg = cfg.get("environment", {})
    obs_beta = float(env_cfg.get("obstacle_smoothmin_beta", 20.0))
    obs_agg = str(env_cfg.get("obstacle_aggregation", "min"))
    if "obstacles" in env_cfg:
        obs_list = [CircleObstacle(center=tuple(o["center"]), radius=float(o["radius"])) for o in env_cfg["obstacles"]]
        obs = obs_list
        if obs_agg == "smoothmin":
            h_base = lambda x_in: h_multi_circle_obstacles(x_in, obstacles=obs_list, beta=obs_beta)
        else:
            h_base = lambda x_in: h_min_circle_obstacles(x_in, obstacles=obs_list)
    elif "obstacle" in env_cfg:
        obs_cfg = env_cfg["obstacle"]
        obs = CircleObstacle(center=tuple(obs_cfg["center"]), radius=float(obs_cfg["radius"]))
        h_base = lambda x_in: h_circle_obstacle(x_in, obs=obs)
    else:
        obs = CircleObstacle(center=(5.0, 5.0), radius=1.5)  # fallback
        h_base = default_safe_h_no_obstacles

    # Control constraints: hard box with active-set handling (paper Appendix: control constraints).
    v_min = float(system_cfg["control_bounds"].get("v_min", -dub_cfg.v_max))
    u_min = torch.tensor([v_min, -dub_cfg.omega_max], device=device, dtype=dtype)
    u_max = torch.tensor([dub_cfg.v_max, dub_cfg.omega_max], device=device, dtype=dtype)
    ctrl = BoxClampControl(u_min=u_min, u_max=u_max)

    # Nominal dynamics f(x,u)
    f = lambda x, u: dubins_step(x, u, cfg=dub_cfg)

    # Initial conditions
    # Paper: start at origin facing upper-right (theta=pi/4)
    x = torch.tensor([0.0, 0.0, float(np.pi / 4)], device=device, dtype=dtype)
    # Initialize nominal state separately (tube structure): \bar{x}_0 := x_0
    x_bar = x.clone()

    # Paper (Dubins): nominal fixed, ancillary adapts; inverse barrier with alpha=0, gamma=0 fixed.
    adapt_cfg = cfg.get("adaptation", {})
    adapt_nominal = bool(adapt_cfg.get("adapt_nominal", True))
    adapt_ancillary = bool(adapt_cfg.get("adapt_ancillary", True))

    # Initialize parameters θ̄ and θ (raw tensors requiring grad when adapted)
    Q0 = torch.tensor(cfg["cost_nominal"]["Q"], device=device, dtype=dtype)
    R0 = torch.tensor(cfg["cost_nominal"]["R"], device=device, dtype=dtype)
    Qf0 = torch.tensor(cfg["cost_nominal"]["Qf"], device=device, dtype=dtype)
    qb0 = torch.tensor(float(cfg["cost_nominal"]["q_b"]), device=device, dtype=dtype)

    alpha0 = torch.tensor(float(cfg["dbas"]["alpha"]), device=device, dtype=dtype)
    gamma0 = torch.tensor(float(cfg["dbas"]["gamma"]), device=device, dtype=dtype)
    tight0 = torch.tensor(float(cfg["dbas"].get("nominal_tightening", 0.0)), device=device, dtype=dtype)

    theta_bar = NominalTheta(
        Q_raw=Q0.clone().requires_grad_(adapt_nominal),
        R_raw=R0.clone().requires_grad_(adapt_nominal),
        Qf_raw=Qf0.clone().requires_grad_(adapt_nominal),
        qb_raw=qb0.clone().requires_grad_(adapt_nominal),
        alpha_raw=alpha0.clone().requires_grad_(adapt_nominal),
        gamma_raw=gamma0.clone().requires_grad_(adapt_nominal),
        tight_raw=tight0.clone().requires_grad_(adapt_nominal),
    )

    qb_aux0 = torch.tensor(float(cfg.get("cost_auxiliary", {}).get("q_b", float(qb0))), device=device, dtype=dtype)
    theta = AuxiliaryTheta(
        Q_raw=Q0.clone().requires_grad_(adapt_ancillary),
        R_raw=R0.clone().requires_grad_(adapt_ancillary),
        Qf_raw=Qf0.clone().requires_grad_(adapt_ancillary),
        qb_raw=qb_aux0.clone().requires_grad_(adapt_ancillary),
        alpha_raw=alpha0.clone().requires_grad_(adapt_ancillary),
        gamma_raw=gamma0.clone().requires_grad_(adapt_ancillary),
    )

    # DBaS configs are derived from parameters (strict online update)
    def db_cfg_aux() -> DBaSConfig:
        return DBaSConfig(
            barrier_type=cfg["dbas"]["barrier_type"],
            alpha=theta.alpha(),
            gamma=theta.gamma(),
            eps=float(cfg["dbas"].get("eps", DBaSConfig().eps)),
        )

    def db_cfg_nom() -> DBaSConfig:
        return DBaSConfig(
            barrier_type=cfg["dbas"]["barrier_type"],
            alpha=theta_bar.alpha(),
            gamma=theta_bar.gamma(),
            eps=float(cfg["dbas"].get("eps", DBaSConfig().eps)),
        )

    # Tightened safety for nominal MPC: h̄(x)=h(x) - s,  s=tight(θ̄) >= 0
    def h_nom(x_in: Tensor) -> Tensor:
        return h_base(x_in) - theta_bar.tight()

    def h_aux(x_in: Tensor) -> Tensor:
        return h_base(x_in)

    # Initialize barrier states
    b = dbas_init_b0(x, h=h_aux, cfg=db_cfg_aux())
    b_bar = dbas_init_b0(x_bar, h=h_nom, cfg=db_cfg_nom())

    # Solver configs
    reg = float(system_cfg.get("ilqr_reg", 1e-6))
    ilqr_cfg_nom = ILQRConfig(horizon=N, nx=4, nu=2, max_iter=int(system_cfg.get("nominal_max_iter", 10)), reg=reg)
    ilqr_cfg_aux = ILQRConfig(horizon=N, nx=4, nu=2, max_iter=int(system_cfg.get("aux_max_iter", 10)), reg=reg)

    target = torch.tensor(dub_cfg.x_target, device=device, dtype=dtype)

    # Warm starts
    V_nom_ws = torch.zeros(N, 2, device=device, dtype=dtype)
    V_aux_ws = torch.zeros(N, 2, device=device, dtype=dtype)

    traj = ExperimentTrajectories(
        x_real=[], u_real=[], x_bar=[], u_bar=[], loss=[], b_real=[],
        Qa_history=[], Ra_history=[], qba_history=[]
    )
    lr = float(adapt_cfg.get("lr_eta", 1e-3))
    adapt_steps = int(adapt_cfg.get("steps", 1))
    clip_norm = float(adapt_cfg.get("grad_clip_norm", 0.0))
    momentum = float(adapt_cfg.get("momentum", 0.0))
    project_params = bool(adapt_cfg.get("project_params", False))

    # Simple momentum buffers (paper uses Nesterov momentum; we use standard momentum for stability)
    v_theta = [torch.zeros_like(p) for p in theta.tensors()]
    v_theta_bar = [torch.zeros_like(p) for p in theta_bar.tensors()]

    def _project(p: Tensor, name: str) -> None:
        """Projected gradient descent constraints (paper Section 4, Algorithm 2).
        
        Physical meaning and constraints:
        - Q, Qf: state cost weights, must be non-negative (PSD quadratic form)
        - R: control cost weights, must be positive definite (R >= eps > 0)
        - q_b: barrier penalty weight, bounded to prevent over-penalization
        - gamma: DBaS feedback gain, must be in [-1, 1] for stability
        - alpha: relaxed barrier smoothness parameter, non-negative
        - tight: nominal constraint tightening, non-negative (safety margin)
        """
        if name in ("Q_raw", "Qf_raw"):
            # State cost weights: Q >= 0 (semi-definite is sufficient)
            p.clamp_(min=0.0)
            # Optional: upper bound to prevent overfitting
            # p.clamp_(min=0.0, max=1e4)
        elif name == "R_raw":
            # Control cost weights: R > 0 (positive definite required for convergence)
            # Paper supplementary: R >= 1e-4 ensures Q_uu is well-conditioned
            p.clamp_(min=1e-4, max=1e4)
        elif name == "qb_raw":
            # Barrier penalty: q_b in [0, 1] balances safety vs task performance
            # Too large q_b → overly conservative; too small → safety violations
            p.clamp_(min=0.0, max=1.0)
        elif name == "gamma_raw":
            # DBaS feedback gain: gamma in [-1, 1] (paper Section 2.2)
            # gamma=0: no feedback (forward barrier propagation only)
            # gamma=-1: full error feedback (aggressive tracking)
            p.clamp_(min=-1.0, max=1.0)
        elif name == "alpha_raw":
            # Relaxed barrier smoothness: alpha >= 0 (paper Section 2.2)
            # alpha → 0 recovers exact inverse barrier (paper Eq. 9)
            # Larger alpha increases constraint relaxation (recursive feasibility)
            p.clamp_(min=0.0, max=1.0)
        elif name == "tight_raw":
            # Nominal constraint tightening: s >= 0 (paper Problem 5)
            # Ensures nominal trajectory stays away from boundary by margin s
            p.clamp_(min=0.0, max=2.0)

    def _names_theta():
        return ["Q_raw", "R_raw", "Qf_raw", "qb_raw", "alpha_raw", "gamma_raw"]

    def _names_theta_bar():
        return ["Q_raw", "R_raw", "Qf_raw", "qb_raw", "alpha_raw", "gamma_raw", "tight_raw"]

    def _apply_update(params, grads, vel, names):
        with torch.no_grad():
            for i, (p, g) in enumerate(zip(params, grads)):
                if g is None:
                    continue
                g_eff = g
                if clip_norm and clip_norm > 0:
                    n = torch.linalg.vector_norm(g_eff).item()
                    if n > clip_norm:
                        g_eff = g_eff * (clip_norm / (n + 1e-12))
                if momentum and momentum > 0:
                    vel[i].mul_(momentum).add_(g_eff)
                    step = vel[i]
                else:
                    step = g_eff
                p -= lr * step
                if project_params:
                    _project(p, names[i])

    for t in range(H):
        if (t % 25) == 0:
            print(f"[step {t}/{H}] running...", flush=True)
        # -------------------------
        # (A) Solve nominal MPC (Problem 5) on augmented state (x,b)
        # -------------------------
        x_hat_nom0 = torch.cat([x_bar, b_bar.view(1)], dim=0)

        # Detach parameters for *solving* the lower-level problem (we use implicit gradients, not backprop through solver).
        Qn = theta_bar.Q().detach()
        Rn = theta_bar.R().detach()
        Qfn = theta_bar.Qf().detach()
        qbn = theta_bar.qb().detach()
        alpha_n = theta_bar.alpha().detach()
        gamma_n = theta_bar.gamma().detach()
        tight_n = theta_bar.tight().detach()

        def f_hat_nom(x_hat_k: Tensor, v_k: Tensor) -> Tensor:
            u_k = v_k
            xk = x_hat_k[:-1]
            bk = x_hat_k[-1]
            # Use detached tightening/DBaS params during solve
            def h_nom_solve(xx: Tensor) -> Tensor:
                return h_base(xx) - tight_n
            x_next, b_next = dbas_step(
                x_k=xk, u_k=u_k, b_k=bk, f=f, h=h_nom_solve,
                cfg=DBaSConfig(barrier_type=cfg["dbas"]["barrier_type"], alpha=alpha_n, gamma=gamma_n)
            )
            return torch.cat([x_next, b_next.view(1)], dim=0)

        def stage_cost_nom(x_hat_k: Tensor, v_k: Tensor, k: int) -> Tensor:
            u_k = v_k
            xk = x_hat_k[:-1]
            bk = x_hat_k[-1]
            dx = xk - target
            return (Qn * dx * dx).sum() + (Rn * u_k * u_k).sum() + qbn * (bk * bk)

        def terminal_cost_nom(x_hat_N: Tensor) -> Tensor:
            xN = x_hat_N[:-1]
            bN = x_hat_N[-1]
            dxN = xN - target
            # Paper: terminal cost includes barrier state penalty
            return (Qfn * dxN * dxN).sum() + qbn * (bN * bN)

        def stage_derivs_nom(x_hat_k: Tensor, v_k: Tensor, k: int):
            return nominal_cost_derivs_u(x_hat=x_hat_k, u=v_k, target=target, Q=Qn, R=Rn, qb=qbn)

        def term_derivs_nom(x_hat_N: Tensor):
            # Paper: terminal derivatives include barrier state
            phi_x, phi_xx = nominal_terminal_derivs(x_hat_N=x_hat_N, target=target, Qf=Qfn)
            phi_x[-1] = 2.0 * qbn * x_hat_N[-1]
            phi_xx[-1, -1] = 2.0 * qbn
            return phi_x, phi_xx

        X_nom_hat, V_nom = ilqr_solve(
            x0=x_hat_nom0,
            V_init=V_nom_ws,
            cfg=ilqr_cfg_nom,
            f=f_hat_nom,
            stage_cost=stage_cost_nom,
            terminal_cost=terminal_cost_nom,
            stage_derivs=stage_derivs_nom,
            terminal_derivs=term_derivs_nom,
            ctrl=ctrl,
            f_jac=lambda xh, vk: dubins_augmented_jacobian(
                xh, vk, cfg=dub_cfg, obs=obs, obs_beta=obs_beta, obs_agg=obs_agg,
                db_cfg=DBaSConfig(barrier_type="inverse", alpha=alpha_n, gamma=gamma_n, eps=float(cfg["dbas"].get("eps", DBaSConfig().eps)))
            ),
        )

        U_nom = V_nom

        # Nominal references for auxiliary MPC
        X_ref = X_nom_hat[:, :-1]  # [N+1, 3]
        U_ref = U_nom              # [N, 2]

        # -------------------------
        # (B) Solve auxiliary MPC (Problem 6) tracking nominal
        # -------------------------
        x_hat0 = torch.cat([x, b.view(1)], dim=0)

        # Make references require_grad so we can compute the full coupled ∇_{θ̄} L through the auxiliary reference dependence.
        X_ref_param = X_ref.detach().clone().requires_grad_(True)
        U_ref_param = U_ref.detach().clone().requires_grad_(True)

        # Detach θ for solving (implicit gradients later).
        Qa = theta.Q().detach()
        Ra = theta.R().detach()
        Qfa = theta.Qf().detach()
        qba = theta.qb().detach()
        alpha_a = theta.alpha().detach()
        gamma_a = theta.gamma().detach()

        # Also detach references for solving (treat as fixed parameters in the lower-level solve).
        X_ref_solve = X_ref_param.detach()
        U_ref_solve = U_ref_param.detach()

        def f_hat_aux(x_hat_k: Tensor, v_k: Tensor) -> Tensor:
            u_k = v_k
            xk = x_hat_k[:-1]
            bk = x_hat_k[-1]
            x_next, b_next = dbas_step(
                x_k=xk, u_k=u_k, b_k=bk, f=f, h=h_aux,
                cfg=DBaSConfig(barrier_type=cfg["dbas"]["barrier_type"], alpha=alpha_a, gamma=gamma_a)
            )
            return torch.cat([x_next, b_next.view(1)], dim=0)

        def stage_cost_aux(x_hat_k: Tensor, v_k: Tensor, k: int) -> Tensor:
            u_k = v_k
            xk = x_hat_k[:-1]
            bk = x_hat_k[-1]
            dx = xk - X_ref_solve[k]
            du = u_k - U_ref_solve[k]
            return (Qa * dx * dx).sum() + (Ra * du * du).sum() + qba * (bk * bk)

        def terminal_cost_aux(x_hat_N: Tensor) -> Tensor:
            xN = x_hat_N[:-1]
            bN = x_hat_N[-1]
            dxN = xN - X_ref_solve[N]
            # Paper: terminal cost includes barrier state penalty
            return (Qfa * dxN * dxN).sum() + qba * (bN * bN)

        def stage_derivs_aux(x_hat_k: Tensor, v_k: Tensor, k: int):
            return auxiliary_cost_derivs_u(
                x_hat=x_hat_k, u=v_k, x_ref=X_ref_solve[k], u_ref=U_ref_solve[k], Q=Qa, R=Ra, qb=qba
            )

        def term_derivs_aux(x_hat_N: Tensor):
            # Paper: terminal derivatives include barrier state
            phi_x, phi_xx = auxiliary_terminal_derivs(x_hat_N=x_hat_N, x_ref_N=X_ref_solve[N], Qf=Qfa)
            phi_x[-1] = 2.0 * qba * x_hat_N[-1]
            phi_xx[-1, -1] = 2.0 * qba
            return phi_x, phi_xx

        X_aux_hat, V_aux = ilqr_solve(
            x0=x_hat0,
            V_init=V_aux_ws,
            cfg=ilqr_cfg_aux,
            f=f_hat_aux,
            stage_cost=stage_cost_aux,
            terminal_cost=terminal_cost_aux,
            stage_derivs=stage_derivs_aux,
            terminal_derivs=term_derivs_aux,
            ctrl=ctrl,
            f_jac=lambda xh, vk: dubins_augmented_jacobian(
                xh, vk, cfg=dub_cfg, obs=obs, obs_beta=obs_beta, obs_agg=obs_agg,
                db_cfg=DBaSConfig(barrier_type="inverse", alpha=alpha_a, gamma=gamma_a, eps=float(cfg["dbas"].get("eps", DBaSConfig().eps)))
            ),
        )

        U_aux = V_aux
        u = U_aux[0]

        # -------------------------
        # (C) Online adaptation (Algorithm 2): update θ̄ and θ using strict DDP sensitivity
        # -------------------------
        for _ in range(adapt_steps):
            # Upper loss L = ||x* - x̄||^2 + ||b*||^2
            x_aux = X_aux_hat[:, :-1]
            b_aux = X_aux_hat[:, -1]
            x_nom = X_nom_hat[:, :-1]
            loss_track = (x_aux - x_nom).pow(2).sum()
            loss_bar = b_aux.pow(2).sum()
            loss_val = (loss_track + loss_bar).detach()

            # ---- Auxiliary: δz solve (DDP structure) ----
            def stage_hess_aux(x_hat_k: Tensor, v_k: Tensor, k: int):
                _, _, l_xx, l_vv, l_vx = stage_derivs_aux(x_hat_k, v_k, k)
                return l_xx, l_vv, l_vx

            def term_hess_aux(x_hat_N: Tensor):
                _, phi_xx = term_derivs_aux(x_hat_N)
                return phi_xx

            def upper_gx_aux(x_hat_k: Tensor, k: int) -> Tensor:
                xk = x_hat_k[:-1]
                bk = x_hat_k[-1]
                gx = 2.0 * (xk - x_nom[k])
                gb = 2.0 * bk
                return torch.cat([gx, gb.view(1)], dim=0)

            def upper_gv_zero(_v_k: Tensor, _k: int) -> Tensor:
                return torch.zeros_like(_v_k)

            def upper_gxN_aux(x_hat_N: Tensor) -> Tensor:
                xN = x_hat_N[:-1]
                bN = x_hat_N[-1]
                gx = 2.0 * (xN - x_nom[N])
                gb = 2.0 * bN
                return torch.cat([gx, gb.view(1)], dim=0)

            sens_aux = ddp_sensitivity(
                X=X_aux_hat,
                V=V_aux,
                f=f_hat_aux,
                ctrl=ctrl,
                f_jac=lambda xh, vk: dubins_augmented_jacobian(
                    xh, vk, cfg=dub_cfg, obs=obs, obs_beta=obs_beta, obs_agg=obs_agg,
                    db_cfg=DBaSConfig(barrier_type="inverse", alpha=alpha_a, gamma=gamma_a, eps=float(cfg["dbas"].get("eps", DBaSConfig().eps)))
                ),
                stage_hess=stage_hess_aux,
                terminal_hess=term_hess_aux,
                upper_grad_x=upper_gx_aux,
                upper_grad_u=upper_gv_zero,
                upper_grad_xN=upper_gxN_aux,
            )

            # ---- Auxiliary: full IFT gradients ----
            aux_theta_tensors = theta.tensors()
            # If nominal is fixed (Dubins paper setting), we do not need reference gradients.
            all_theta_aux = aux_theta_tensors if not adapt_nominal else (aux_theta_tensors + [X_ref_param, U_ref_param])

            # Build *gradient* versions of dynamics/cost that depend on non-detached parameters.
            def f_hat_aux_grad(x_hat_k: Tensor, v_k: Tensor) -> Tensor:
                u_k = v_k
                xk = x_hat_k[:-1]
                bk = x_hat_k[-1]
                x_next, b_next = dbas_step(
                    x_k=xk, u_k=u_k, b_k=bk, f=f, h=h_aux,
                    cfg=db_cfg_aux()
                )
                return torch.cat([x_next, b_next.view(1)], dim=0)

            def stage_cost_aux_grad(x_hat_k: Tensor, v_k: Tensor, k: int) -> Tensor:
                u_k = v_k
                xk = x_hat_k[:-1]
                bk = x_hat_k[-1]
                dx = xk - X_ref_param[k]
                du = u_k - U_ref_param[k]
                return (theta.Q() * dx * dx).sum() + (theta.R() * du * du).sum() + theta.qb() * (bk * bk)

            def terminal_cost_aux_grad(x_hat_N: Tensor) -> Tensor:
                xN = x_hat_N[:-1]
                bN = x_hat_N[-1]
                dxN = xN - X_ref_param[N]
                # Paper: terminal cost includes barrier state penalty
                return (theta.Qf() * dxN * dxN).sum() + theta.qb() * (bN * bN)

            grads = ift_gradient(
                inputs=IFTInputs(
                    X=X_aux_hat,
                    V=V_aux,
                    delta_X=sens_aux.delta_X,
                    delta_V=sens_aux.delta_V,
                    delta_lambda=sens_aux.delta_lambda,
                ),
                theta_tensors=all_theta_aux,
                xi_fn=lambda: x_hat0.detach(),
                f_fn=f_hat_aux_grad,
                stage_cost_fn=stage_cost_aux_grad,
                terminal_cost_fn=terminal_cost_aux_grad,
            )

            # Split gradients
            g_theta_raw = grads[: len(aux_theta_tensors)]
            g_xref = None
            g_uref = None
            if adapt_nominal:
                g_xref = grads[len(aux_theta_tensors) + 0]
                g_uref = grads[len(aux_theta_tensors) + 1]

            # ---- Update auxiliary θ (Algorithm 2) ----
            if adapt_ancillary:
                _apply_update(aux_theta_tensors, g_theta_raw, v_theta, _names_theta())

            # ---- Nominal: only needed when nominal is adapted (not in Dubins paper setup) ----
            if not adapt_nominal:
                continue

            # Nominal: δz solve with upper gradients coming from auxiliary's total derivative wrt references
            # upper gradients for nominal x_hat: [g_xref, 0_for_b]
            def upper_gx_nom(x_hat_k: Tensor, k: int) -> Tensor:
                gx = g_xref[k]
                gb = torch.zeros((), device=gx.device, dtype=gx.dtype)
                return torch.cat([gx, gb.view(1)], dim=0)

            def upper_gxN_nom(x_hat_N: Tensor) -> Tensor:
                gx = g_xref[N]
                gb = torch.zeros((), device=gx.device, dtype=gx.dtype)
                return torch.cat([gx, gb.view(1)], dim=0)

            # Map dL/du_ref to dL/dv for nominal decision variables via u = ctrl.u(v)
            def upper_gv_nom(v_k: Tensor, k: int) -> Tensor:
                # With u as the decision variable, δu is directly constrained by the active set.
                return g_uref[k]

            def stage_hess_nom(x_hat_k: Tensor, v_k: Tensor, k: int):
                _, _, l_xx, l_vv, l_vx = stage_derivs_nom(x_hat_k, v_k, k)
                return l_xx, l_vv, l_vx

            def term_hess_nom(x_hat_N: Tensor):
                _, phi_xx = term_derivs_nom(x_hat_N)
                return phi_xx

            sens_nom = ddp_sensitivity(
                X=X_nom_hat,
                V=V_nom,
                f=f_hat_nom,
                ctrl=ctrl,
                f_jac=lambda xh, vk: dubins_augmented_jacobian(
                    xh, vk, cfg=dub_cfg, obs=obs, obs_beta=obs_beta, obs_agg=obs_agg,
                    db_cfg=DBaSConfig(barrier_type="inverse", alpha=alpha_n, gamma=gamma_n, eps=float(cfg["dbas"].get("eps", DBaSConfig().eps)))
                ),
                stage_hess=stage_hess_nom,
                terminal_hess=term_hess_nom,
                upper_grad_x=upper_gx_nom,
                upper_grad_u=upper_gv_nom,
                upper_grad_xN=upper_gxN_nom,
            )

            # Full IFT gradient wrt θ̄
            def f_hat_nom_grad(x_hat_k: Tensor, v_k: Tensor) -> Tensor:
                u_k = v_k
                xk = x_hat_k[:-1]
                bk = x_hat_k[-1]
                x_next, b_next = dbas_step(
                    x_k=xk, u_k=u_k, b_k=bk, f=f, h=h_nom,
                    cfg=db_cfg_nom()
                )
                return torch.cat([x_next, b_next.view(1)], dim=0)

            def stage_cost_nom_grad(x_hat_k: Tensor, v_k: Tensor, k: int) -> Tensor:
                u_k = v_k
                xk = x_hat_k[:-1]
                bk = x_hat_k[-1]
                dx = xk - target
                return (theta_bar.Q() * dx * dx).sum() + (theta_bar.R() * u_k * u_k).sum() + theta_bar.qb() * (bk * bk)

            def terminal_cost_nom_grad(x_hat_N: Tensor) -> Tensor:
                xN = x_hat_N[:-1]
                bN = x_hat_N[-1]
                dxN = xN - target
                # Paper: terminal cost includes barrier state penalty
                return (theta_bar.Qf() * dxN * dxN).sum() + theta_bar.qb() * (bN * bN)

            g_theta_bar = ift_gradient(
                inputs=IFTInputs(
                    X=X_nom_hat,
                    V=V_nom,
                    delta_X=sens_nom.delta_X,
                    delta_V=sens_nom.delta_V,
                    delta_lambda=sens_nom.delta_lambda,
                ),
                theta_tensors=theta_bar.tensors(),
                xi_fn=lambda: x_hat_nom0.detach(),
                f_fn=f_hat_nom_grad,
                stage_cost_fn=stage_cost_nom_grad,
                terminal_cost_fn=terminal_cost_nom_grad,
            )

            if adapt_nominal:
                _apply_update(theta_bar.tensors(), g_theta_bar, v_theta_bar, _names_theta_bar())

        # -------------------------
        # (D) True system step with disturbance + DBaS state update
        # -------------------------
        w = sample_disturbance(x, cfg=dub_cfg)
        x_next = f(x, u) + w
        _, b_next = dbas_step(x_k=x, u_k=u, b_k=b, f=f, h=h_aux, cfg=db_cfg_aux())

        # Nominal state propagation (tube structure): \bar{x}_{t+1} = f(\bar{x}_t, \bar{u}_0)
        u_bar0 = U_nom[0].detach()
        x_bar_next = f(x_bar, u_bar0)
        _, b_bar_next = dbas_step(x_k=x_bar, u_k=u_bar0, b_k=b_bar, f=f, h=h_nom, cfg=db_cfg_nom())

        # -------------------------
        # (E) Logging + warm-start shift
        # -------------------------
        traj.x_real.append(x.detach().cpu().numpy())
        traj.u_real.append(u.detach().cpu().numpy())
        traj.x_bar.append(x_bar.detach().cpu().numpy())
        traj.u_bar.append(u_bar0.detach().cpu().numpy())
        traj.b_real.append(b.detach().cpu().numpy())
        traj.loss.append(float(loss_val.cpu().item()))
        # Log adaptive parameters (if ancillary adapts)
        if adapt_ancillary:
            traj.Qa_history.append(theta.Q().detach().cpu().numpy().copy())
            traj.Ra_history.append(theta.R().detach().cpu().numpy().copy())
            traj.qba_history.append(float(theta.qb().detach().cpu().item()))

        def _shift(U_seq: Tensor) -> Tensor:
            return torch.cat([U_seq[1:], U_seq[-1:].clone()], dim=0)

        V_nom_ws = _shift(V_nom.detach())
        V_aux_ws = _shift(V_aux.detach())

        x, b = x_next.detach(), b_next.detach()
        x_bar, b_bar = x_bar_next.detach(), b_bar_next.detach()

    # Save arrays
    os.makedirs(run_dir, exist_ok=True)
    np.save(os.path.join(run_dir, "x_real.npy"), np.stack(traj.x_real, axis=0))
    np.save(os.path.join(run_dir, "u_real.npy"), np.stack(traj.u_real, axis=0))
    np.save(os.path.join(run_dir, "x_bar.npy"), np.stack(traj.x_bar, axis=0))
    np.save(os.path.join(run_dir, "u_bar.npy"), np.stack(traj.u_bar, axis=0))
    np.save(os.path.join(run_dir, "b_real.npy"), np.stack(traj.b_real, axis=0))
    np.save(os.path.join(run_dir, "loss.npy"), np.asarray(traj.loss, dtype=np.float64))
    # Save adaptive parameters if they were tracked
    if len(traj.Qa_history) > 0:
        np.save(os.path.join(run_dir, "Qa_history.npy"), np.stack(traj.Qa_history, axis=0))
        np.save(os.path.join(run_dir, "Ra_history.npy"), np.stack(traj.Ra_history, axis=0))
        np.save(os.path.join(run_dir, "qba_history.npy"), np.asarray(traj.qba_history, dtype=np.float64))

    summary = {
        "system": system_cfg["name"],
        "H": H,
        "N": N,
        "final_state": traj.x_real[-1].tolist(),
        "final_barrier_state": float(np.array(traj.b_real[-1]).reshape(-1)[0]),
        "final_loss": float(traj.loss[-1]),
    }

    return {"summary": summary}


def _run_dubins_paper(cfg: Dict[str, Any], *, device: torch.device, run_dir: str) -> Dict[str, Any]:
    """Paper-aligned Dubins experiment.

    - Nominal MPC parameters fixed (Appendix:experiments, Dubins Vehicle)
    - Ancillary MPC adapts via minimizing eq (dt_mpc_loss)
    - Inverse barrier, alpha=0, gamma=0 fixed
    - Projected gradient descent constraints: Q>=0, R>=1e-4, q_b in [0,1]
    """
    system_cfg = cfg["system"]
    dtype = torch.float64 if bool(cfg.get("use_float64", False)) else torch.float32
    dt = float(system_cfg["dt"])
    N = int(system_cfg["horizon_N"])
    H = int(system_cfg["task_horizon_H"])

    dub_cfg = DubinsConfig(
        dt=dt,
        v_max=float(system_cfg["control_bounds"]["v_max"]),
        omega_max=float(system_cfg["control_bounds"]["omega_max"]),
        w_low=tuple(system_cfg["disturbance"]["w_low"]),
        w_high=tuple(system_cfg["disturbance"]["w_high"]),
        x_target=tuple(system_cfg["target"]),
    )

    # Obstacle field (paper Dubins: multiple circles; use smooth-min for differentiability)
    env_cfg = cfg.get("environment", {})
    obs_beta = float(env_cfg.get("obstacle_smoothmin_beta", 20.0))
    obs_agg = str(env_cfg.get("obstacle_aggregation", "min"))
    if "obstacles" in env_cfg:
        obs_list = [CircleObstacle(center=tuple(o["center"]), radius=float(o["radius"])) for o in env_cfg["obstacles"]]
        obs = obs_list
        if obs_agg == "smoothmin":
            h = lambda x_in: h_multi_circle_obstacles(x_in, obstacles=obs_list, beta=obs_beta)
        else:
            h = lambda x_in: h_min_circle_obstacles(x_in, obstacles=obs_list)
    else:
        # Fallback to single obstacle if user didn't provide the obstacle list
        obs_cfg = env_cfg.get("obstacle", {"center": [5.0, 5.0], "radius": 1.5})
        obs = CircleObstacle(center=tuple(obs_cfg["center"]), radius=float(obs_cfg["radius"]))
        h = lambda x_in: h_circle_obstacle(x_in, obs=obs)

    # Fixed DBaS params for Dubins in paper
    db_cfg = DBaSConfig(
        barrier_type="inverse",
        alpha=torch.tensor(0.0, device=device, dtype=dtype),
        gamma=torch.tensor(0.0, device=device, dtype=dtype),
        eps=float(cfg["dbas"].get("eps", DBaSConfig().eps)),
    )

    # Dynamics f(x,u)
    f = lambda x, u: dubins_step(x, u, cfg=dub_cfg)

    # Control param (smooth box)
    v_min = float(system_cfg["control_bounds"].get("v_min", -dub_cfg.v_max))
    u_min = torch.tensor([v_min, -dub_cfg.omega_max], device=device, dtype=dtype)
    u_max = torch.tensor([dub_cfg.v_max, dub_cfg.omega_max], device=device, dtype=dtype)
    ctrl = BoxClampControl(u_min=u_min, u_max=u_max)

    # Nominal fixed weights (Appendix:experiments)
    target = torch.tensor(dub_cfg.x_target, device=device, dtype=dtype)
    Qn = torch.tensor(cfg["cost_nominal"]["Q"], device=device, dtype=dtype)
    Rn = torch.tensor(cfg["cost_nominal"]["R"], device=device, dtype=dtype)
    Qfn = torch.tensor(cfg["cost_nominal"]["Qf"], device=device, dtype=dtype)
    qbn = torch.tensor(float(cfg["cost_nominal"]["q_b"]), device=device, dtype=dtype)

    # Ancillary learnable weights (init ones)
    # Use config values if provided, otherwise default to [1,1,1]
    aux_cfg = cfg.get("cost_auxiliary", {})
    if "Q" in aux_cfg:
        Qa = torch.tensor(aux_cfg["Q"], device=device, dtype=dtype)
    else:
        Qa = torch.ones(3, device=device, dtype=dtype)
    if "R" in aux_cfg:
        Ra = torch.tensor(aux_cfg["R"], device=device, dtype=dtype)
    else:
        Ra = torch.ones(2, device=device, dtype=dtype)
    if "q_b" in aux_cfg:
        qba = torch.tensor(float(aux_cfg["q_b"]), device=device, dtype=dtype)
    else:
        qba = torch.ones((), device=device, dtype=dtype)

    # Momentum buffers
    adapt_cfg = cfg.get("adaptation", {})
    eta = float(adapt_cfg.get("lr_eta", 1e-2))
    mom = float(adapt_cfg.get("momentum", 0.9))
    vQ = torch.zeros_like(Qa)
    vR = torch.zeros_like(Ra)
    vqb = torch.zeros_like(qba)

    # iLQR configs
    # Speed: paper uses iLQR; we keep only alpha=1.0 line search by default (can be expanded if needed).
    alphas = tuple(system_cfg.get("line_search_alphas", [1.0]))
    ilqr_nom = ILQRConfig(
        horizon=N, nx=4, nu=2,
        max_iter=int(system_cfg.get("nominal_max_iter", 10)),
        tol=1e-3,
        line_search_alphas=alphas,
    )
    ilqr_aux = ILQRConfig(
        horizon=N, nx=4, nu=2,
        max_iter=int(system_cfg.get("aux_max_iter", 10)),
        tol=1e-3,
        line_search_alphas=alphas,
    )

    # Initial states
    # Paper: start at origin facing upper-right (theta=pi/4)
    x = torch.tensor([0.0, 0.0, float(np.pi / 4)], device=device, dtype=dtype)
    x_bar = x.clone()
    b = dbas_init_b0(x, h=h, cfg=db_cfg)
    b_bar = dbas_init_b0(x_bar, h=h, cfg=db_cfg)

    # Warm starts in v-space
    V_nom_ws = torch.zeros(N, 2, device=device, dtype=dtype)
    V_aux_ws = torch.zeros(N, 2, device=device, dtype=dtype)

    traj = ExperimentTrajectories(
        x_real=[], u_real=[], x_bar=[], u_bar=[], loss=[], b_real=[],
        Qa_history=[], Ra_history=[], qba_history=[]
    )

    debug_num = bool(cfg.get("debug_numerics", False))

    def _ensure_finite(t: torch.Tensor, name: str, *, step: int) -> None:
        if torch.isfinite(t).all():
            return
        finite = torch.isfinite(t)
        bad = (~finite).sum().item()
        if t.numel() and finite.any():
            vals = t[finite]
            t_min = vals.min().item()
            t_max = vals.max().item()
        else:
            t_min = float("nan")
            t_max = float("nan")
        print(f"[NUMERIC-FAIL] t={step}: {name} has {bad} non-finite entries (min={t_min}, max={t_max})", flush=True)
        raise FloatingPointError(f"Non-finite detected at t={step} in {name}")

    for t in range(H):
        if (t % 25) == 0:
            print(f"[step {t}/{H}] running...", flush=True)

        if debug_num:
            _ensure_finite(x, "x", step=t)
            _ensure_finite(x_bar, "x_bar", step=t)
            _ensure_finite(b, "b", step=t)
            _ensure_finite(b_bar, "b_bar", step=t)

        # Nominal MPC solve (fixed)
        x_hat_nom0 = torch.cat([x_bar, b_bar.view(1)], dim=0)

        def f_hat_nom(x_hat_k: Tensor, v_k: Tensor) -> Tensor:
            u_k = v_k
            xk = x_hat_k[:-1]
            bk = x_hat_k[-1]
            x_next, b_next = dbas_step(x_k=xk, u_k=u_k, b_k=bk, f=f, h=h, cfg=db_cfg)
            return torch.cat([x_next, b_next.view(1)], dim=0)

        def stage_cost_nom(x_hat_k: Tensor, v_k: Tensor, k: int) -> Tensor:
            u_k = v_k
            dx = x_hat_k[:-1] - target
            bk = x_hat_k[-1]
            return (Qn * dx * dx).sum() + (Rn * u_k * u_k).sum() + qbn * (bk * bk)

        def terminal_cost_nom(x_hat_N: Tensor) -> Tensor:
            dxN = x_hat_N[:-1] - target
            bN = x_hat_N[-1]
            return (Qfn * dxN * dxN).sum() + qbn * (bN * bN)

        def stage_derivs_nom(x_hat_k: Tensor, v_k: Tensor, k: int):
            return nominal_cost_derivs_u(x_hat=x_hat_k, u=v_k, target=target, Q=Qn, R=Rn, qb=qbn)

        def term_derivs_nom(x_hat_N: Tensor):
            # terminal includes b_N^2 in paper
            phi_x, phi_xx = nominal_terminal_derivs(x_hat_N=x_hat_N, target=target, Qf=Qfn)
            phi_x[-1] = 2.0 * qbn * x_hat_N[-1]
            phi_xx[-1, -1] = 2.0 * qbn
            return phi_x, phi_xx

        X_nom_hat, V_nom = ilqr_solve(
            x0=x_hat_nom0,
            V_init=V_nom_ws,
            cfg=ilqr_nom,
            f=f_hat_nom,
            ctrl=ctrl,
            f_jac=lambda xh, vk: dubins_augmented_jacobian(xh, vk, cfg=dub_cfg, obs=obs, obs_beta=obs_beta, obs_agg=obs_agg, db_cfg=db_cfg),
            stage_cost=stage_cost_nom,
            terminal_cost=terminal_cost_nom,
            stage_derivs=stage_derivs_nom,
            terminal_derivs=term_derivs_nom,
            debug=debug_num,
            debug_name="iLQR-nominal",
        )
        U_nom = V_nom
        if debug_num:
            _ensure_finite(X_nom_hat, "X_nom_hat", step=t)
            _ensure_finite(V_nom, "U_nom", step=t)

        # Aux MPC solve (tracking)
        X_ref = X_nom_hat[:, :-1]
        U_ref = U_nom
        x_hat0 = torch.cat([x, b.view(1)], dim=0)

        def f_hat_aux(x_hat_k: Tensor, v_k: Tensor) -> Tensor:
            u_k = v_k
            xk = x_hat_k[:-1]
            bk = x_hat_k[-1]
            x_next, b_next = dbas_step(x_k=xk, u_k=u_k, b_k=bk, f=f, h=h, cfg=db_cfg)
            return torch.cat([x_next, b_next.view(1)], dim=0)

        def stage_cost_aux(x_hat_k: Tensor, v_k: Tensor, k: int) -> Tensor:
            u_k = v_k
            dx = x_hat_k[:-1] - X_ref[k]
            du = u_k - U_ref[k]
            bk = x_hat_k[-1]
            return (Qa * dx * dx).sum() + (Ra * du * du).sum() + qba * (bk * bk)

        def terminal_cost_aux(x_hat_N: Tensor) -> Tensor:
            dxN = x_hat_N[:-1] - X_ref[N]
            bN = x_hat_N[-1]
            return (Qa * dxN * dxN).sum() + qba * (bN * bN)

        def stage_derivs_aux(x_hat_k: Tensor, v_k: Tensor, k: int):
            return auxiliary_cost_derivs_u(x_hat=x_hat_k, u=v_k, x_ref=X_ref[k], u_ref=U_ref[k], Q=Qa, R=Ra, qb=qba)

        def term_derivs_aux(x_hat_N: Tensor):
            phi_x, phi_xx = auxiliary_terminal_derivs(x_hat_N=x_hat_N, x_ref_N=X_ref[N], Qf=Qa)
            phi_x[-1] = 2.0 * qba * x_hat_N[-1]
            phi_xx[-1, -1] = 2.0 * qba
            return phi_x, phi_xx

        X_aux_hat, V_aux = ilqr_solve(
            x0=x_hat0,
            V_init=V_aux_ws,
            cfg=ilqr_aux,
            f=f_hat_aux,
            ctrl=ctrl,
            f_jac=lambda xh, vk: dubins_augmented_jacobian(xh, vk, cfg=dub_cfg, obs=obs, obs_beta=obs_beta, obs_agg=obs_agg, db_cfg=db_cfg),
            stage_cost=stage_cost_aux,
            terminal_cost=terminal_cost_aux,
            stage_derivs=stage_derivs_aux,
            terminal_derivs=term_derivs_aux,
            debug=debug_num,
            debug_name="iLQR-aux",
        )
        U_aux = V_aux
        if debug_num:
            _ensure_finite(X_aux_hat, "X_aux_hat", step=t)
            _ensure_finite(V_aux, "U_aux", step=t)

        # Upper loss (eq dt_mpc_loss)
        x_aux = X_aux_hat[:, :-1]
        b_aux = X_aux_hat[:, -1]
        x_nom = X_nom_hat[:, :-1]
        L = (x_aux - x_nom).pow(2).sum() + (b_aux.pow(2)).sum()
        if debug_num:
            _ensure_finite(L, "upper_loss_L", step=t)

        # DOC δz solve
        def stage_hess_aux(x_hat_k: Tensor, v_k: Tensor, k: int):
            _, _, l_xx, l_vv, l_vx = stage_derivs_aux(x_hat_k, v_k, k)
            return l_xx, l_vv, l_vx

        def term_hess_aux(x_hat_N: Tensor):
            _, phi_xx = term_derivs_aux(x_hat_N)
            return phi_xx

        def upper_gx_aux(x_hat_k: Tensor, k: int) -> Tensor:
            # ∂L/∂x = 2(x - x_nom), ∂L/∂b = 2b
            gx = 2.0 * (x_hat_k[:-1] - x_nom[k])
            gb = 2.0 * x_hat_k[-1]
            return torch.cat([gx, gb.view(1)], dim=0)

        def upper_gv_zero(v_k: Tensor, k: int) -> Tensor:
            return torch.zeros_like(v_k)

        def upper_gxN_aux(x_hat_N: Tensor) -> Tensor:
            gx = 2.0 * (x_hat_N[:-1] - x_nom[N])
            gb = 2.0 * x_hat_N[-1]
            return torch.cat([gx, gb.view(1)], dim=0)

        sens = ddp_sensitivity(
            X=X_aux_hat,
            V=V_aux,
            f=f_hat_aux,
            ctrl=ctrl,
            f_jac=lambda xh, vk: dubins_augmented_jacobian(xh, vk, cfg=dub_cfg, obs=obs, obs_beta=obs_beta, obs_agg=obs_agg, db_cfg=db_cfg),
            stage_hess=stage_hess_aux,
            terminal_hess=term_hess_aux,
            upper_grad_x=upper_gx_aux,
            upper_grad_u=upper_gv_zero,
            upper_grad_xN=upper_gxN_aux,
        )
        if debug_num:
            _ensure_finite(sens.delta_X, "delta_X", step=t)
            _ensure_finite(sens.delta_V, "delta_U", step=t)
            _ensure_finite(sens.delta_lambda, "delta_lambda", step=t)

        # Fast analytic DOC gradient for (Q,R,q_b) of ancillary cost.
        # Uses ℒ_{θx}δx + ℒ_{θu}δu + φ_{θx}δxN; dynamics do not depend on θ here.
        dX = sens.delta_X
        dV = sens.delta_V
        dU = dV

        dx = x_aux - x_nom
        du = U_aux - U_ref
        db = X_aux_hat[:, -1]
        ddb = dX[:, -1]

        gQ = (2.0 * dx[:-1] * dX[:-1, :-1]).sum(dim=0) + 2.0 * dx[-1] * dX[-1, :-1]
        gR = (2.0 * du * dU).sum(dim=0)
        gqb = (2.0 * db[:-1] * ddb[:-1]).sum() + 2.0 * db[-1] * ddb[-1]

        # Momentum + update + projection (paper)
        vQ = mom * vQ + gQ
        vR = mom * vR + gR
        vqb = mom * vqb + gqb
        Qa = (Qa - eta * vQ).clamp(min=0.0)
        Ra = (Ra - eta * vR).clamp(min=1e-4)
        qba = (qba - eta * vqb).clamp(min=0.0, max=1.0)
        if debug_num:
            _ensure_finite(Qa, "Qa", step=t)
            _ensure_finite(Ra, "Ra", step=t)
            _ensure_finite(qba, "qba", step=t)

        # Apply control
        u = U_aux[0]
        if debug_num:
            _ensure_finite(u, "u_apply", step=t)
        w = sample_disturbance(x, cfg=dub_cfg)
        x_next = f(x, u) + w
        _, b_next = dbas_step(x_k=x, u_k=u, b_k=b, f=f, h=h, cfg=db_cfg)

        # Nominal propagation
        u_bar0 = U_nom[0].detach()
        x_bar_next = f(x_bar, u_bar0)
        _, b_bar_next = dbas_step(x_k=x_bar, u_k=u_bar0, b_k=b_bar, f=f, h=h, cfg=db_cfg)

        # Log
        traj.x_real.append(x.detach().cpu().numpy())
        traj.u_real.append(u.detach().cpu().numpy())
        traj.x_bar.append(x_bar.detach().cpu().numpy())
        traj.u_bar.append(u_bar0.detach().cpu().numpy())
        traj.b_real.append(b.detach().cpu().numpy())
        traj.loss.append(float(L.detach().cpu().item()))
        # Log adaptive parameters
        traj.Qa_history.append(Qa.detach().cpu().numpy().copy())
        traj.Ra_history.append(Ra.detach().cpu().numpy().copy())
        traj.qba_history.append(float(qba.detach().cpu().item()))

        # Shift warm starts
        def _shift(U_seq: Tensor) -> Tensor:
            return torch.cat([U_seq[1:], U_seq[-1:].clone()], dim=0)

        V_nom_ws = _shift(V_nom.detach())
        V_aux_ws = _shift(V_aux.detach())

        x, b = x_next.detach(), b_next.detach()
        x_bar, b_bar = x_bar_next.detach(), b_bar_next.detach()

    # Save arrays
    os.makedirs(run_dir, exist_ok=True)
    np.save(os.path.join(run_dir, "x_real.npy"), np.stack(traj.x_real, axis=0))
    np.save(os.path.join(run_dir, "u_real.npy"), np.stack(traj.u_real, axis=0))
    np.save(os.path.join(run_dir, "x_bar.npy"), np.stack(traj.x_bar, axis=0))
    np.save(os.path.join(run_dir, "u_bar.npy"), np.stack(traj.u_bar, axis=0))
    np.save(os.path.join(run_dir, "b_real.npy"), np.stack(traj.b_real, axis=0))
    np.save(os.path.join(run_dir, "loss.npy"), np.asarray(traj.loss, dtype=np.float64))
    # Save adaptive parameters
    if len(traj.Qa_history) > 0:
        np.save(os.path.join(run_dir, "Qa_history.npy"), np.stack(traj.Qa_history, axis=0))
        np.save(os.path.join(run_dir, "Ra_history.npy"), np.stack(traj.Ra_history, axis=0))
        np.save(os.path.join(run_dir, "qba_history.npy"), np.asarray(traj.qba_history, dtype=np.float64))

    summary = {
        "system": "dubins",
        "H": H,
        "N": N,
        "final_state": traj.x_real[-1].tolist(),
        "final_barrier_state": float(np.array(traj.b_real[-1]).reshape(-1)[0]),
        "final_loss": float(traj.loss[-1]),
        "note": "Dubins run aligned to paper: nominal fixed, ancillary adapts, alpha=0, gamma=0.",
    }
    return {"summary": summary}

