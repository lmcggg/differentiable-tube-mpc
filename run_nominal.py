from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import torch
import yaml


def _ensure_project_root_on_path() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _wrap_angle(err: torch.Tensor) -> torch.Tensor:
    # maps to (-pi, pi]
    return torch.atan2(torch.sin(err), torch.cos(err))


def run_nominal_once(cfg: Dict[str, Any], *, device: torch.device, run_dir: str) -> Dict[str, Any]:
    from diff_tube_mpc_strict_pt.core.barrier import DBaSConfig, dbas_init_b0, dbas_step
    from diff_tube_mpc_strict_pt.core.control import BoxClampControl
    from diff_tube_mpc_strict_pt.core.cost_derivs import nominal_cost_derivs_u
    from diff_tube_mpc_strict_pt.core.ddp import ILQRConfig, ilqr_solve
    from diff_tube_mpc_strict_pt.core.systems.dubins import DubinsConfig, dubins_step
    from diff_tube_mpc_strict_pt.core.systems.dubins_aug_jac import dubins_augmented_jacobian
    from diff_tube_mpc_strict_pt.core.systems.dubins_obstacles import (
        CircleObstacle,
        h_circle_obstacle,
        h_min_circle_obstacles,
        h_multi_circle_obstacles,
    )

    system_cfg = cfg["system"]
    assert system_cfg["name"] == "dubins"
    N = int(system_cfg["horizon_N"])
    dt = float(system_cfg["dt"])

    dub_cfg = DubinsConfig(
        dt=dt,
        v_max=float(system_cfg["control_bounds"]["v_max"]),
        omega_max=float(system_cfg["control_bounds"]["omega_max"]),
        w_low=(0.0, 0.0, 0.0),
        w_high=(0.0, 0.0, 0.0),
        x_target=tuple(system_cfg["target"]),
    )

    # Obstacles
    env_cfg = cfg.get("environment", {})
    obs_beta = float(env_cfg.get("obstacle_smoothmin_beta", 20.0))
    obs_agg = str(env_cfg.get("obstacle_aggregation", "min"))
    obs: Any
    if "obstacles" in env_cfg:
        obs_list = [CircleObstacle(center=tuple(o["center"]), radius=float(o["radius"])) for o in env_cfg["obstacles"]]
        obs = obs_list
        if obs_agg == "smoothmin":
            h = lambda x_in: h_multi_circle_obstacles(x_in, obstacles=obs_list, beta=obs_beta)
        else:
            h = lambda x_in: h_min_circle_obstacles(x_in, obstacles=obs_list)
    elif "obstacle" in env_cfg:
        o = env_cfg["obstacle"]
        obs = CircleObstacle(center=tuple(o["center"]), radius=float(o["radius"]))
        h = lambda x_in: h_circle_obstacle(x_in, obs=obs)
    else:
        obs = []
        h = lambda x_in: torch.ones((), device=x_in.device, dtype=x_in.dtype)

    # Control bounds
    v_min = float(system_cfg["control_bounds"].get("v_min", -dub_cfg.v_max))
    u_min = torch.tensor([v_min, -dub_cfg.omega_max], device=device, dtype=torch.float32)
    u_max = torch.tensor([dub_cfg.v_max, dub_cfg.omega_max], device=device, dtype=torch.float32)
    ctrl = BoxClampControl(u_min=u_min, u_max=u_max)

    # Start state per paper
    x0 = torch.tensor([0.0, 0.0, float(np.pi / 4)], device=device, dtype=torch.float32)

    # DBaS config
    db_cfg = DBaSConfig(
        barrier_type=str(cfg["dbas"]["barrier_type"]),
        alpha=torch.tensor(float(cfg["dbas"]["alpha"]), device=device),
        gamma=torch.tensor(float(cfg["dbas"]["gamma"]), device=device),
        eps=float(cfg["dbas"].get("eps", DBaSConfig().eps)),
    )
    b0 = dbas_init_b0(x0, h=h, cfg=db_cfg)
    x_hat0 = torch.cat([x0, b0.view(1)], dim=0)

    # Nominal cost weights
    target = torch.tensor(dub_cfg.x_target, device=device, dtype=torch.float32)
    Qn = torch.tensor(cfg["cost_nominal"]["Q"], device=device, dtype=torch.float32)
    Rn = torch.tensor(cfg["cost_nominal"]["R"], device=device, dtype=torch.float32)
    Qfn = torch.tensor(cfg["cost_nominal"]["Qf"], device=device, dtype=torch.float32)
    qbn = torch.tensor(float(cfg["cost_nominal"]["q_b"]), device=device, dtype=torch.float32)

    f = lambda x, u: dubins_step(x, u, cfg=dub_cfg)

    def f_hat(x_hat_k: torch.Tensor, u_k: torch.Tensor) -> torch.Tensor:
        xk = x_hat_k[:-1]
        bk = x_hat_k[-1]
        x_next, b_next = dbas_step(x_k=xk, u_k=u_k, b_k=bk, f=f, h=h, cfg=db_cfg)
        return torch.cat([x_next, b_next.view(1)], dim=0)

    def feasible(x_hat_k: torch.Tensor, _k: int) -> bool:
        # Require h(x)>0 (strictly) along the trajectory to avoid obstacle penetration.
        xk = x_hat_k[:-1]
        hk = h(xk)
        # hk is scalar tensor
        return bool((hk > 0.0).item())

    def stage_cost(x_hat_k: torch.Tensor, u_k: torch.Tensor, _k: int) -> torch.Tensor:
        xk = x_hat_k[:-1]
        bk = x_hat_k[-1]
        dx = xk - target
        dx = torch.cat([dx[:2], _wrap_angle(dx[2]).view(1)], dim=0)
        return (Qn * dx * dx).sum() + (Rn * u_k * u_k).sum() + qbn * (bk * bk)

    def terminal_cost(x_hat_N: torch.Tensor) -> torch.Tensor:
        xN = x_hat_N[:-1]
        bN = x_hat_N[-1]
        dx = xN - target
        dx = torch.cat([dx[:2], _wrap_angle(dx[2]).view(1)], dim=0)
        return (Qfn * dx * dx).sum() + qbn * (bN * bN)

    def stage_derivs(x_hat_k: torch.Tensor, u_k: torch.Tensor, k: int):
        # Derivatives are exact for x,y,b and u. For theta we use wrapped error (d/dtheta ~ 1).
        # Implement by feeding a "wrapped target" locally (equivalent to using wrapped error).
        xk = x_hat_k[:-1]
        target_k = target.clone()
        target_k[2] = xk[2] - _wrap_angle(xk[2] - target[2])
        return nominal_cost_derivs_u(x_hat=x_hat_k, u=u_k, target=target_k, Q=Qn, R=Rn, qb=qbn)

    def term_derivs(x_hat_N: torch.Tensor):
        from diff_tube_mpc_strict_pt.core.cost_derivs import nominal_terminal_derivs

        xN = x_hat_N[:-1]
        target_k = target.clone()
        target_k[2] = xN[2] - _wrap_angle(xN[2] - target[2])
        phi_x, phi_xx = nominal_terminal_derivs(x_hat_N=x_hat_N, target=target_k, Qf=Qfn)
        # add qb*bN^2 terminal term (paper)
        phi_x[-1] = phi_x[-1] + 2.0 * qbn * x_hat_N[-1]
        phi_xx[-1, -1] = phi_xx[-1, -1] + 2.0 * qbn
        return phi_x, phi_xx

    ilqr_cfg = ILQRConfig(
        horizon=N,
        nx=4,
        nu=2,
        max_iter=int(system_cfg.get("nominal_max_iter", 10)),
        tol=1e-3,
        reg=float(system_cfg.get("ilqr_reg", 1e-6)),
        line_search_alphas=tuple(system_cfg.get("line_search_alphas", [1.0, 0.5, 0.25, 0.1])),
    )
    # Better warm-start: forward at v=v_max, omega=0
    U_ws = torch.zeros(N, 2, device=device, dtype=torch.float32)
    U_ws[:, 0] = float(dub_cfg.v_max)

    X_hat, U = ilqr_solve(
        x0=x_hat0,
        V_init=U_ws,
        cfg=ilqr_cfg,
        f=f_hat,
        ctrl=ctrl,
        f_jac=lambda xh, uk: dubins_augmented_jacobian(xh, uk, cfg=dub_cfg, obs=obs, obs_beta=obs_beta, obs_agg=obs_agg, db_cfg=db_cfg),
        stage_cost=stage_cost,
        terminal_cost=terminal_cost,
        stage_derivs=stage_derivs,
        terminal_derivs=term_derivs,
    )

    # Save nominal-only single solve artifacts for inspection
    x_bar = X_hat[:, :-1].detach().cpu().numpy()  # [N+1,3]
    u_bar = U.detach().cpu().numpy()              # [N,2]
    os.makedirs(run_dir, exist_ok=True)
    np.save(os.path.join(run_dir, "x_bar_single.npy"), x_bar)
    np.save(os.path.join(run_dir, "u_bar_single.npy"), u_bar)

    return {
        "summary": {
            "system": "dubins",
            "mode": "nominal_only",
            "N": N,
            "x0": x_bar[0].tolist(),
            "xN": x_bar[-1].tolist(),
        }
    }


def run_nominal_receding(cfg: Dict[str, Any], *, device: torch.device, run_dir: str) -> Dict[str, Any]:
    """Run *nominal MPC only* in receding-horizon fashion (no disturbances).

    This matches how the nominal trajectory in the paper is generated over the task horizon H.
    """
    from diff_tube_mpc_strict_pt.core.barrier import DBaSConfig, dbas_init_b0, dbas_step
    from diff_tube_mpc_strict_pt.core.control import BoxClampControl
    from diff_tube_mpc_strict_pt.core.cost_derivs import nominal_cost_derivs_u, nominal_terminal_derivs
    from diff_tube_mpc_strict_pt.core.ddp import ILQRConfig, ilqr_solve
    from diff_tube_mpc_strict_pt.core.systems.dubins import DubinsConfig, dubins_step
    from diff_tube_mpc_strict_pt.core.systems.dubins_aug_jac import dubins_augmented_jacobian
    from diff_tube_mpc_strict_pt.core.systems.dubins_obstacles import (
        CircleObstacle,
        h_circle_obstacle,
        h_min_circle_obstacles,
        h_multi_circle_obstacles,
        h_circle_obstacle as h_one,
    )

    system_cfg = cfg["system"]
    assert system_cfg["name"] == "dubins"
    N = int(system_cfg["horizon_N"])
    H = int(system_cfg["task_horizon_H"])
    dt = float(system_cfg["dt"])

    dtype = torch.float64

    dub_cfg = DubinsConfig(
        dt=dt,
        v_max=float(system_cfg["control_bounds"]["v_max"]),
        omega_max=float(system_cfg["control_bounds"]["omega_max"]),
        w_low=(0.0, 0.0, 0.0),
        w_high=(0.0, 0.0, 0.0),
        x_target=tuple(system_cfg["target"]),
    )

    # Obstacles + h(x)
    env_cfg = cfg.get("environment", {})
    obs_beta = float(env_cfg.get("obstacle_smoothmin_beta", 20.0))
    obs_agg = str(env_cfg.get("obstacle_aggregation", "min"))
    obstacles: List[CircleObstacle] = []
    if "obstacles" in env_cfg:
        obstacles = [CircleObstacle(center=tuple(o["center"]), radius=float(o["radius"])) for o in env_cfg["obstacles"]]
        if obs_agg == "smoothmin":
            h = lambda x_in: h_multi_circle_obstacles(x_in, obstacles=obstacles, beta=obs_beta)
        else:
            h = lambda x_in: h_min_circle_obstacles(x_in, obstacles=obstacles)
    elif "obstacle" in env_cfg:
        o = env_cfg["obstacle"]
        obstacles = [CircleObstacle(center=tuple(o["center"]), radius=float(o["radius"]))]
        h = lambda x_in: h_one(x_in, obs=obstacles[0])
    else:
        h = lambda x_in: torch.ones((), device=x_in.device, dtype=x_in.dtype)

    # Control bounds
    v_min = float(system_cfg["control_bounds"].get("v_min", -dub_cfg.v_max))
    u_min = torch.tensor([v_min, -dub_cfg.omega_max], device=device, dtype=dtype)
    u_max = torch.tensor([dub_cfg.v_max, dub_cfg.omega_max], device=device, dtype=dtype)
    ctrl = BoxClampControl(u_min=u_min, u_max=u_max)

    # Start state per paper
    x = torch.tensor([0.0, 0.0, float(np.pi / 4)], device=device, dtype=dtype)

    # DBaS config (fixed for nominal, paper Dubins)
    db_cfg = DBaSConfig(
        barrier_type=str(cfg["dbas"]["barrier_type"]),
        alpha=torch.tensor(float(cfg["dbas"]["alpha"]), device=device, dtype=dtype),
        gamma=torch.tensor(float(cfg["dbas"]["gamma"]), device=device, dtype=dtype),
        eps=float(cfg["dbas"].get("eps", DBaSConfig().eps)),
    )
    b = dbas_init_b0(x, h=h, cfg=db_cfg)

    # Nominal cost weights
    target = torch.tensor(dub_cfg.x_target, device=device, dtype=dtype)
    Qn = torch.tensor(cfg["cost_nominal"]["Q"], device=device, dtype=dtype)
    Rn = torch.tensor(cfg["cost_nominal"]["R"], device=device, dtype=dtype)
    Qfn = torch.tensor(cfg["cost_nominal"]["Qf"], device=device, dtype=dtype)
    qbn = torch.tensor(float(cfg["cost_nominal"]["q_b"]), device=device, dtype=dtype)

    f = lambda xx, uu: dubins_step(xx, uu, cfg=dub_cfg)

    def f_hat(x_hat_k: torch.Tensor, u_k: torch.Tensor) -> torch.Tensor:
        xk = x_hat_k[:-1]
        bk = x_hat_k[-1]
        x_next, b_next = dbas_step(x_k=xk, u_k=u_k, b_k=bk, f=f, h=h, cfg=db_cfg)
        return torch.cat([x_next, b_next.view(1)], dim=0)

    def feasible(x_hat_k: torch.Tensor, _k: int) -> bool:
        # Require h(x)>0 (strictly) along the trajectory.
        xk = x_hat_k[:-1]
        hk = h(xk)
        return bool((hk > 0.0).item())

    def stage_cost(x_hat_k: torch.Tensor, u_k: torch.Tensor, _k: int) -> torch.Tensor:
        xk = x_hat_k[:-1]
        bk = x_hat_k[-1]
        dx = xk - target
        dx = torch.cat([dx[:2], _wrap_angle(dx[2]).view(1)], dim=0)
        return (Qn * dx * dx).sum() + (Rn * u_k * u_k).sum() + qbn * (bk * bk)

    def terminal_cost(x_hat_N: torch.Tensor) -> torch.Tensor:
        xN = x_hat_N[:-1]
        bN = x_hat_N[-1]
        dx = xN - target
        dx = torch.cat([dx[:2], _wrap_angle(dx[2]).view(1)], dim=0)
        return (Qfn * dx * dx).sum() + qbn * (bN * bN)

    def stage_derivs(x_hat_k: torch.Tensor, u_k: torch.Tensor, k: int):
        xk = x_hat_k[:-1]
        target_k = target.clone()
        target_k[2] = xk[2] - _wrap_angle(xk[2] - target[2])
        return nominal_cost_derivs_u(x_hat=x_hat_k, u=u_k, target=target_k, Q=Qn, R=Rn, qb=qbn)

    def term_derivs(x_hat_N: torch.Tensor):
        xN = x_hat_N[:-1]
        target_k = target.clone()
        target_k[2] = xN[2] - _wrap_angle(xN[2] - target[2])
        phi_x, phi_xx = nominal_terminal_derivs(x_hat_N=x_hat_N, target=target_k, Qf=Qfn)
        phi_x[-1] = phi_x[-1] + 2.0 * qbn * x_hat_N[-1]
        phi_xx[-1, -1] = phi_xx[-1, -1] + 2.0 * qbn
        return phi_x, phi_xx

    ilqr_cfg = ILQRConfig(
        horizon=N,
        nx=4,
        nu=2,
        max_iter=int(system_cfg.get("nominal_max_iter", 10)),
        tol=1e-3,
        reg=float(system_cfg.get("ilqr_reg", 1e-6)),
        line_search_alphas=tuple(system_cfg.get("line_search_alphas", [1.0, 0.5, 0.25, 0.1])),
    )

    # Warm start: forward v=v_max, omega=0
    U_ws = torch.zeros(N, 2, device=device, dtype=dtype)
    U_ws[:, 0] = float(dub_cfg.v_max)

    xs = []
    us = []
    bs = []

    success_r = 0.25
    collided = False
    success = False
    success_t = None

    for t in range(H):
        if (t % 25) == 0:
            print(f"[nominal step {t}/{H}] ...", flush=True)
        x_hat0 = torch.cat([x, b.view(1)], dim=0)
        X_hat, U = ilqr_solve(
            x0=x_hat0,
            V_init=U_ws,
            cfg=ilqr_cfg,
            f=f_hat,
            ctrl=ctrl,
            f_jac=lambda xh, uk: dubins_augmented_jacobian(xh, uk, cfg=dub_cfg, obs=obstacles, obs_beta=obs_beta, obs_agg=obs_agg, db_cfg=db_cfg),
            stage_cost=stage_cost,
            terminal_cost=terminal_cost,
            stage_derivs=stage_derivs,
            terminal_derivs=term_derivs,
        )

        u0 = U[0]
        x_next = f(x, u0)
        _, b_next = dbas_step(x_k=x, u_k=u0, b_k=b, f=f, h=h, cfg=db_cfg)

        xs.append(x.detach().cpu().numpy().astype(np.float64))
        us.append(u0.detach().cpu().numpy().astype(np.float64))
        bs.append(b.detach().cpu().numpy().astype(np.float64))

        # collision check (true min over obstacles)
        if obstacles:
            hs = []
            for o in obstacles:
                hs.append(float(h_circle_obstacle(x.detach().cpu(), obs=o)))
            if min(hs) <= 0.0:
                collided = True
                break

        # success check
        if float(torch.linalg.vector_norm(x[:2] - target[:2]).item()) <= success_r:
            success = True
            success_t = t
            break

        # shift warm start
        U_ws = torch.cat([U[1:].detach(), U[-1:].detach().clone()], dim=0)
        x, b = x_next.detach(), b_next.detach()

    xs_np = np.asarray(xs, dtype=np.float64)
    us_np = np.asarray(us, dtype=np.float64)
    bs_np = np.asarray(bs, dtype=np.float64)

    os.makedirs(run_dir, exist_ok=True)
    np.save(os.path.join(run_dir, "x_bar.npy"), xs_np)
    np.save(os.path.join(run_dir, "u_bar.npy"), us_np)
    np.save(os.path.join(run_dir, "x_real.npy"), xs_np)
    np.save(os.path.join(run_dir, "u_real.npy"), us_np)
    np.save(os.path.join(run_dir, "b_real.npy"), bs_np)
    np.save(os.path.join(run_dir, "loss.npy"), np.zeros((xs_np.shape[0],), dtype=np.float64))

    return {
        "summary": {
            "system": "dubins",
            "mode": "nominal_receding",
            "H_ran": int(xs_np.shape[0]),
            "success": bool(success),
            "success_t": None if success_t is None else int(success_t),
            "collided": bool(collided),
            "final_state": xs_np[-1].tolist() if xs_np.size else x0.detach().cpu().numpy().tolist(),
        }
    }


def main() -> None:
    _ensure_project_root_on_path()

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    _set_seed(int(cfg.get("seed", 0)))
    device = torch.device(cfg.get("device", "cpu"))

    out_dir = cfg.get("out_dir", "diff_tube_mpc_strict_pt/outputs")
    run_name = (cfg.get("run_name", "dubins") + "_nominal")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, f"{run_name}_{stamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Fast default: nominal-only receding horizon (tests whether nominal can reach target safely).
    results = run_nominal_receding(cfg, device=device, run_dir=run_dir)

    with open(os.path.join(run_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    with open(os.path.join(run_dir, "results_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results["summary"], f, indent=2, ensure_ascii=False)

    print(f"Saved run to: {run_dir}")
    print(json.dumps(results["summary"], indent=2, ensure_ascii=False))

    if bool(cfg.get("plot", False)) or bool(args.plot):
        from diff_tube_mpc_strict_pt.plot_results import plot_run

        plot_run(run_dir, show=False)
        print("Plots saved.")


if __name__ == "__main__":
    main()

