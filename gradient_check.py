from __future__ import annotations

import argparse
import copy
import math
import os
import sys
from typing import Any, Dict, Tuple

import torch
import yaml


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def _clone_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(cfg)


def main() -> None:
    # Ensure repo root on sys.path when executed as a script.
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="diff_tube_mpc_strict_pt/configs/dubins_smoke.yaml")
    ap.add_argument("--eps", type=float, default=1e-3)
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    device = torch.device(cfg.get("device", "cpu"))

    # Keep it small for finite differences
    cfg = _clone_cfg(cfg)
    cfg["system"]["horizon_N"] = int(min(8, cfg["system"]["horizon_N"]))
    cfg["system"]["task_horizon_H"] = int(min(2, cfg["system"]["task_horizon_H"]))
    cfg["system"]["nominal_max_iter"] = int(min(3, cfg["system"].get("nominal_max_iter", 3)))
    cfg["system"]["aux_max_iter"] = int(min(3, cfg["system"].get("aux_max_iter", 3)))

    from diff_tube_mpc_strict_pt.core.tube_mpc import run_closed_loop_experiment

    # Baseline run
    base = run_closed_loop_experiment(cfg, device=device, run_dir="diff_tube_mpc_strict_pt/outputs/_gradcheck_tmp")
    base_loss = float(base["summary"]["final_loss"])

    # Finite difference on one scalar parameter: nominal Q_raw[0] (implemented via softplus inside NominalTheta)
    eps = float(args.eps)
    cfg_p = _clone_cfg(cfg)
    cfg_m = _clone_cfg(cfg)

    # We perturb the *configured* nominal Q[0] (acts like perturbing theta_bar.Q_raw pre-softplus approximation).
    cfg_p["cost_nominal"]["Q"][0] = float(cfg["cost_nominal"]["Q"][0]) + eps
    cfg_m["cost_nominal"]["Q"][0] = float(cfg["cost_nominal"]["Q"][0]) - eps

    out_p = run_closed_loop_experiment(cfg_p, device=device, run_dir="diff_tube_mpc_strict_pt/outputs/_gradcheck_tmp_p")
    out_m = run_closed_loop_experiment(cfg_m, device=device, run_dir="diff_tube_mpc_strict_pt/outputs/_gradcheck_tmp_m")

    loss_p = float(out_p["summary"]["final_loss"])
    loss_m = float(out_m["summary"]["final_loss"])
    fd = (loss_p - loss_m) / (2.0 * eps)

    print("Finite-difference check (coarse):")
    print(f"- baseline loss: {base_loss:.6f}")
    print(f"- loss(+eps):    {loss_p:.6f}")
    print(f"- loss(-eps):    {loss_m:.6f}")
    print(f"- fd approx dL/dQ0: {fd:.6f}")
    print()
    print("Note: This is a coarse check because the config perturbs the *interpreted* weight,")
    print("not the internal raw parameter after softplus. It is meant to catch gross sign/magnitude bugs.")


if __name__ == "__main__":
    main()

