from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict

import numpy as np
import torch
import yaml


def _ensure_project_root_on_path() -> None:
    # When executing as `python diff_tube_mpc_strict_pt/run_experiment.py`,
    # sys.path[0] is the package directory, not the repo root.
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


def main() -> None:
    _ensure_project_root_on_path()

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--plot", action="store_true", help="Generate plots into output directory")
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    if bool(cfg.get("debug_detect_anomaly", False)):
        torch.autograd.set_detect_anomaly(True)
    seed = int(cfg.get("seed", 0))
    _set_seed(seed)

    device = torch.device(cfg.get("device", "cpu"))

    # Lazy imports to keep entrypoint lightweight
    from diff_tube_mpc_strict_pt.core.tube_mpc import run_closed_loop_experiment

    out_dir = cfg.get("out_dir", "diff_tube_mpc_strict_pt/outputs")
    run_name = cfg.get("run_name", os.path.splitext(os.path.basename(args.config))[0])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, f"{run_name}_{stamp}")
    os.makedirs(run_dir, exist_ok=True)

    results = run_closed_loop_experiment(cfg, device=device, run_dir=run_dir)

    # Save config + results metadata
    with open(os.path.join(run_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    with open(os.path.join(run_dir, "results_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results["summary"], f, indent=2, ensure_ascii=False)

    print(f"Saved run to: {run_dir}")
    print(json.dumps(results["summary"], indent=2, ensure_ascii=False))

    do_plot = bool(cfg.get("plot", False)) or bool(args.plot)
    if do_plot:
        from diff_tube_mpc_strict_pt.plot_results import plot_run
        plot_run(run_dir, show=False)
        print("Plots saved.")


if __name__ == "__main__":
    main()

