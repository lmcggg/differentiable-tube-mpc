from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def _load_npy(run_dir: str, name: str) -> Optional[np.ndarray]:
    path = os.path.join(run_dir, name)
    if not os.path.exists(path):
        return None
    return np.load(path)

def _load_json(run_dir: str, name: str) -> Optional[dict]:
    path = os.path.join(run_dir, name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_run(run_dir: str, *, show: bool = False) -> None:
    x_real = _load_npy(run_dir, "x_real.npy")  # [H,3]
    u_real = _load_npy(run_dir, "u_real.npy")  # [H,2]
    b_real = _load_npy(run_dir, "b_real.npy")  # [H] or [H,1]
    loss = _load_npy(run_dir, "loss.npy")      # [H]
    x_bar = _load_npy(run_dir, "x_bar.npy")    # [H,3]
    u_bar = _load_npy(run_dir, "u_bar.npy")    # [H,2]

    if x_real is None:
        raise FileNotFoundError(f"Missing `x_real.npy` in {run_dir}")

    cfg = _load_json(run_dir, "config_used.json") or {}
    env = cfg.get("environment", {}) if isinstance(cfg, dict) else {}
    obstacles = env.get("obstacles", []) if isinstance(env, dict) else []
    target = (cfg.get("system", {}) or {}).get("target", None) if isinstance(cfg, dict) else None

    H = x_real.shape[0]
    t = np.arange(H)

    # 1) XY trajectory
    fig, ax = plt.subplots(figsize=(7, 6))
    # Draw obstacles if present
    if isinstance(obstacles, list) and len(obstacles) > 0:
        for o in obstacles:
            c = o.get("center", None)
            r = float(o.get("radius", 0.0))
            if c is None or len(c) != 2:
                continue
            circ = Circle((float(c[0]), float(c[1])), r, fill=False, linewidth=2, color="black", alpha=0.6)
            ax.add_patch(circ)

    # Draw target if present
    if isinstance(target, list) and len(target) >= 2:
        ax.scatter([float(target[0])], [float(target[1])], marker="*", s=140, color="gold", edgecolor="black", label="target")

    ax.plot(x_real[:, 0], x_real[:, 1], label="real", linewidth=2)
    if x_bar is not None:
        ax.plot(x_bar[:, 0], x_bar[:, 1], label="nominal ($\\bar{x}$)", linewidth=2, linestyle="--")
    ax.scatter([x_real[0, 0]], [x_real[0, 1]], label="start", s=60)
    ax.scatter([x_real[-1, 0]], [x_real[-1, 1]], label="end", s=60)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Dubins trajectory (x-y)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "traj_xy.png"), dpi=160)
    if show:
        plt.show()
    plt.close(fig)

    # 2) States over time
    plt.figure(figsize=(10, 6))
    plt.plot(t, x_real[:, 0], label="x")
    plt.plot(t, x_real[:, 1], label="y")
    plt.plot(t, x_real[:, 2], label="theta")
    if x_bar is not None:
        plt.plot(t, x_bar[:, 0], label="x_bar", linestyle="--")
        plt.plot(t, x_bar[:, 1], label="y_bar", linestyle="--")
        plt.plot(t, x_bar[:, 2], label="theta_bar", linestyle="--")
    plt.xlabel("t")
    plt.ylabel("state")
    plt.title("States over time")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "states.png"), dpi=160)
    if show:
        plt.show()
    plt.close()

    # 3) Controls over time
    if u_real is not None:
        plt.figure(figsize=(10, 5))
        plt.plot(t, u_real[:, 0], label="v (real)")
        plt.plot(t, u_real[:, 1], label="omega (real)")
        if u_bar is not None:
            plt.plot(t, u_bar[:, 0], label="v_bar", linestyle="--")
            plt.plot(t, u_bar[:, 1], label="omega_bar", linestyle="--")
        plt.xlabel("t")
        plt.ylabel("control")
        plt.title("Controls over time")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "controls.png"), dpi=160)
        if show:
            plt.show()
        plt.close()

    # 4) Barrier state + loss
    if b_real is not None or loss is not None:
        fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        if b_real is not None:
            b_flat = b_real.reshape(H, -1)[:, 0]
            ax[0].plot(t, b_flat, label="b (real)")
            ax[0].set_ylabel("b")
            ax[0].set_title("DBaS barrier state")
            ax[0].grid(True, alpha=0.3)
            ax[0].legend()
        if loss is not None:
            ax[1].plot(t, loss, label="L", color="tab:red")
            ax[1].set_xlabel("t")
            ax[1].set_ylabel("loss")
            ax[1].set_title("Online loss")
            ax[1].grid(True, alpha=0.3)
            ax[1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "barrier_and_loss.png"), dpi=160)
        if show:
            plt.show()
        plt.close()

    # 5) Adaptive parameters evolution
    Qa_history = _load_npy(run_dir, "Qa_history.npy")
    Ra_history = _load_npy(run_dir, "Ra_history.npy")
    qba_history = _load_npy(run_dir, "qba_history.npy")
    
    if Qa_history is not None or Ra_history is not None or qba_history is not None:
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        
        if Qa_history is not None:
            H_params = Qa_history.shape[0]
            t_params = np.arange(H_params)
            axes[0].plot(t_params, Qa_history[:, 0], label="$Q_a[0]$ (x)", linewidth=2)
            axes[0].plot(t_params, Qa_history[:, 1], label="$Q_a[1]$ (y)", linewidth=2)
            axes[0].plot(t_params, Qa_history[:, 2], label="$Q_a[2]$ (theta)", linewidth=2)
            axes[0].set_ylabel("$Q_a$")
            axes[0].set_title("Auxiliary MPC State Weight Evolution")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(loc='best')
            axes[0].set_yscale('log')  # Use log scale for better visualization
        
        if Ra_history is not None:
            H_params = Ra_history.shape[0]
            t_params = np.arange(H_params)
            axes[1].plot(t_params, Ra_history[:, 0], label="$R_a[0]$ (v)", linewidth=2)
            axes[1].plot(t_params, Ra_history[:, 1], label="$R_a[1]$ (omega)", linewidth=2)
            axes[1].set_ylabel("$R_a$")
            axes[1].set_title("Auxiliary MPC Control Weight Evolution")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(loc='best')
            axes[1].set_yscale('log')  # Use log scale for better visualization
        
        if qba_history is not None:
            H_params = len(qba_history)
            t_params = np.arange(H_params)
            axes[2].plot(t_params, qba_history, label="$q_{b,a}$", linewidth=2, color="tab:green")
            axes[2].set_xlabel("t")
            axes[2].set_ylabel("$q_{b,a}$")
            axes[2].set_title("Auxiliary MPC Barrier Weight Evolution")
            axes[2].grid(True, alpha=0.3)
            axes[2].legend(loc='best')
        
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "adaptive_parameters.png"), dpi=160)
        if show:
            plt.show()
        plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="Output directory containing *.npy files")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    plot_run(args.run_dir, show=args.show)
    print(f"Saved plots to: {args.run_dir}")


if __name__ == "__main__":
    main()

