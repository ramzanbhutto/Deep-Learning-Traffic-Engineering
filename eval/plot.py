from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_congestion_ratio(curves: Dict, title: str, save_path: str):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(curves["epochs"], curves["agent"], marker="o", markersize=4, linewidth=1.8, color="tab:blue", label="Softmin-Routing (ours)")

    colors = {"Prev": "tab:orange", "Avg_k": "tab:green", "Oblivious": "tab:red"}
    styles = {"Prev": "--", "Avg_k": "-.", "Oblivious": ":"}
    for name, (mean, std) in curves["baselines"].items():
        ax.axhline(mean, linestyle=styles[name], color=colors[name], linewidth=1.5, label=f"{name} ({mean:.2f})")

    ax.set_xlabel("Learning epoch")
    ax.set_ylabel("Congestion ratio  ( U / U* )")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def plot_comparison_grid(runs: Dict[str, Dict], save_path: str):
    num_runs = len(runs)
    fig, axes = plt.subplots(1, num_runs, figsize=(6 * num_runs, 5), sharey=False)
    if num_runs == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, runs.items()):
        epochs = []
        agent = []
        for entry in data["history"]:
            if "test_agent_ratio_mean" in entry:
                epochs.append(entry["epoch"])
                agent.append(entry["test_agent_ratio_mean"])
        final = data["history"][-1]

        ax.plot(epochs, agent, marker="o", markersize=3, color="tab:blue", label="Softmin-Routing")

        ax.axhline(final["final_prev_ratio_mean"], ls="--", c="tab:orange", label=f"Prev ({final['final_prev_ratio_mean']:.2f})")
        
        ax.axhline(final["final_avgk_ratio_mean"], ls="-.", c="tab:green", label=f"Avg_k ({final['final_avgk_ratio_mean']:.2f})")
        
        ax.axhline(final["final_oblivious_ratio_mean"], ls=":", c="tab:red", label=f"Oblivious ({final['final_oblivious_ratio_mean']:.2f})")

        cfg = data["config"]
        model = cfg["traffic_model"]
        extra = (f"p={cfg['sparsity']}" if model == "gravity"
                 else f"elephants={int(cfg['elephant_frac']*100)}%")
        ax.set_title(f"{model} DMs ({extra})")
        ax.set_xlabel("Learning epoch")
        ax.set_ylabel("Congestion ratio (U/U*)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")
