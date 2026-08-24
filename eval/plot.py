from pathlib import Path
from typing import Dict

import numpy as np
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


def plot_five_method_comparison(results: Dict, title: str, save_path: str):
    methods = ["PPO-softmin", "DDPG-SDN", "Prev", "Avg_k", "Oblivious"]
    palette = ["tab:blue", "tab:purple", "tab:orange", "tab:green", "tab:red"]
    metrics = [("ratio_mean", "ratio_std", "Congestion ratio (U/U*)"),
               ("delay_mean", "delay_std", "Mean delay (s)"),
               ("loss_mean", "loss_std", "Loss fraction")]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    x = np.arange(len(methods))
    for ax, (mean_key, std_key, label) in zip(axes, metrics):
        means = [results[m][mean_key] for m in methods]
        stds = [results[m].get(std_key, 0.0) for m in methods]
        ax.bar(x, means, yerr=stds, color=palette, capsize=4, alpha=0.85)
        for xi, value in zip(x, means):
            ax.text(xi, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def plot_method_grid(all_results: Dict[str, Dict], save_path: str):
    import numpy as np
    num = len(all_results)
    fig, axes = plt.subplots(1, num, figsize=(6 * max(num, 1), 5), sharey=False)
    if num == 1:
        axes = [axes]

    methods = ["PPO-softmin", "DDPG-SDN", "Prev", "Avg_k", "Oblivious"]
    palette = ["tab:blue", "tab:purple", "tab:orange", "tab:green", "tab:red"]
    for ax, (name, results) in zip(axes, all_results.items()):
        means = [results[m]["ratio_mean"] for m in methods]
        bars = ax.bar(np.arange(len(methods)), means, color=palette, alpha=0.85)
        for xi, value in zip(np.arange(len(methods)), means):
            ax.text(xi, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels(methods, rotation=25, fontsize=8)
        cfg_title = name.replace("_", " ")
        ax.set_title(f"{cfg_title}", fontsize=10)
        ax.set_ylabel("Congestion ratio (U/U*)")
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")
