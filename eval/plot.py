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
        mode = "cyclic q={}".format(cfg.get("cyclic_q", 6)) \
            if cfg.get("sequence_mode") == "cyclic" else "iid"
        extra = (f"p={cfg['sparsity']}" if model == "gravity"
                 else f"elephants={int(cfg['elephant_frac']*100)}%")
        ax.set_title(f"{model} DMs ({extra}, {mode})")
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


def plot_ddpg_training(data: Dict, title: str, save_path: str):
    epochs = [e["epoch"] for e in data["history"]]
    reward = [e["train_reward_mean"] for e in data["history"]]
    ratio_eps, ratio = [], []
    delay_eps, delay = [], []
    loss_eps, loss = [], []
    for e in data["history"]:
        if "test_agent_ratio_mean" in e and np.isfinite(e["test_agent_ratio_mean"]):
            ratio_eps.append(e["epoch"])
            ratio.append(e["test_agent_ratio_mean"])
        if "test_agent_delay_mean" in e and np.isfinite(e["test_agent_delay_mean"]):
            delay_eps.append(e["epoch"])
            delay.append(e["test_agent_delay_mean"])
        if "test_agent_loss_fraction_mean" in e and np.isfinite(e["test_agent_loss_fraction_mean"]):
            loss_eps.append(e["epoch"])
            loss.append(e["test_agent_loss_fraction_mean"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(epochs, reward, color="tab:purple", linewidth=1.5)
    axes[0].set_title("Training reward R(s,a)", fontsize=10)
    axes[1].plot(ratio_eps, ratio, marker="o", markersize=3, color="tab:blue")
    axes[1].set_title("Test congestion ratio (U/U*)", fontsize=10)
    axes[2].plot(loss_eps, loss, color="tab:red", linewidth=1.5, label="loss frac")
    ax2 = axes[2].twinx()
    ax2.plot(delay_eps, delay, color="tab:green", linewidth=1.2)
    ax2.set_ylabel("mean delay (s)", fontsize=8)
    axes[2].set_title("Packet loss + delay", fontsize=10)
    axes[2].legend(loc="upper left", fontsize=7)

    for ax in axes:
        ax.set_xlabel("Learning epoch")
        ax.grid(True, alpha=0.3)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def plot_epoch_sweep(sweep_rows: list, save_path: str):
    eps = [r["epochs"] for r in sweep_rows]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(eps, [r["PPO"] for r in sweep_rows], marker="o",
            color="tab:blue", label="PPO-softmin")
    ax.plot(eps, [r["DDPG-TD3"] for r in sweep_rows], marker="s",
            color="tab:purple", label="DDPG-SDN (TD3)")
    ax.axhline(1.575, ls="--", c="tab:red", label="Oblivious (180-ep ref)")
    ax.set_xscale("log")
    ax.set_xticks(eps)
    ax.set_xticklabels([str(e) for e in eps])
    ax.get_xaxis().set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("Training epochs (log scale)")
    ax.set_ylabel("Congestion ratio (U/U*)")
    ax.set_title("Epoch sweep - gravity regime", fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")
