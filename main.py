import argparse
import json
import time
from pathlib import Path

from eval.evaluate import summarize_run
from eval.plot import plot_comparison_grid, plot_congestion_ratio
from graph.network import create_12_node_topology
from train.trainer import TrainConfig, Trainer

CONFIGS = {
    "gravity": {"traffic_model": "gravity", "sparsity": 0.3, "elephant_frac": 0.4},
    "bimodal": {"traffic_model": "bimodal", "sparsity": 1.0, "elephant_frac": 0.4},
    "gravity_cyclic": {"traffic_model": "gravity", "sparsity": 0.3,
                       "elephant_frac": 0.4, "sequence_mode": "cyclic",
                       "cyclic_q": 6},
    "bimodal_cyclic": {"traffic_model": "bimodal", "sparsity": 1.0,
                       "elephant_frac": 0.4, "sequence_mode": "cyclic",
                       "cyclic_q": 6},
}


def run_config(name: str, overrides: dict, outroot: str) -> dict:
    params = {
        "epochs": 60,
        "eval_every": 5,
        "sequence_length": 40,
        "num_train_sequences": 7,
        "num_test_sequences": 3,
        "history_k": 10,
        "reward_normalizer": "lp",
        "seed": 42,
    }
    params.update(CONFIGS[name])
    params.update(overrides)
    outdir = str(Path(outroot) / name)

    config = TrainConfig(outdir=outdir, **params)
    trainer = Trainer(create_12_node_topology(), config)
    print(f"\n=== running config '{name}' -> {outdir} ===")
    trainer.run()

    with open(Path(outdir) / "metrics.json") as f:
        data = json.load(f)
    curves = {
        "epochs": [e["epoch"] for e in data["history"]
                   if "test_agent_ratio_mean" in e],
        "agent": [e["test_agent_ratio_mean"] for e in data["history"]
                  if "test_agent_ratio_mean" in e],
        "baselines": {
            "Prev": (data["history"][-1]["final_prev_ratio_mean"],
                     data["history"][-1].get("final_prev_ratio_std", 0.0)),
            "Avg_k": (data["history"][-1]["final_avgk_ratio_mean"],
                      data["history"][-1].get("final_avgk_ratio_std", 0.0)),
            "Oblivious": (data["history"][-1]["final_oblivious_ratio_mean"],
                          data["history"][-1].get("final_oblivious_ratio_std", 0.0)),
        },
    }
    plot_congestion_ratio(
        curves,
        f"Learning To Route: {name} DMs "
        f"(sparsity p={config.sparsity}, k={config.history_k})",
        str(Path(outdir) / "congestion_ratio.png"))
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=["gravity"],
                        choices=list(CONFIGS.keys()))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=40)
    parser.add_argument("--reward-normalizer", default="lp",
                        choices=["lp", "bound"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outroot", default="results")
    args = parser.parse_args()

    results_root = Path(args.outroot) / f"seed{args.seed}"
    results_root.mkdir(parents=True, exist_ok=True)

    all_runs = {}
    summaries = {}
    for name in args.configs:
        t0 = time.perf_counter()
        data = run_config(name, {
            "epochs": args.epochs,
            "eval_every": args.eval_every,
            "sequence_length": args.seq_len,
            "reward_normalizer": args.reward_normalizer,
            "seed": args.seed,
        }, str(results_root))
        summaries[name] = summarize_run(data)
        summaries[name]["wall_time_s"] = round(time.perf_counter() - t0, 1)
        all_runs[name] = data

    plot_comparison_grid(all_runs, str(results_root / "congestion_ratio_combined.png"))

    with open(results_root / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2)

    print("\n===== RESULTS SUMMARY =====")
    for name, s in summaries.items():
        print(f"[{name}] agent U/U*: {s['initial_agent_ratio']:.3f} -> "
              f"{s['final_agent_ratio']:.3f} | Prev={s['prev_ratio']:.3f} "
              f"Avg_k={s['avgk_ratio']:.3f} Oblivious={s['oblivious_ratio']:.3f} "
              f"| min delivered={s['agent_delivered_min']:.4f} "
              f"({s['wall_time_s']}s)")


if __name__ == "__main__":
    main()
