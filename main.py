import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for package_root in ("Learning-To-Route", "SDN-DDPG"):
    path_entry = str(ROOT / package_root)
    if path_entry not in sys.path:
        sys.path.insert(0, path_entry)

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
    params.pop("td3", None)
    outdir = str(Path(outroot) / f"{name}_ltr")

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


def run_sdn_config(name: str, overrides: dict, outroot: str) -> dict:
    from sdn_ddpg.train.trainer import DDPGConfig, SdnDDPGTrainer

    params = {
        "epochs": 60,
        "eval_every": 5,
        "sequence_length": 40,
        "num_train_sequences": 7,
        "num_test_sequences": 3,
        "seed": 42,
    }
    params.update(CONFIGS[name])
    params.update(overrides)
    params["use_twin_critics"] = params.pop("td3", False)
    if params["use_twin_critics"]:
        params.setdefault("reward_scale", 0.01)
        params.setdefault("gamma", 0.9)
    suffix = f"{name}_sdn_td3" if params.get("use_twin_critics") \
        else f"{name}_sdn"
    outdir = str(Path(outroot) / suffix)

    allowed = DDPGConfig.__dataclass_fields__.keys()
    config = DDPGConfig(outdir=outdir,
                        **{k: v for k, v in params.items() if k in allowed})
    trainer = SdnDDPGTrainer(create_12_node_topology(), config)
    print(f"\n=== running SDN-DDPG config '{name}' -> {outdir} ===")
    trainer.run()

    from eval.plot import plot_ddpg_training
    with open(Path(outdir) / "metrics.json") as f:
        data = json.load(f)
    model = params.get("traffic_model", "gravity")
    extra = (f"p={params.get('sparsity')}" if model == "gravity"
             else f"elephants={int(params.get('elephant_frac', 0.4) * 100)}%")
    variant = "TD3" if params.get("use_twin_critics") else "paper-DDPG"
    plot_ddpg_training(
        data,
        f"DDPG {variant}: {name} ({extra}, 180 epochs)",
        str(Path(outdir) / "training_curves.png"))
    return data


def run_compare(configs, results_root: str, stride: int, force: bool,
                td3: bool):
    from sdn_ddpg.comparison.compare import compare_config
    from eval.plot import plot_five_method_comparison, plot_method_grid

    all_results = {}
    for name in configs:
        all_results[name] = compare_config(name, str(results_root),
                                           stride=stride, force=force,
                                           ddpg_variant="td3" if td3
                                           else "paper")
        plot_five_method_comparison(
            all_results[name],
            f"Five-method comparison: {name}",
            str(results_root / f"five_method_{name}.png"))

    plot_method_grid(all_results,
                     str(results_root / "five_method_combined.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=["gravity"],
                        choices=list(CONFIGS.keys()))
    parser.add_argument("--method", default="ppo", choices=["ppo", "sdn-ddpg"])
    parser.add_argument("--td3", action="store_true",
                        help="train SDN-DDPG with twin critics + target "
                             "action smoothing (extension, not in paper)")
    parser.add_argument("--compare", action="store_true",
                        help="evaluate PPO+DDPG checkpoints plus classical "
                             "baselines on shared test streams")
    parser.add_argument("--stride", type=int, default=2,
                        help="window stride for --compare evaluation")
    parser.add_argument("--force", action="store_true",
                        help="overwrite comparison.json entries even when "
                             "the stored stride differs")
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

    if args.compare:
        run_compare(args.configs, results_root, args.stride, args.force,
                    args.td3)
        return

    results_root = Path(args.outroot) / f"seed{args.seed}"
    results_root.mkdir(parents=True, exist_ok=True)

    all_runs = {}
    summaries = {}
    for name in args.configs:
        t0 = time.perf_counter()
        runner = run_config if args.method == "ppo" else run_sdn_config
        data = runner(name, {
            "td3": args.td3,
            "epochs": args.epochs,
            "eval_every": args.eval_every,
            "sequence_length": args.seq_len,
            "reward_normalizer": args.reward_normalizer,
            "seed": args.seed,
        }, str(results_root))
        if args.method == "ppo":
            summaries[name] = summarize_run(data)
            summaries[name]["wall_time_s"] = round(time.perf_counter() - t0, 1)
        else:
            final = data["history"][-1]
            summaries[name] = {
                "final_reward": final.get("final_agent_reward_mean",
                                          final.get("test_agent_reward_mean")),
                "final_delay": final.get("final_agent_delay_mean",
                                         final.get("test_agent_delay_mean")),
                "final_loss": final.get("final_agent_loss_fraction_mean",
                                        final.get("test_agent_loss_fraction_mean")),
                "final_U_over_Ustar": final.get("final_agent_ratio_mean",
                                                final.get("test_agent_ratio_mean")),
                "wall_time_s": round(time.perf_counter() - t0, 1),
            }
        all_runs[name] = data

    if args.method == "ppo":
        plot_comparison_grid(all_runs, str(results_root / "congestion_ratio_combined.png"))

    with open(results_root / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2)

    print("\n===== RESULTS SUMMARY =====")
    if args.method == "ppo":
        for name, s in summaries.items():
            print(f"[{name}] agent U/U*: {s['initial_agent_ratio']:.3f} -> "
                  f"{s['final_agent_ratio']:.3f} | Prev={s['prev_ratio']:.3f} "
                  f"Avg_k={s['avgk_ratio']:.3f} Oblivious={s['oblivious_ratio']:.3f} "
                  f"| min delivered={s['agent_delivered_min']:.4f} "
                  f"({s['wall_time_s']}s)")
    else:
        for name, s in summaries.items():
            delay = s['final_delay']
            loss = s['final_loss']
            ratio = s['final_U_over_Ustar']
            print(f"[{name}] DDPG reward={s['final_reward']:.4f} "
                  f"delay={delay:.4f}s loss={loss:.4f} "
                  f"U/U*={ratio:.3f}" if delay is not None else
                  f"[{name}] DDPG metrics incomplete")


if __name__ == "__main__":
    main()
