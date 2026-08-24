import json
from pathlib import Path

import numpy as np

from graph.network import create_12_node_topology
from sdn_ddpg.train.trainer import DDPGConfig, SdnDDPGTrainer


def make_tiny_config(outdir: str, epochs=2) -> DDPGConfig:
    return DDPGConfig(
        num_train_sequences=2,
        num_test_sequences=1,
        sequence_length=6,
        hidden_dims=(64, 64),
        batch_size=16,
        warmup_steps=10,
        buffer_capacity_rl=500,
        epochs=epochs,
        eval_every=1,
        outdir=outdir,
    )


def test_calibration_brings_hotspot_into_regime():
    trainer = SdnDDPGTrainer(create_12_node_topology(), make_tiny_config("/tmp/learnedrouting/sdn_smoke"))
    probe_dm = trainer.train_sequences[0][0]
    flows = trainer._dm_to_flows(probe_dm, scale=trainer.rate_scale)
    mid = (trainer.config.weight_low + trainer.config.weight_high) / 2.0
    weights = np.full(len(trainer.graph.edges), mid)
    from sdn_ddpg.forwarding.routing_sdn import aggregate_arrival_rates, route_flows
    routes = route_flows(trainer.graph, weights, flows)
    rates, _ = aggregate_arrival_rates(trainer.graph, routes, trainer.config.service_rate, trainer.config.buffer_capacity)
    rho_peak = rates.max() / trainer.config.service_rate

    assert 0.3 <= rho_peak <= 1.4, \
        f"calibrated hotspot rho {rho_peak:.2f} outside target regime"
    print(f"Rate calibration OK (peak rho={rho_peak:.3f} after "
          f"scale={trainer.rate_scale:.4f})")


def test_trainer_mini_run():
    outdir = "/tmp/learnedrouting/sdn_mini"
    trainer = SdnDDPGTrainer(create_12_node_topology(), make_tiny_config(outdir))
    history = trainer.run()

    assert len(history) == 2
    final = history[-1]
    for key in ("test_agent_reward_mean", "test_agent_delay_mean",
                "test_agent_loss_fraction_mean", "final_agent_ratio_mean"):
        assert key in final and np.isfinite(final[key]), f"{key} missing/NaN"

    metrics_path = Path(outdir) / "metrics.json"
    checkpoint_path = Path(outdir) / "checkpoint.pt"
    assert metrics_path.exists() and checkpoint_path.exists()

    data = json.loads(metrics_path.read_text())
    assert data["method"] == "sdn-ddpg"
    assert len(data["history"]) == 2
    print(f"Mini-run OK (reward {history[0]['train_reward_mean']:.4f} -> "
          f"{history[-1]['train_reward_mean']:.4f}, "
          f"U/U*={final['final_agent_ratio_mean']:.3f})")


def test_deterministic_evaluation_is_stable():
    outdir = "/tmp/learnedrouting/sdn_eval"
    trainer = SdnDDPGTrainer(create_12_node_topology(), make_tiny_config(outdir))

    first = trainer.evaluate()
    second = trainer.evaluate()
    for key in first:
        if "ratio" in key and np.isnan(first[key]):
            continue
        assert abs(first[key] - second[key]) < 1e-9, \
            f"deterministic eval must be repeatable ({key})"
    print("Deterministic evaluation repeatability OK")


if __name__ == "__main__":
    test_calibration_brings_hotspot_into_regime()
    test_trainer_mini_run()
    test_deterministic_evaluation_is_stable()
    print("\n=== ALL SDN-DDPG TRAINER TESTS PASSED ===")
