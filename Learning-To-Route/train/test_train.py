import json
from pathlib import Path

import numpy as np

from graph.network import create_12_node_topology
from train.trainer import TrainConfig, Trainer


def make_tiny_config(outdir, reward_normalizer="bound", epochs=2):
    return TrainConfig(
        num_train_sequences=2,
        num_test_sequences=1,
        sequence_length=8,
        history_k=3,
        epochs=epochs,
        eval_every=1,
        hidden_dim=32,
        batch_size=16,
        update_epochs=2,
        reward_normalizer=reward_normalizer,
        outdir=outdir,
    )


def test_trainer_smoke_bound():
    outdir = "/tmp/ramzan/lr_smoke_bound"
    trainer = Trainer(create_12_node_topology(), make_tiny_config(outdir))

    assert len(trainer.train_sequences) == 2
    assert len(trainer.test_sequences) == 1
    assert len(trainer.train_sequences[0]) == 8
    state = trainer._state_of(trainer.train_sequences[0], 5)
    assert state.shape == (3 * 12 * 12,)
    print("Trainer construction + state shape OK")

    stats = trainer.run()

    assert len(stats) == 2
    for entry in stats:
        for key in ("train_reward_mean", "train_ratio_mean"):
            assert np.isfinite(entry[key])
    print(f"Training epochs OK (rewards: "
          f"{[round(e['train_reward_mean'], 3) for e in stats]})")


def test_trainer_smoke_lp():
    outdir = "/tmp/ramzan/lr_smoke_lp"
    trainer = Trainer(create_12_node_topology(), make_tiny_config(outdir, reward_normalizer="lp"))
    stats = trainer.run()

    final = stats[-1]
    for key in ("test_agent_ratio_mean", "test_prev_ratio_mean", "test_avgk_ratio_mean", "test_oblivious_ratio_mean"):
        assert key in final and np.isfinite(final[key])

    assert final["test_prev_ratio_mean"] >= 0.99 - 1e-6
    print(f"LP-normalized run OK (agent={final['test_agent_ratio_mean']:.3f}, "
          f"prev={final['test_prev_ratio_mean']:.3f}, "
          f"avgk={final['test_avgk_ratio_mean']:.3f}, "
          f"oblivious={final['test_oblivious_ratio_mean']:.3f})")

    metrics_path = Path(outdir) / "metrics.json"
    checkpoint_path = Path(outdir) / "checkpoint.pt"
    assert metrics_path.exists() and checkpoint_path.exists()
    data = json.loads(metrics_path.read_text())
    assert len(data["history"]) == 2
    print("Artifacts saved OK (metrics.json, checkpoint.pt)")


def test_baseline_caching():
    outdir = "/tmp/ramzan/lr_smoke_cache"
    trainer = Trainer(create_12_node_topology(), make_tiny_config(outdir))
    seq = trainer.test_sequences[0]
    r1 = trainer.baseline_for_window("oblivious", 0, seq, 5)
    r2 = trainer.baseline_for_window("oblivious", 0, seq, 5)
    assert r1 == r2
    assert ("oblivious", 0, 5) in trainer._baseline_cache
    print(f"Baseline caching OK (oblivious ratio {r1:.3f})")


if __name__ == "__main__":
    import time
    t0 = time.perf_counter()
    test_trainer_smoke_bound()
    test_baseline_caching()
    test_trainer_smoke_lp()
    print(f"\n=== ALL TRAINER TESTS PASSED ({time.perf_counter() - t0:.1f}s) ===")
