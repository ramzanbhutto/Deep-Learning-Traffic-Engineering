import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from agent.ppo import PPOAgent
from baselines.classical import AvgKBaseline, ObliviousRouting, PrevBaseline
from graph.network import NetworkGraph
from routing.optimal import OptimalCongestionCache, congestion_lower_bound
from softmin_routing.softmin import (compute_multicommodity_flow, max_link_utilization, softmin_splitting_ratios)
from traffic.generator import DemandMatrixGenerator


@dataclass
class TrainConfig:
    num_nodes: int = 12
    num_edges: int = 34
    traffic_model: str = "gravity"
    sparsity: float = 0.3
    elephant_frac: float = 0.4
    target_lb: float = 0.5
    sequence_mode: str = "iid"
    cyclic_q: int = 6
    num_train_sequences: int = 7
    num_test_sequences: int = 3
    sequence_length: int = 60
    history_k: int = 10
    softmin_gamma: float = 2.0
    reward_normalizer: str = "lp"
    hidden_dim: int = 128
    lr: float = 5e-4
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    update_epochs: int = 4
    batch_size: int = 64
    rl_gamma: float = 0.95
    gae_lambda: float = 0.95
    epochs: int = 60
    eval_every: int = 5
    seed: int = 42
    outdir: str = "results/run"


class Trainer:
    def __init__(self, graph: NetworkGraph, config: TrainConfig):
        self.graph = graph
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        self.train_generator = DemandMatrixGenerator(
            graph, model=config.traffic_model, sparsity=config.sparsity,
            elephant_frac=config.elephant_frac, seed=config.seed)
        self.test_generator = DemandMatrixGenerator(
            graph, model=config.traffic_model, sparsity=config.sparsity,
            elephant_frac=config.elephant_frac, seed=config.seed + 1000,
            out_bw=self.train_generator.out_bw,
            in_bw=self.train_generator.in_bw)

        self.train_sequences = self._make_sequences(self.train_generator, config.num_train_sequences)
        self.test_sequences = self._make_sequences(self.test_generator, config.num_test_sequences)

        all_train_dms = [dm for seq in self.train_sequences for dm in seq]
        positive = np.concatenate([dm[dm > 0].ravel() for dm in all_train_dms])
        self.demand_scale = float(np.median(positive)) if positive.size else 1.0

        self.opt_cache = OptimalCongestionCache(graph)
        self.bound_cache: Dict[bytes, float] = {}

        self.agent = PPOAgent(
            state_dim=config.history_k * graph.num_nodes * graph.num_nodes,
            action_dim=len(graph.edges),
            hidden_dim=config.hidden_dim,
            lr=config.lr,
            gamma=config.rl_gamma,
            gae_lambda=config.gae_lambda,
            clip_eps=config.clip_eps,
            entropy_coef=config.entropy_coef,
            update_epochs=config.update_epochs,
            batch_size=config.batch_size,
            seed=config.seed)

        from agent.ppo import RolloutBuffer
        self.rollout = RolloutBuffer()

        self.prev_baseline = PrevBaseline(graph, gamma=config.softmin_gamma)
        self.avgk_baseline = AvgKBaseline(graph, k=config.history_k, gamma=config.softmin_gamma)
        self.oblivious: Optional[ObliviousRouting] = None
        self._baseline_cache: Dict[Tuple[str, int, int], float] = {}

        self.outdir = Path(config.outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.metrics_history: List[Dict] = []

    def _make_sequences(self, generator: DemandMatrixGenerator, count: int) -> List[List[np.ndarray]]:
        sequences = []
        for _ in range(count):
            if self.config.sequence_mode == "cyclic":
                base = [generator.sample()
                        for _ in range(self.config.cyclic_q)]
                seq = [base[t % self.config.cyclic_q]
                       for t in range(self.config.sequence_length)]
            else:
                seq = [generator.sample()
                       for _ in range(self.config.sequence_length)]

            scaled = []
            for dm in seq:
                lb = congestion_lower_bound(self.graph, dm)
                if lb > 0:
                    dm = (dm * (self.config.target_lb / lb)).astype(np.float32)
                np.fill_diagonal(dm, 0.0)
                scaled.append(dm)
            sequences.append(scaled)
        return sequences

    def _state_of(self, sequence: List[np.ndarray], end_idx: int) -> np.ndarray:
        k = self.config.history_k
        window = sequence[end_idx - k:end_idx]
        stacked = np.log1p(np.stack(window) / self.demand_scale)
        return stacked.ravel().astype(np.float32)

    def _normalizer(self, dm: np.ndarray) -> float:
        if self.config.reward_normalizer == "lp":
            return max(self.opt_cache.get(dm), 1e-9)
        key = dm.tobytes()
        if key not in self.bound_cache:
            self.bound_cache[key] = max(
                congestion_lower_bound(self.graph, dm), 1e-9)
        return self.bound_cache[key]

    def reward_for_weights(self, weights: np.ndarray, dm: np.ndarray) -> Tuple[float, float, float]:
        ratios = softmin_splitting_ratios(self.graph, weights, self.config.softmin_gamma)
        edge_flow, delivered = compute_multicommodity_flow( self.graph, ratios, dm)
        utilization = max_link_utilization(self.graph, edge_flow)
        optimal = self._normalizer(dm)
        reward = -(utilization / optimal)
        return reward, utilization / optimal, delivered

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        from agent.ppo import RolloutBuffer
        rollout = RolloutBuffer()

        rewards_log, ratios_log, delivered_log = [], [], []

        for seq_id, sequence in enumerate(self.train_sequences):
            for end_idx in range(self.config.history_k, len(sequence)):
                state = self._state_of(sequence, end_idx)
                action, log_prob, value = self.agent.act(state)
                weights = np.exp(np.clip(action, -6.0, 6.0)).astype(np.float32)

                reward, ratio, delivered = self.reward_for_weights(weights, sequence[end_idx])

                done = end_idx == len(sequence) - 1
                rollout.add(state, action, log_prob, reward, value, done)

                rewards_log.append(reward)
                ratios_log.append(ratio)
                delivered_log.append(delivered)

        last_state_value = rollout.values[-1]
        stats = self.agent.update(rollout)

        return {
            "epoch": epoch,
            "train_reward_mean": float(np.mean(rewards_log)),
            "train_ratio_mean": float(np.mean(ratios_log)),
            "delivered_min": float(np.min(delivered_log)),
            **stats,
        }

    def ensure_oblivious(self):
        if self.oblivious is None:
            rng = np.random.default_rng(self.config.seed + 7)
            scenario_pool = []
            for i in range(6):
                gen = DemandMatrixGenerator(
                    self.graph, model=self.config.traffic_model,
                    sparsity=self.config.sparsity,
                    elephant_frac=self.config.elephant_frac,
                    seed=int(rng.integers(1e6)),
                    out_bw=self.train_generator.out_bw,
                    in_bw=self.train_generator.in_bw)
                dm = gen.sample()
                lb = congestion_lower_bound(self.graph, dm)
                if lb > 0:
                    dm = (dm * (self.config.target_lb / lb)).astype(np.float32)
                np.fill_diagonal(dm, 0.0)
                scenario_pool.append(dm)
            self.oblivious = ObliviousRouting(self.graph, scenario_pool)

    def evaluate_agent_window(self, sequence: List[np.ndarray], end_idx: int) -> Tuple[float, float]:
        state = self._state_of(sequence, end_idx)
        action, _, _ = self.agent.act(state, deterministic=True)
        weights = np.exp(np.clip(action, -6.0, 6.0)).astype(np.float32)
        _, ratio, delivered = self.reward_for_weights(weights, sequence[end_idx])
        return ratio, delivered

    def baseline_for_window(self, method: str, sequence_id: int, sequence: List[np.ndarray], end_idx: int) -> float:
        key = (method, sequence_id, end_idx)
        if key in self._baseline_cache:
            return self._baseline_cache[key]

        dm_target = sequence[end_idx]
        optimal = max(self.opt_cache.get(dm_target), 1e-9)

        if method == "prev":
            util, _ = self.prev_baseline.routing_congestion(sequence[end_idx - 1], dm_target)
        elif method == "avgk":
            util, _ = self.avgk_baseline.routing_congestion(sequence[max(0, end_idx - self.config.history_k):end_idx], dm_target)
        elif method == "oblivious":
            self.ensure_oblivious()
            util = self.oblivious.evaluate(dm_target)
        else:
            raise ValueError(method)

        ratio = util / optimal
        self._baseline_cache[key] = ratio
        return ratio

    def evaluate(self) -> Dict[str, float]:
        agent_ratios, agent_delivered = [], []
        baseline_ratios = {"prev": [], "avgk": [], "oblivious": []}

        for seq_id, sequence in enumerate(self.test_sequences):
            for end_idx in range(self.config.history_k, len(sequence)):
                ratio, delivered = self.evaluate_agent_window(sequence, end_idx)
                agent_ratios.append(ratio)
                agent_delivered.append(delivered)
                for method in baseline_ratios:
                    baseline_ratios[method].append(
                        self.baseline_for_window(method, seq_id,
                                                 sequence, end_idx))

        return {
            "agent_ratio_mean": float(np.mean(agent_ratios)),
            "agent_ratio_std": float(np.std(agent_ratios)),
            "agent_delivered_min": float(np.min(agent_delivered)),
            **{f"{method}_ratio_mean": float(np.mean(vals))
               for method, vals in baseline_ratios.items()},
            **{f"{method}_ratio_std": float(np.std(vals))
               for method, vals in baseline_ratios.items()},
        }

    def save_metrics(self):
        with open(self.outdir / "metrics.json", "w") as f:
            json.dump({"config": asdict(self.config), "history": self.metrics_history}, f, indent=2)

    def save_checkpoint(self):
        torch.save({"policy": self.agent.policy.state_dict(),
                    "optimizer": self.agent.optimizer.state_dict(),
                    "demand_scale": self.demand_scale},
                   self.outdir / "checkpoint.pt")

    def run(self) -> List[Dict]:
        start = time.perf_counter()
        for epoch in range(1, self.config.epochs + 1):
            epoch_stats = self.train_epoch(epoch)
            entry = {"epoch": epoch, **epoch_stats}

            if epoch % self.config.eval_every == 0 or epoch == self.config.epochs:
                eval_stats = self.evaluate()
                entry.update({f"test_{k}": v for k, v in eval_stats.items()})
                print(f"[epoch {epoch:4d}] train_reward={epoch_stats['train_reward_mean']:.4f} "
                      f"agent_U/U*={eval_stats['agent_ratio_mean']:.3f} "
                      f"prev={eval_stats['prev_ratio_mean']:.3f} "
                      f"avgk={eval_stats['avgk_ratio_mean']:.3f} "
                      f"obliv={eval_stats['oblivious_ratio_mean']:.3f} "
                      f"({time.perf_counter() - start:.0f}s)")

            self.metrics_history.append(entry)
            self.save_metrics()

        self.save_checkpoint()
        final_eval = self.evaluate()
        self.metrics_history[-1].update({f"final_{k}": v
                                         for k, v in final_eval.items()})
        self.save_metrics()
        return self.metrics_history
