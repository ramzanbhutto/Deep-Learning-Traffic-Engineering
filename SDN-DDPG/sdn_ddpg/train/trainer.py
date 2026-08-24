import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from graph.network import NetworkGraph
from routing.optimal import OptimalCongestionCache
from sdn_ddpg.agent.ddpg import DDPGAgent
from sdn_ddpg.environment.environment import ModeledNetworkEnv
from sdn_ddpg.forwarding.routing_sdn import Flow, aggregate_arrival_rates, route_flows
from traffic.generator import DemandMatrixGenerator


@dataclass
class DDPGConfig:
    traffic_model: str = "gravity"
    sparsity: float = 0.3
    elephant_frac: float = 0.4
    target_lb: float = 0.5
    sequence_mode: str = "iid"
    cyclic_q: int = 6
    num_train_sequences: int = 7
    num_test_sequences: int = 3
    sequence_length: int = 40
    service_rate: float = 3000.0
    buffer_capacity: int = 10_000
    alpha: float = 0.9
    weight_low: float = 1.0
    weight_high: float = 5.0
    target_hotspot_rho: float = 0.9
    hidden_dims: tuple = (256, 256)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    buffer_capacity_rl: int = 50_000
    batch_size: int = 100
    warmup_steps: int = 100
    update_every: int = 2
    noise_sigma: float = 0.15
    use_twin_critics: bool = False
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5
    reward_scale: float = 1.0
    epochs: int = 80
    eval_every: int = 5
    seed: int = 42
    outdir: str = "results/run"


class SdnDDPGTrainer:
    def __init__(self, graph: NetworkGraph, config: DDPGConfig):
        self.graph = graph
        self.config = config

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

        self.env = ModeledNetworkEnv(graph,
                                     service_rate=config.service_rate,
                                     buffer_capacity=config.buffer_capacity,
                                     alpha=config.alpha,
                                     weight_low=config.weight_low,
                                     weight_high=config.weight_high)

        self.agent = DDPGAgent(
            state_dim=graph.num_nodes * graph.num_nodes,
            action_dim=len(graph.edges),
            action_low=config.weight_low,
            action_high=config.weight_high,
            hidden_dims=tuple(config.hidden_dims),
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            gamma=config.gamma,
            tau=config.tau,
            buffer_capacity=config.buffer_capacity_rl,
            batch_size=config.batch_size,
            warmup_steps=config.warmup_steps,
            noise_sigma=config.noise_sigma,
            seed=config.seed,
            use_twin_critics=config.use_twin_critics,
            target_policy_noise=config.target_policy_noise,
            target_noise_clip=config.target_noise_clip)

        self.opt_cache = OptimalCongestionCache(graph)
        self.env._opt_cache = self.opt_cache

        self.rate_scale = self._calibrate_rates()
        self.outdir = Path(config.outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.metrics_history: List[Dict] = []

    def _make_sequences(self, generator: DemandMatrixGenerator, count: int) -> List[List[np.ndarray]]:
        from routing.optimal import congestion_lower_bound

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

    def _dm_to_flows(self, dm: np.ndarray, scale: float = 1.0) -> List[Flow]:
        flows = []
        n = dm.shape[0]
        for s in range(n):
            for t in range(n):
                if s != t and dm[s, t] > 0:
                    flows.append(Flow(src=s, dst=t,
                                      rate=float(dm[s, t]) * scale))
        return flows

    def _calibrate_rates(self) -> float:
        probe_dm = self.train_sequences[0][0]
        probe_flows = self._dm_to_flows(probe_dm)
        mid_weight = (self.config.weight_low + self.config.weight_high) / 2.0
        weights = np.full(len(self.graph.edges), mid_weight)
        routes = route_flows(self.graph, weights, probe_flows)
        rates, _ = aggregate_arrival_rates(self.graph, routes, self.config.service_rate, self.config.buffer_capacity)

        peak = rates.max()
        if peak <= 0:
            return 1.0
        return self.config.target_hotspot_rho * \
            self.config.service_rate / peak

    def _evaluate_window(self, sequence: List[np.ndarray], end_idx: int) -> Dict:
        dm = sequence[end_idx]
        flows = self._dm_to_flows(dm, scale=self.rate_scale)
        state = self.env.reset(flows)
        action = self.agent.act(state, explore=False)
        _, reward, info = self.env.step(action)
        result = {
            "reward": reward,
            "mean_delay": info["mean_delay"],
            "loss_fraction": info["loss_fraction"],
            "throughput": info["throughput"],
        }
        if np.isfinite(info["congestion_ratio"]):
            result["congestion_ratio"] = info["congestion_ratio"]
        return result

    def evaluate(self) -> Dict[str, float]:
        records = []
        for sequence in self.test_sequences:
            for end_idx in range(0, len(sequence)):
                records.append(self._evaluate_window(sequence, end_idx))

        def mean_of(key):
            values = [r[key] for r in records if key in r]
            return float(np.mean(values)) if values else float("nan")

        return {
            "agent_reward_mean": mean_of("reward"),
            "agent_delay_mean": mean_of("mean_delay"),
            "agent_loss_fraction_mean": mean_of("loss_fraction"),
            "agent_throughput_mean": mean_of("throughput"),
            "agent_ratio_mean": mean_of("congestion_ratio"),
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        rewards, delays, losses = [], [], []
        critic_losses, policy_losses = [], []
        update_count = 0

        for seq_id, sequence in enumerate(self.train_sequences):
            self.agent.noise.reset()
            for end_idx in range(len(sequence)):
                dm = sequence[end_idx]
                flows = self._dm_to_flows(dm, scale=self.rate_scale)

                state = self.env.reset(flows)
                action = self.agent.act(state, explore=True)
                next_state, reward, info = self.env.step(action)
                reward *= self.config.reward_scale

                done = end_idx == len(sequence) - 1
                self.agent.observe(state, action, reward, next_state, done)

                if self.agent.ready_to_update() and \
                        update_count % self.config.update_every == 0:
                    stats = self.agent.update()
                    critic_losses.append(stats["critic_loss"])
                    if stats["policy_loss"] is not None:
                        policy_losses.append(stats["policy_loss"])
                update_count += 1

                rewards.append(reward)
                delays.append(info["mean_delay"])
                losses.append(info["loss_fraction"])

        return {
            "epoch": epoch,
            "train_reward_mean": float(np.mean(rewards)),
            "train_delay_mean": float(np.mean(delays)),
            "train_loss_mean": float(np.mean(losses)),
            "critic_loss_mean": float(np.mean(critic_losses)) if critic_losses else float("nan"),
            "policy_loss_mean": float(np.mean(policy_losses)) if policy_losses else float("nan"),
        }

    def save_metrics(self):
        with open(self.outdir / "metrics.json", "w") as f:
            json.dump({"config": asdict(self.config),
                       "method": "sdn-ddpg",
                       "history": self.metrics_history}, f, indent=2)

    def save_checkpoint(self):
        self.agent.save(str(self.outdir / "checkpoint.pt"))

    def run(self) -> List[Dict]:
        start = time.perf_counter()
        for epoch in range(1, self.config.epochs + 1):
            entry = self.train_epoch(epoch)

            if epoch % self.config.eval_every == 0 or \
                    epoch == self.config.epochs:
                eval_stats = self.evaluate()
                entry.update({f"test_{k}": v
                              for k, v in eval_stats.items()})
                print(f"[epoch {epoch:4d}] train_reward={entry['train_reward_mean']:.4f} "
                      f"test_reward={eval_stats['agent_reward_mean']:.4f} "
                      f"delay={eval_stats['agent_delay_mean']:.4f}s "
                      f"loss={eval_stats['agent_loss_fraction_mean']:.4f} "
                      f"U/U*={eval_stats['agent_ratio_mean']:.3f} "
                      f"({time.perf_counter() - start:.0f}s)")

            self.metrics_history.append(entry)
            self.save_metrics()

        self.save_checkpoint()
        final_eval = self.evaluate()
        self.metrics_history[-1].update({f"final_{k}": v
                                         for k, v in final_eval.items()})
        self.save_metrics()
        return self.metrics_history
