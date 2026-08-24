import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from baselines.classical import AvgKBaseline, ObliviousRouting, PrevBaseline
from graph.network import NetworkGraph
from routing.optimal import OptimalCongestionCache
from softmin_routing.softmin import compute_multicommodity_flow, softmin_splitting_ratios
from sdn_ddpg.train.trainer import DDPGConfig, SdnDDPGTrainer
from traffic.generator import DemandMatrixGenerator


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _build_sequences(graph: NetworkGraph, gen_params: dict, count: int, length: int, out_bw=None, in_bw=None) -> List[List[np.ndarray]]:
    from routing.optimal import congestion_lower_bound

    generator = DemandMatrixGenerator(
        graph, model=gen_params["traffic_model"],
        sparsity=gen_params["sparsity"],
        elephant_frac=gen_params["elephant_frac"],
        seed=gen_params["seed"],
        out_bw=out_bw, in_bw=in_bw)

    sequences = []
    for _ in range(count):
        if gen_params.get("sequence_mode") == "cyclic":
            base = [generator.sample()
                    for _ in range(gen_params.get("cyclic_q", 6))]
            seq = [base[t % len(base)] for t in range(length)]
        else:
            seq = [generator.sample() for _ in range(length)]

        scaled = []
        for dm in seq:
            lb = congestion_lower_bound(graph, dm)
            if lb > 0:
                dm = (dm * (gen_params["target_lb"] / lb)).astype(np.float32)
            np.fill_diagonal(dm, 0.0)
            scaled.append(dm)
        sequences.append(scaled)
    return sequences


class MethodComparator:
    def __init__(self, config_name: str, results_root: str,
                 queue_service_rate: float = 3000.0,
                 queue_capacity: int = 10_000,
                 ddpg_variant: str = "paper"):
        self.config_name = config_name
        self.ddpg_variant = ddpg_variant
        self.ddpg_dirname = (f"{config_name}_sdn_td3" if ddpg_variant == "td3"
                             else f"{config_name}_sdn")
        self.root = Path(results_root)
        self.graph = None

        self.ppo_meta = _load_json(self.root / f"{config_name}_ltr"
                                   / "metrics.json")
        self.ddpg_meta = _load_json(self.root / self.ddpg_dirname / "metrics.json")

    def prepare(self):
        ppo_cfg = self.ppo_meta["config"]
        ddpg_cfg = self.ddpg_meta["config"]

        from graph.network import create_12_node_topology
        self.graph = create_12_node_topology()

        self.opt_cache = OptimalCongestionCache(self.graph)

        ppo_train_params = {
            "traffic_model": ppo_cfg["traffic_model"],
            "sparsity": ppo_cfg["sparsity"],
            "elephant_frac": ppo_cfg["elephant_frac"],
            "sequence_mode": ppo_cfg.get("sequence_mode", "iid"),
            "cyclic_q": ppo_cfg.get("cyclic_q", 6),
            "target_lb": ppo_cfg["target_lb"],
            "seed": ppo_cfg["seed"] + 1000,
        }
        ddpg_train_params = {
            "traffic_model": ddpg_cfg["traffic_model"],
            "sparsity": ddpg_cfg["sparsity"],
            "elephant_frac": ddpg_cfg["elephant_frac"],
            "sequence_mode": ddpg_cfg.get("sequence_mode", "iid"),
            "cyclic_q": ddpg_cfg.get("cyclic_q", 6),
            "target_lb": ddpg_cfg["target_lb"],
            "seed": ddpg_cfg["seed"] + 1000,
        }

        length = min(int(ppo_cfg.get("sequence_length", 40)), int(ddpg_cfg.get("sequence_length", 40)))

        self.k = int(ppo_cfg["history_k"])
        self._setup_ppo(ppo_cfg)
        self._setup_ddpg(ddpg_cfg)

        ppo_train_gen = DemandMatrixGenerator(
            self.graph, model=ppo_cfg["traffic_model"],
            sparsity=ppo_cfg["sparsity"],
            elephant_frac=ppo_cfg["elephant_frac"],
            seed=ppo_cfg["seed"])

        self.test_sequences = _build_sequences(
            self.graph, ppo_train_params,
            count=int(ppo_cfg["num_test_sequences"]), length=length,
            out_bw=ppo_train_gen.out_bw, in_bw=ppo_train_gen.in_bw)

        reference = _build_sequences(
            self.graph, ddpg_train_params,
            count=int(ddpg_cfg["num_test_sequences"]), length=length,
            out_bw=self.ddpg_trainer.train_generator.out_bw,
            in_bw=self.ddpg_trainer.train_generator.in_bw)
        for seq_a, seq_b in zip(self.test_sequences, reference):
            if not all(np.allclose(a, b, rtol=1e-4) for a, b in zip(seq_a, seq_b)):
                raise RuntimeError("PPO/DDPG test streams diverge - "
                                   "comparison would not be apples-to-apples")
        for seq_a, seq_b in zip(self.test_sequences,
                                self.ddpg_trainer.test_sequences):
            if not all(np.allclose(a, b, rtol=1e-4) for a, b in zip(seq_a, seq_b)):
                raise RuntimeError("rebuilt streams differ from the DDPG "
                                   "trainer's own evaluation streams")

        self.prev_baseline = PrevBaseline(self.graph, gamma=ppo_cfg["softmin_gamma"])

        self.avgk_baseline = AvgKBaseline(self.graph, k=self.k, gamma=ppo_cfg["softmin_gamma"])
        
        self.oblivious = None

    def _setup_ppo(self, cfg: dict):
        from agent.ppo import PPOAgent

        state_dim = self.k * self.graph.num_nodes ** 2
        agent = PPOAgent(state_dim=state_dim,
                         action_dim=len(self.graph.edges),
                         hidden_dim=int(cfg["hidden_dim"]),
                         gamma=float(cfg["rl_gamma"]),
                         gae_lambda=float(cfg["gae_lambda"]),
                         clip_eps=float(cfg["clip_eps"]),
                         entropy_coef=float(cfg["entropy_coef"]),
                         batch_size=int(cfg["batch_size"]))
        checkpoint = torch.load(self.root / f"{self.config_name}_ltr"
                                / "checkpoint.pt")
        agent.policy.load_state_dict(checkpoint["policy"])
        agent.policy.eval()
        self.ppo_agent = agent
        self.demand_scale = float(checkpoint["demand_scale"])

    def _setup_ddpg(self, cfg: dict):
        ddpg_config = DDPGConfig(**{
            key: value for key, value in cfg.items()
            if key in DDPGConfig.__dataclass_fields__
        })
        trainer = SdnDDPGTrainer(self.graph, ddpg_config)
        trainer.agent.load(str(self.root / self.ddpg_dirname
                               / "checkpoint.pt"))
        trainer.agent.actor.eval()
        self.ddpg_trainer = trainer

    def _ensure_oblivious(self):
        if self.oblivious is None:
            rng = np.random.default_rng(7)
            scenarios = []
            gen_params = {
                "traffic_model": self.ppo_meta["config"]["traffic_model"],
                "sparsity": self.ppo_meta["config"]["sparsity"],
                "elephant_frac": self.ppo_meta["config"]["elephant_frac"],
            }
            for _ in range(6):
                gen = DemandMatrixGenerator(
                    self.graph, model=gen_params["traffic_model"],
                    sparsity=gen_params["sparsity"],
                    elephant_frac=gen_params["elephant_frac"],
                    out_bw=self.ddpg_trainer.train_generator.out_bw,
                    in_bw=self.ddpg_trainer.train_generator.in_bw,
                    seed=int(rng.integers(1e6)))
                dm = gen.sample()
                from routing.optimal import congestion_lower_bound
                lb = congestion_lower_bound(self.graph, dm)
                if lb > 0:
                    dm = (dm * (self.ppo_meta["config"]["target_lb"]
                                / lb)).astype(np.float32)
                np.fill_diagonal(dm, 0.0)
                scenarios.append(dm)
            self.oblivious = ObliviousRouting(self.graph, scenarios)

    def _queue_projection(self, edge_loads: np.ndarray) -> Dict[str, float]:
        rates = np.zeros(self.graph.num_nodes)
        scaled = edge_loads * self.rate_scale
        for idx, (u, v) in enumerate(self.graph.edges):
            rates[u] += max(scaled[idx], 0.0)

        from sdn_ddpg.queue_model.delay_model import network_queue_metrics
        metrics = network_queue_metrics(
            rates, self.queue_service_rate, self.queue_capacity)
        return {"delay": metrics["mean_delay"],
                "loss": metrics["loss_fraction"]}

    def _ppo_loads(self, sequence: List[np.ndarray], end_idx: int):
        window = sequence[end_idx - self.k:end_idx] \
            if end_idx >= self.k else [sequence[0]] * (self.k - end_idx) + \
            sequence[:end_idx]
        stacked = np.log1p(np.stack(window) / self.demand_scale)
        state = stacked.ravel().astype(np.float32)

        action, _, _ = self.ppo_agent.act(state, deterministic=True)
        weights = np.exp(np.clip(action, -6.0, 6.0)).astype(np.float32)
        ratios = softmin_splitting_ratios(self.graph, weights, gamma=2.0)
        loads, delivered = compute_multicommodity_flow(self.graph, ratios, sequence[end_idx])
        utilization = float((loads / self.graph.capacities.astype(float)).max())
        return {"loads": loads, "utilization": utilization,
                "delivered": delivered}

    def _ddpg_eval(self, sequence: List[np.ndarray], end_idx: int):
        dm = sequence[end_idx]
        flows = self.ddpg_trainer._dm_to_flows(dm, scale=self.ddpg_trainer.rate_scale)
        state = self.ddpg_trainer.env.reset(flows)
        action = self.ddpg_trainer.agent.act(state, explore=False)
        _, reward, info = self.ddpg_trainer.env.step(action)
        return {"info": info}

    def _prev_eval(self, sequence, end_idx):
        util, weights = self.prev_baseline.routing_congestion(
            sequence[end_idx - 1], sequence[end_idx])
        ratios = softmin_splitting_ratios(self.graph, weights, gamma=2.0)
        loads, _ = compute_multicommodity_flow(self.graph, ratios, sequence[end_idx])
        return {"utilization": util, "loads": loads}

    def _avgk_eval(self, sequence, end_idx):
        history = sequence[max(0, end_idx - self.k):end_idx]
        util, weights = self.avgk_baseline.routing_congestion(history,
                                                              sequence[end_idx])
        ratios = softmin_splitting_ratios(self.graph, weights, gamma=2.0)
        loads, _ = compute_multicommodity_flow(self.graph, ratios,
                                               sequence[end_idx])
        return {"utilization": util, "loads": loads}

    def _oblivious_eval(self, dm):
        self._ensure_oblivious()
        util = self.oblivious.evaluate(dm)
        pair_demands = np.array([dm[u, v] for u, v in self.oblivious.pairs])
        loads = pair_demands @ self.oblivious.flows
        return {"utilization": util, "loads": loads}

    def run(self, stride: int = 1) -> Dict[str, Dict[str, float]]:
        self.queue_service_rate = self.ddpg_meta["config"]["service_rate"]
        self.queue_capacity = int(self.ddpg_meta["config"]["buffer_capacity"])
        self.prepare()
        self.rate_scale = self.ddpg_trainer.rate_scale

        records = {name: {"ratio": [], "delay": [], "loss": []}
                   for name in ("PPO-softmin", "DDPG-SDN",
                                "Prev", "Avg_k", "Oblivious")}

        for seq_id, sequence in enumerate(self.test_sequences):
            first_valid = max(self.k, 1)
            for end_idx in range(first_valid, len(sequence), stride):
                dm = sequence[end_idx]
                optimal = max(self.opt_cache.get(dm), 1e-9)

                ppo = self._ppo_loads(sequence, end_idx)
                records["PPO-softmin"]["ratio"].append(
                    ppo["utilization"] / optimal)
                proj = self._queue_projection(ppo["loads"])
                records["PPO-softmin"]["delay"].append(proj["delay"])
                records["PPO-softmin"]["loss"].append(proj["loss"])

                ddpg = self._ddpg_eval(sequence, end_idx)
                info = ddpg["info"]
                records["DDPG-SDN"]["ratio"].append(info["congestion_ratio"])
                proj = self._queue_projection(info["edge_loads"])
                records["DDPG-SDN"]["delay"].append(proj["delay"])
                records["DDPG-SDN"]["loss"].append(proj["loss"])
                if "ddpg_native" not in records:
                    records["ddpg_native"] = {"delay": [], "loss": []}
                records["ddpg_native"]["delay"].append(info["mean_delay"])
                records["ddpg_native"]["loss"].append(info["loss_fraction"])

                prev = self._prev_eval(sequence, end_idx)
                records["Prev"]["ratio"].append(prev["utilization"] / optimal)
                proj = self._queue_projection(prev["loads"])
                records["Prev"]["delay"].append(proj["delay"])
                records["Prev"]["loss"].append(proj["loss"])

                avgk = self._avgk_eval(sequence, end_idx)
                records["Avg_k"]["ratio"].append(avgk["utilization"] / optimal)
                proj = self._queue_projection(avgk["loads"])
                records["Avg_k"]["delay"].append(proj["delay"])
                records["Avg_k"]["loss"].append(proj["loss"])

                obl = self._oblivious_eval(dm)
                records["Oblivious"]["ratio"].append(obl["utilization"] / optimal)
                proj = self._queue_projection(obl["loads"])
                records["Oblivious"]["delay"].append(proj["delay"])
                records["Oblivious"]["loss"].append(proj["loss"])

        summary = {}
        for name, series in records.items():
            if name == "ddpg_native":
                continue
            summary[name] = {}
            for metric, values in series.items():
                clean = [v for v in values if np.isfinite(v)]
                summary[name][metric + "_mean"] = float(np.mean(clean)) if clean else float("nan")
                summary[name][metric + "_std"] = float(np.std(clean)) if clean else float("nan")

        summary["DDPG-SDN"]["delay_native_mean"] = \
            float(np.mean(records["ddpg_native"]["delay"]))
        summary["DDPG-SDN"]["loss_native_mean"] = \
            float(np.mean(records["ddpg_native"]["loss"]))
        return summary


def compare_config(config_name: str, results_root: str,
                   stride: int = 2, save: bool = True,
                   force: bool = False,
                   ddpg_variant: str = "paper") -> Dict:
    comparator = MethodComparator(config_name, results_root, ddpg_variant=ddpg_variant)
    
    summary = comparator.run(stride=stride)

    if save:
        out = Path(results_root) / "comparison.json"
        payload = {}
        if out.exists():
            payload = _load_json(out)

        existing = payload.get(config_name)
        if existing and not force and existing.get("_stride") != stride:
            raise RuntimeError(
                f"{config_name}: existing entry used stride "
                f"{existing.get('_stride')}, requested {stride}. "
                f"Re-run with matching --stride or pass --force.")

        payload[config_name] = {"_stride": stride,
                                "_ddpg_variant": ddpg_variant,
                                **summary}
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"\n[{config_name}] five-method comparison")
        header = f"{'method':<14}{'U/U*':>8}{'delay(s)':>10}{'loss':>9}"
        print(header)
        for name, m in summary.items():
            print(f"{name:<14}{m['ratio_mean']:>8.3f}"
                  f"{m['delay_mean']:>10.4f}{m['loss_mean']:>9.4f}")
    return summary
