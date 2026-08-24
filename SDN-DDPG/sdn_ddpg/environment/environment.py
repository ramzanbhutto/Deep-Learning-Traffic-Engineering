from typing import Dict, Tuple

import numpy as np

from graph.network import NetworkGraph
from sdn_ddpg.queue_model.delay_model import expected_switch_delay, network_queue_metrics
from sdn_ddpg.reward.reward import delay_reward, loss_reward, combined_reward
from sdn_ddpg.forwarding.routing_sdn import Flow, aggregate_arrival_rates, build_atvm, route_flows


class ModeledNetworkEnv:
    def __init__(self,
                 graph: NetworkGraph,
                 service_rate: float = 3000.0,
                 buffer_capacity: int = 10_000,
                 alpha: float = 0.9,
                 weight_low: float = 1.0,
                 weight_high: float = 5.0,
                 max_service_rate: float | None = None):
        self.graph = graph
        self.service_rate = service_rate
        self.buffer_capacity = buffer_capacity
        self.alpha = alpha
        self.weight_low = weight_low
        self.weight_high = weight_high
        self.max_service_rate = max_service_rate or service_rate

        self._flows: list = []
        self.current_weights = np.full(len(graph.edges), (weight_low + weight_high) / 2.0)
        self.current_state = np.zeros((graph.num_nodes, graph.num_nodes), dtype=np.float32)
        self.last_info: Dict = {}

    def reset(self, flows: list) -> np.ndarray:
        self._flows = list(flows)
        self.current_weights = np.full(
            len(self.graph.edges),
            (self.weight_low + self.weight_high) / 2.0)
        routes = route_flows(self.graph, self.current_weights, self._flows)
        rates, loss_probs = aggregate_arrival_rates(self.graph, routes, self.service_rate, self.buffer_capacity)
        self.current_state = build_atvm(self.graph, routes, loss_probs, self.max_service_rate).astype(np.float32)
        return self.current_state.ravel().copy()

    def step(self, weights: np.ndarray) -> Tuple[np.ndarray, float, Dict]:
        weights = np.asarray(weights, dtype=np.float64)

        routes = route_flows(self.graph, weights, self._flows)
        rates, loss_probs = aggregate_arrival_rates(self.graph, routes, self.service_rate, self.buffer_capacity)

        queue_metrics = network_queue_metrics(rates, self.service_rate, self.buffer_capacity)

        worst_hops = max((len(route.nodes) for route in routes), default=1)
        worst_path_drain_time = worst_hops * self.buffer_capacity / self.service_rate

        r_d = delay_reward(queue_metrics["mean_delay"], worst_path_drain_time)
        r_p = loss_reward(queue_metrics["loss_fraction"] * rates.sum(), rates.sum())
        reward = combined_reward(r_d, r_p, self.alpha)

        new_state = build_atvm(self.graph, routes, loss_probs, self.max_service_rate).astype(np.float32)

        edge_loads = np.zeros(len(self.graph.edges))
        for route in routes:
            surviving_fraction = 1.0 - queue_metrics["loss_fraction"]
            for hop in range(len(route.nodes) - 1):
                u, v = route.nodes[hop], route.nodes[hop + 1]
                edge_loads[self.graph.edge_index(u, v)] += \
                    route.rate * surviving_fraction

        optimal_cache = getattr(self, "_opt_cache", None)
        if optimal_cache is not None and self._flows:
            dm = np.zeros((self.graph.num_nodes,) * 2)
            for flow in self._flows:
                dm[flow.src, flow.dst] += flow.rate
            optimal = optimal_cache.get(dm.astype(np.float32))
            utilization = max(
                edge_loads[i] / cap for i, cap in
                enumerate(self.graph.capacities))
            congestion_ratio = utilization / max(optimal, 1e-9)
        else:
            utilization = max(
                (edge_loads[i] / cap for i, cap in
                 enumerate(self.graph.capacities)), default=0.0)
            congestion_ratio = np.nan

        info = {
            "mean_delay": queue_metrics["mean_delay"],
            "loss_fraction": queue_metrics["loss_fraction"],
            "throughput": queue_metrics["throughput"],
            "delay_reward": r_d,
            "loss_reward": r_p,
            "edge_loads": edge_loads,
            "utilization": float(utilization),
            "congestion_ratio": float(congestion_ratio),
            "worst_path_hops": worst_hops,
        }
        self.last_info = info

        previous_weights = self.current_weights
        self.current_weights = weights
        self.current_state = new_state

        return new_state.ravel().copy(), reward, info
