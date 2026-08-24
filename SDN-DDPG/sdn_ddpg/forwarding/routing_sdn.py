from dataclasses import dataclass

import numpy as np

from graph.network import NetworkGraph
from sdn_ddpg.queue_model.delay_model import blocking_probability


@dataclass
class Flow:
    src: int
    dst: int
    rate: float


@dataclass
class FlowRoute:
    src: int
    dst: int
    rate: float
    nodes: list


def reverse_graph(graph: NetworkGraph) -> NetworkGraph:
    reversed_edges = [(v, u, cap) for u, v, cap in graph._edges]
    return NetworkGraph(graph.num_nodes, reversed_edges)


def reconstruct_path(graph: NetworkGraph,
                     source: int,
                     destination: int,
                     dist_to_destination: np.ndarray,
                     weights: np.ndarray) -> list:
    if source == destination:
        return [source]
    if not np.isfinite(dist_to_destination[source]):
        return [source]

    path = [source]
    current = source
    while current != destination:
        best_next = None
        best_value = np.inf
        for v in graph.neighbors(current):
            value = weights[graph.edge_index(current, v)] \
                + dist_to_destination[v]
            if value < best_value - 1e-9:
                best_value = value
                best_next = v
        if best_next is None:
            return path
        path.append(best_next)
        current = best_next
        if len(path) > 2 * graph.num_nodes:
            break
    return path


def route_flows(graph: NetworkGraph,
                weights: np.ndarray,
                flows: list) -> list:
    weights = np.asarray(weights, dtype=np.float64)
    reversed_view = reverse_graph(graph)

    unique_destinations = {flow.dst for flow in flows}
    dist_to_destination = {
        dst: reversed_view.dijkstra(dst, weights)
        for dst in unique_destinations
    }

    routes = []
    for flow in flows:
        nodes = reconstruct_path(graph, flow.src, flow.dst, dist_to_destination[flow.dst], weights)
        routes.append(FlowRoute(flow.src, flow.dst, float(flow.rate), nodes))
    return routes


def aggregate_arrival_rates(graph: NetworkGraph,
                            routes: list,
                            service_rate: float,
                            capacity: int,
                            refinement_passes: int = 2) -> tuple[np.ndarray,
                                                                 np.ndarray]:
    n = graph.num_nodes
    loss_probs = np.zeros(n)

    rates = np.zeros(n)
    for _ in range(max(1, refinement_passes)):
        rates = np.zeros(n)
        for route in routes:
            surviving = route.rate
            for hop, node in enumerate(route.nodes):
                rates[node] += surviving
                if hop < len(route.nodes) - 1:
                    surviving *= (1.0 - loss_probs[node])

        for node in range(n):
            if rates[node] > 0:
                loss_probs[node] = blocking_probability(
                    rates[node] / service_rate, capacity)
    return rates, loss_probs


def build_atvm(graph: NetworkGraph,
               routes: list,
               switch_loss_probs: np.ndarray,
               max_service_rate: float) -> np.ndarray:
    n = graph.num_nodes
    atvm = np.zeros((n, n))

    for route in routes:
        surviving = route.rate
        for hop in range(len(route.nodes) - 1):
            u, v = route.nodes[hop], route.nodes[hop + 1]
            atvm[u, v] += surviving
            surviving *= (1.0 - switch_loss_probs[u])

    if max_service_rate <= 0:
        return np.zeros_like(atvm)
    return np.minimum(atvm / max_service_rate, 1.0)
