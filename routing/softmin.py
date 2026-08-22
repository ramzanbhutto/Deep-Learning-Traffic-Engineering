import numpy as np

from graph.network import NetworkGraph


def compute_shortest_path_distances(graph: NetworkGraph, weights: np.ndarray) -> np.ndarray:
    return graph.all_pairs_shortest_paths(weights)


def compute_sp_via_neighbor(graph: NetworkGraph, weights: np.ndarray, sp: np.ndarray) -> np.ndarray:
    n = graph.num_nodes
    sp_via = np.full((n, n, n), np.inf, dtype=np.float64)
    for u in range(n):
        for v in graph.neighbors(u):
            w_uv = weights[graph.edge_index(u, v)]
            sp_via[u, v, :] = w_uv + sp[v, :]
    return sp_via


def softmin_splitting_ratios(graph: NetworkGraph, weights: np.ndarray, gamma: float = 2.0) -> np.ndarray:
    n = graph.num_nodes
    weights = np.asarray(weights, dtype=np.float64)

    sp = compute_shortest_path_distances(graph, weights)
    sp_via = compute_sp_via_neighbor(graph, weights, sp)

    finite = np.isfinite(sp_via)
    reachable_any = finite.any(axis=1, keepdims=True)
    safe_scores = np.where(finite, sp_via, 0.0)

    shifted = np.full_like(sp_via, np.inf)
    for u in range(n):
        if not reachable_any[u, 0, :].any():
            continue
        col_min = np.where(finite[u], sp_via[u], np.inf).min(axis=0)
        shifted[u] = sp_via[u] - col_min

    logits = np.where(finite, np.exp(np.clip(-gamma * shifted, -700.0, 700.0)), 0.0)
    totals = logits.sum(axis=1, keepdims=True)
    has_mass = totals[:, 0, :] > 0
    uniform = np.zeros_like(logits)

    for u in range(n):
        degenerate = ~has_mass[u]
        if not degenerate.any():
            continue
        for d in np.where(degenerate)[0]:
            if d == u:
                continue
            nbr_count = len(graph.neighbors(u))
            if nbr_count == 0:
                continue
            uniform[u, graph.neighbors(u), d] = 1.0 / nbr_count

    ratios = np.where(has_mass[:, None, :], logits / np.maximum(totals, 1e-300), uniform)

    idx = np.arange(n)
    ratios[idx, :, idx] = 0.0
    return ratios.astype(np.float32)


class FlowPropagator:
    def __init__(self, graph: NetworkGraph):
        self.graph = graph
        self.n = graph.num_nodes
        self.num_edges = len(graph.edges)

        edge_array = np.array([(u, v) for u, v in graph.edges])
        self.tails = edge_array[:, 0]
        self.heads = edge_array[:, 1]

        neighbor_ratio_index = np.zeros((self.n, self.n), dtype=bool)
        for u in range(self.n):
            for v in graph.neighbors(u):
                neighbor_ratio_index[u, v] = True
        self.neighbor_mask = neighbor_ratio_index

    def propagate(self,
                  ratios: np.ndarray,
                  dm: np.ndarray,
                  max_steps: int = 200,
                  tolerance: float = 1e-3) -> tuple[np.ndarray, float]:
        total_demand = dm.sum()
        if total_demand <= 0:
            return np.zeros(self.num_edges), 1.0

        remaining = dm.astype(np.float64).T.copy()
        edge_flow = np.zeros(self.num_edges)

        ratio_edge = ratios[self.tails, self.heads, :].astype(np.float64)

        for _ in range(max_steps):
            undelivered = remaining.sum()
            if undelivered <= tolerance * total_demand:
                break

            arriving = np.einsum("du,uvd->dv", remaining, ratios, optimize=True)

            loads = (ratio_edge * remaining[:, self.tails].T).sum(axis=1)
            edge_flow += loads

            idx = np.arange(self.n)
            arriving[:, idx] -= np.einsum("du,ud->d", remaining, ratios[:, idx, idx], optimize=True)
            arriving[idx, idx] = 0.0
            remaining = np.maximum(arriving, 0.0)

        delivered_fraction = 1.0 - remaining.sum() / total_demand
        return edge_flow, float(delivered_fraction)


_default_propagators: dict[int, FlowPropagator] = {}


def get_propagator(graph: NetworkGraph) -> FlowPropagator:
    key = id(graph)
    if key not in _default_propagators:
        _default_propagators[key] = FlowPropagator(graph)
    return _default_propagators[key]


def compute_multicommodity_flow(graph: NetworkGraph,
                                ratios: np.ndarray,
                                dm: np.ndarray,
                                max_steps: int = 200,
                                tolerance: float = 1e-3) -> tuple[np.ndarray, float]:
    propagator = get_propagator(graph)
    return propagator.propagate(ratios, dm, max_steps, tolerance)


def max_link_utilization(graph: NetworkGraph, edge_flow: np.ndarray) -> float:
    capacities = graph.capacities.astype(np.float64)
    utilization = edge_flow / capacities
    return float(utilization.max())


def congestion_ratio(graph: NetworkGraph,
                     ratios: np.ndarray,
                     dm: np.ndarray,
                     optimal_utilization: float | None = None) -> tuple[float, float]:
    edge_flow, delivered = compute_multicommodity_flow(graph, ratios, dm)
    utilization = max_link_utilization(graph, edge_flow)
    if optimal_utilization is None:
        optimal_utilization = utilization
    return utilization / optimal_utilization, delivered
