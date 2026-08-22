from typing import List, Tuple

import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from graph.network import NetworkGraph


def active_commodities(dm: np.ndarray, tol: float = 1e-12) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    pairs = [(s, t) for s in range(dm.shape[0])
             for t in range(dm.shape[1]) if s != t and dm[s, t] > tol]
    demands = np.array([dm[s, t] for s, t in pairs], dtype=np.float64)
    return pairs, demands


def optimal_congestion_lp(graph: NetworkGraph, dm: np.ndarray) -> float:
    n = graph.num_nodes
    num_edges = len(graph.edges)
    pairs, demands = active_commodities(dm)

    if not pairs:
        return 0.0

    num_commodities = len(pairs)
    num_vars = num_commodities * num_edges + 1
    capacities = graph.capacities.astype(np.float64)

    edge_array = np.array(graph.edges)
    tails = edge_array[:, 0]
    heads = edge_array[:, 1]

    base_rows = np.concatenate([tails, heads])
    base_cols = np.tile(np.arange(num_edges), 2)
    base_data = np.concatenate([np.ones(num_edges), -np.ones(num_edges)])

    eq_row_idx = (np.repeat(np.arange(num_commodities), 2 * num_edges) * n + np.tile(base_rows, num_commodities))
    eq_col_idx = (np.repeat(np.arange(num_commodities), 2 * num_edges) * num_edges + np.tile(base_cols, num_commodities))
    eq_data = np.tile(base_data, num_commodities)

    a_eq = sparse.csr_matrix((eq_data, (eq_row_idx, eq_col_idx)), shape=(num_commodities * n, num_vars))

    b_eq = np.zeros(num_commodities * n)
    supply_rows = np.arange(num_commodities) * n
    b_eq[supply_rows + np.array([s for s, _ in pairs])] = demands
    b_eq[supply_rows + np.array([t for _, t in pairs])] = -demands

    ub_edge_idx = np.tile(np.arange(num_edges), num_commodities)
    ub_col_idx = np.array([c * num_edges + e
                           for c in range(num_commodities)
                           for e in range(num_edges)])
    ub_data = np.ones(num_edges * num_commodities)
    cap_col = np.arange(num_edges)
    rows = np.concatenate([ub_edge_idx, cap_col])
    cols = np.concatenate([ub_col_idx, np.full(num_edges, num_vars - 1)])

    data = np.concatenate([ub_data, -capacities])
    a_ub = sparse.csr_matrix((data, (rows, cols)), shape=(num_edges, num_vars))
    b_ub = np.zeros(num_edges)

    objective = np.zeros(num_vars)
    objective[-1] = 1.0

    bounds = [(0, None)] * num_vars

    result = linprog(objective, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if not result.success:
        raise RuntimeError(f"OPT LP failed: {result.message}")

    return float(result.x[-1])


def congestion_lower_bound(graph: NetworkGraph, dm: np.ndarray) -> float:
    capacities = graph.capacities.astype(np.float64)
    lb = 0.0
    total_cap = capacities.sum()
    total_demand = dm.sum()
    if total_demand > 0:
        lb = max(lb, total_demand / total_cap)

    for t in range(dm.shape[0]):
        inflow = dm[:, t].sum()
        cap_in = sum(capacities[i]
                     for i, (u, v) in enumerate(graph.edges) if v == t)
        if cap_in > 0:
            lb = max(lb, inflow / cap_in)

    for s in range(dm.shape[0]):
        outflow = dm[s, :].sum()
        cap_out = sum(capacities[i]
                      for i, (u, v) in enumerate(graph.edges) if u == s)
        if cap_out > 0:
            lb = max(lb, outflow / cap_out)

    return float(lb)


class OptimalCongestionCache:
    def __init__(self, graph: NetworkGraph, solver=optimal_congestion_lp):
        self.graph = graph
        self.solver = solver
        self.cache = {}
        self.misses = 0

    def get(self, dm: np.ndarray) -> float:
        key = dm.tobytes()
        if key not in self.cache:
            self.cache[key] = self.solver(self.graph, dm)
            self.misses += 1
        return self.cache[key]
