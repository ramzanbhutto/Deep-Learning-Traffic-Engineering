from typing import List, Optional

import numpy as np
from scipy import sparse
from scipy.optimize import linprog, minimize

from graph.network import NetworkGraph
from routing.optimal import active_commodities, optimal_congestion_lp
from softmin_routing.softmin import (softmin_splitting_ratios, compute_multicommodity_flow, max_link_utilization)


def congestion_of_weights(graph: NetworkGraph, weights: np.ndarray, dms: List[np.ndarray], gamma: float = 2.0) -> float:
    ratios = softmin_splitting_ratios(graph, weights, gamma)
    total = 0.0
    for dm in dms:
        edge_flow, _ = compute_multicommodity_flow(graph, ratios, dm)
        total += max_link_utilization(graph, edge_flow)
    return total / len(dms)


class SoftminWeightOptimizer:
    def __init__(self, graph: NetworkGraph, gamma: float = 2.0):
        self.graph = graph
        self.gamma = gamma
        self.num_edges = len(graph.edges)
        self.last_log_weights: Optional[np.ndarray] = None

    def optimize(self, dms: List[np.ndarray], maxfev: int = 400) -> np.ndarray:
        def objective(log_w: np.ndarray) -> float:
            log_w = np.clip(log_w, -6.0, 6.0)
            weights = np.exp(log_w)
            return congestion_of_weights(self.graph, np.asarray(weights, dtype=np.float32), dms, self.gamma)

        x0 = self.last_log_weights if self.last_log_weights is not None \
            else np.zeros(self.num_edges)

        result = minimize(objective, x0, method="Powell",
                          options={"maxfev": maxfev, "xtol": 1e-3, "ftol": 1e-4})

        best = result.x if result.fun < objective(x0) else x0
        self.last_log_weights = np.clip(np.asarray(best), -6.0, 6.0)
        return np.exp(self.last_log_weights).astype(np.float32)

    def reset(self):
        self.last_log_weights = None


class PrevBaseline:
    def __init__(self, graph: NetworkGraph, gamma: float = 2.0, maxfev: int = 300):
        self.graph = graph
        self.optimizer = SoftminWeightOptimizer(graph, gamma)
        self.maxfev = maxfev

    def routing_congestion(self, prev_dm: np.ndarray, eval_dm: np.ndarray) -> tuple[float, np.ndarray]:
        weights = self.optimizer.optimize([prev_dm], maxfev=self.maxfev)
        return self._evaluate(weights, eval_dm), weights

    def _evaluate(self, weights: np.ndarray, eval_dm: np.ndarray) -> float:
        ratios = softmin_splitting_ratios(self.graph, weights, self.optimizer.gamma)
        edge_flow, _ = compute_multicommodity_flow(self.graph, ratios, eval_dm)
        return max_link_utilization(self.graph, edge_flow)


class AvgKBaseline:
    def __init__(self, graph: NetworkGraph, k: int = 10, gamma: float = 2.0,
                 maxfev: int = 300):
        self.graph = graph
        self.k = k
        self.optimizer = SoftminWeightOptimizer(graph, gamma)
        self.maxfev = maxfev

    def routing_congestion(self, history: List[np.ndarray], eval_dm: np.ndarray) -> tuple[float, np.ndarray]:
        recent = history[-self.k:]
        mean_dm = np.mean(np.stack(recent), axis=0).astype(np.float32)
        weights = self.optimizer.optimize([mean_dm], maxfev=self.maxfev)
        return self._evaluate(weights, eval_dm), weights

    def _evaluate(self, weights: np.ndarray, eval_dm: np.ndarray) -> float:
        ratios = softmin_splitting_ratios(self.graph, weights, self.optimizer.gamma)
        edge_flow, _ = compute_multicommodity_flow(self.graph, ratios, eval_dm)
        return max_link_utilization(self.graph, edge_flow)


class ObliviousRouting:
    def __init__(self, graph: NetworkGraph, scenario_dms: List[np.ndarray]):
        self.graph = graph
        self.n = graph.num_nodes
        self.num_edges = len(graph.edges)
        self.pairs = [(u, v) for u in range(self.n)
                      for v in range(self.n) if u != v]
        self.num_pairs = len(self.pairs)
        self._solve(scenario_dms)

    def _unit_flow_constraints(self):
        n, E = self.n, self.num_edges
        incidence = np.zeros((n, E))
        for idx, (u, v) in enumerate(self.graph.edges):
            incidence[u, idx] = 1.0
            incidence[v, idx] = -1.0

        eq_rows, eq_rhs = [], []
        for c, (s, t) in enumerate(self.pairs):
            block = sparse.hstack(
                [sparse.csr_matrix((n, c * E)),
                 sparse.csr_matrix(incidence),
                 sparse.csr_matrix((n, (self.num_pairs - c - 1) * E + 1))],
                format="csr")
            rhs = np.zeros(n)
            rhs[s] = 1.0
            rhs[t] = -1.0
            eq_rows.append(block)
            eq_rhs.append(rhs)
        a_eq = sparse.vstack(eq_rows, format="csr")
        b_eq = np.concatenate(eq_rhs)
        return a_eq, b_eq

    def _solve(self, scenario_dms: List[np.ndarray]):
        P, E = self.num_pairs, self.num_edges
        num_vars = P * E + 1

        a_eq, b_eq = self._unit_flow_constraints()

        capacities = self.graph.capacities.astype(np.float64)
        scenario_opts = [optimal_congestion_lp(self.graph, dm)
                         for dm in scenario_dms]

        rows, cols, data = [], [], []
        row_count = 0
        for i, dm in enumerate(scenario_dms):
            opt_i = max(scenario_opts[i], 1e-9)
            dense_dm = np.asarray(dm, dtype=np.float64)
            for e in range(E):
                for c, (u, v) in enumerate(self.pairs):
                    amount = dense_dm[u, v]
                    if amount > 0:
                        rows.append(row_count)
                        cols.append(c * E + e)
                        data.append(amount / opt_i)
                rows.append(row_count)
                cols.append(num_vars - 1)
                data.append(-capacities[e])
                row_count += 1

        a_ub = sparse.csr_matrix((data, (rows, cols)), shape=(row_count, num_vars))
        b_ub = np.zeros(row_count)

        objective = np.zeros(num_vars)
        objective[-1] = 1.0

        result = linprog(objective, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=[(0, None)] * num_vars, method="highs")

        if not result.success:
            raise RuntimeError(f"oblivious LP failed: {result.message}")

        self.flows = result.x[:-1].reshape(P, E)
        self.scenario_ratio = float(result.x[-1])

    def evaluate(self, dm: np.ndarray) -> float:
        pair_demands = np.array([dm[u, v] for u, v in self.pairs], dtype=np.float64)
        loads = pair_demands @ self.flows
        utilization = loads / self.graph.capacities.astype(np.float64)
        return float(utilization.max())
