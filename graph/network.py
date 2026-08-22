import heapq
from typing import List, Tuple, Dict, Optional
import numpy as np


class NetworkGraph:
    def __init__(self, num_nodes: int, edges: List[Tuple[int, int, float]]):
        self._num_nodes = num_nodes
        self._edges = edges
        self._adj: List[List[Tuple[int, float, int]]] = [[] for _ in range(num_nodes)]
        self._edge_to_idx: Dict[Tuple[int, int], int] = {}
        
        for idx, (u, v, cap) in enumerate(edges):
            self._adj[u].append((v, cap, idx))
            self._edge_to_idx[(u, v)] = idx

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @property
    def edges(self) -> List[Tuple[int, int]]:
        return [(u, v) for u, v, _ in self._edges]

    @property
    def capacities(self) -> np.ndarray:
        return np.array([cap for _, _, cap in self._edges], dtype=np.float32)

    def neighbors(self, u: int) -> List[int]:
        return [v for v, _, _ in self._adj[u]]

    def capacity(self, u: int, v: int) -> float:
        for v2, cap, _ in self._adj[u]:
            if v2 == v:
                return cap
        raise ValueError(f"No edge from {u} to {v}")

    def edge_index(self, u: int, v: int) -> int:
        return self._edge_to_idx[(u, v)]

    def dijkstra(self, source: int, weights: np.ndarray) -> np.ndarray:
        dist = np.full(self._num_nodes, np.inf, dtype=np.float64)
        dist[source] = 0.0
        settled = np.zeros(self._num_nodes, dtype=bool)
        pq: List[Tuple[float, int]] = [(0.0, source)]

        while pq:
            d, u = heapq.heappop(pq)
            if settled[u] or d > dist[u]:
                continue
            settled[u] = True
            for v, _, idx in self._adj[u]:
                nd = d + weights[idx]
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

    def all_pairs_shortest_paths(self, weights: np.ndarray) -> np.ndarray:
        sp = np.zeros((self._num_nodes, self._num_nodes), dtype=np.float32)
        for u in range(self._num_nodes):
            sp[u] = self.dijkstra(u, weights)
        return sp

    def get_edge_list(self) -> List[Tuple[int, int]]:
        return [(u, v) for u, v, _ in self._edges]


def create_12_node_topology() -> NetworkGraph:
    edges = [
        (0, 1, 100.0), (1, 0, 100.0),
        (0, 2, 100.0), (2, 0, 100.0),
        (1, 3, 100.0), (3, 1, 100.0),
        (1, 4, 100.0), (4, 1, 100.0),
        (2, 4, 100.0), (4, 2, 100.0),
        (2, 5, 100.0), (5, 2, 100.0),
        (3, 6, 100.0), (6, 3, 100.0),
        (3, 7, 100.0), (7, 3, 100.0),
        (4, 7, 100.0), (7, 4, 100.0),
        (4, 8, 100.0), (8, 4, 100.0),
        (5, 8, 100.0), (8, 5, 100.0),
        (5, 9, 100.0), (9, 5, 100.0),
        (6, 10, 100.0), (10, 6, 100.0),
        (7, 10, 100.0), (10, 7, 100.0),
        (7, 11, 100.0), (11, 7, 100.0),
        (8, 11, 100.0), (11, 8, 100.0),
        (9, 11, 100.0), (11, 9, 100.0),
    ]
    return NetworkGraph(12, edges)