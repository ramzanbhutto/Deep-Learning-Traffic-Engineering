from typing import Callable, List
import numpy as np

from graph.network import NetworkGraph


def generate_node_bandwidths(num_nodes: int,
                             min_bw: float = 10.0,
                             max_bw: float = 10000.0,
                             rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    log_min = np.log10(min_bw)
    log_max = np.log10(max_bw)
    return 10.0 ** rng.uniform(log_min, log_max, size=num_nodes)


def pairwise_hop_distances(graph: NetworkGraph) -> np.ndarray:
    weights = np.ones(len(graph.edges), dtype=np.float32)
    return graph.all_pairs_shortest_paths(weights)


def gravity_model(out_bw: np.ndarray,
                  in_bw: np.ndarray,
                  distances: np.ndarray,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    n = len(out_bw)
    safe_dist = np.maximum(distances, 1.0)
    dm = np.outer(out_bw, in_bw) / (safe_dist ** 2)
    np.fill_diagonal(dm, 0.0)
    return dm.astype(np.float32)


def bimodal_model(out_bw: np.ndarray,
                  in_bw: np.ndarray,
                  distances: np.ndarray,
                  elephant_frac: float = 0.4,
                  elephant_ratio: float = 10.0,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    n = len(out_bw)
    safe_dist = np.maximum(distances, 1.0)
    base = np.outer(out_bw, in_bw) / (safe_dist ** 2)

    is_elephant = rng.random((n, n)) < elephant_frac
    dm = np.where(is_elephant, base * elephant_ratio, base)
    np.fill_diagonal(dm, 0.0)
    return dm.astype(np.float32)


def sparsify(dm: np.ndarray, p: float, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    n = dm.shape[0]
    mask = rng.random((n, n)) < p
    np.fill_diagonal(mask, False)
    sparse = dm * mask
    np.fill_diagonal(sparse, 0.0)
    return sparse.astype(np.float32)


class DemandMatrixGenerator:
    def __init__(self,
                 graph: NetworkGraph,
                 model: str = "gravity",
                 sparsity: float = 1.0,
                 elephant_frac: float = 0.4,
                 seed: int | None = None,
                 out_bw: np.ndarray | None = None,
                 in_bw: np.ndarray | None = None):
        self.graph = graph
        self.model = model
        self.sparsity = sparsity
        self.elephant_frac = elephant_frac
        self.rng = np.random.default_rng(seed)
        self.out_bw = out_bw if out_bw is not None else \
            generate_node_bandwidths(graph.num_nodes, rng=self.rng)
        self.in_bw = in_bw if in_bw is not None else \
            generate_node_bandwidths(graph.num_nodes, rng=self.rng)
        self.distances = pairwise_hop_distances(graph)

    def sample(self) -> np.ndarray:
        if self.model == "gravity":
            dm = gravity_model(self.out_bw, self.in_bw, self.distances, self.rng)
        elif self.model == "bimodal":
            dm = bimodal_model(self.out_bw, self.in_bw, self.distances, self.elephant_frac, rng=self.rng)
        else:
            raise ValueError(f"unknown model: {self.model}")
        if self.sparsity < 1.0:
            dm = sparsify(dm, self.sparsity, self.rng)
        return dm


def generate_iid_sequence(generator: DemandMatrixGenerator, length: int) -> List[np.ndarray]:
    return [generator.sample() for _ in range(length)]


def generate_cyclic_sequence(base_dms: List[np.ndarray], length: int) -> List[np.ndarray]:
    q = len(base_dms)
    return [base_dms[t % q] for t in range(length)]


def generate_averaged_sequence(seed_dms: List[np.ndarray], length: int, window: int = 5) -> List[np.ndarray]:
    seq = list(seed_dms[:window])
    while len(seq) < length:
        avg = np.mean(seq[-window:], axis=0)
        seq.append(avg.astype(np.float32))
    return seq[:length]


def scale_to_target_congestion(dm: np.ndarray, graph, target_lb: float = 0.5) -> np.ndarray:
    from routing.optimal import congestion_lower_bound

    current_lb = congestion_lower_bound(graph, dm)
    if current_lb <= 0:
        return dm
    factor = target_lb / current_lb
    return (dm * factor).astype(np.float32)
