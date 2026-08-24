# Architecture Design: Deep Learning Traffic Engineering

## Module Overview

Paper-specific modules live under `Learning-To-Route/`; shared libraries
(`graph/`, `traffic/`, `routing/optimal.py`, `eval/`) stay at the repo root.

| Module | Paper Section | Course Concept |
|--------|---------------|----------------|
| `graph/` (root) | §2 Network Model | DSA: Graph algorithms, Dijkstra, adjacency structures |
| `traffic/` (root) | §3.1 DM Generation | CN: Traffic models (gravity, bimodal), sparsity |
| `Learning-To-Route/softmin_routing/` | §5 Softmin Routing | CN: Multicommodity flow, hop-by-hop forwarding; DSA: Flow computation |
| `Learning-To-Route/baselines/` | §5 Baselines | CN: Classical TE (Prev, Avg_k, Oblivious) |
| `Learning-To-Route/agent/` | §4 RL Formulation, §5 Learning | AI: PPO (continuous control), neural networks; CN: State/action design |
| `Learning-To-Route/train/` | §5 Training Loop | AI: RL training loop, reward shaping, logging |
| `eval/` (root) | §5 Figure 2 | AI/CN: Benchmarking, congestion ratio plots |
| `routing/optimal.py` (root) | reward normalization U* | DSA/AI: sparse LP construction |

---

## Module Specifications

### 1. `graph/` - Network Graph & Shortest Paths

**Paper mapping**: §2 "Network" - G=(V,E,c), capacities; §5 "softmin routing uses shortest paths under weights w"

**Course concepts**: 
- **DSA**: Adjacency list/matrix, Dijkstra's algorithm (single-source shortest paths), graph traversal
- **CN**: Link capacities, directed topology

**API**:
```python
class NetworkGraph:
    def __init__(self, num_nodes: int, edges: List[Tuple[int, int, float]]):  # (u, v, capacity)
    def neighbors(self, u: int) -> List[int]
    def capacity(self, u: int, v: int) -> float
    def dijkstra(self, source: int, weights: Dict[Tuple[int,int], float]) -> Dict[int, float]
    def all_pairs_shortest_paths(self, weights: Dict[Tuple[int,int], float]) -> Dict[Tuple[int,int], float]
    @property
    def num_nodes(self) -> int
    @property
    def edges(self) -> List[Tuple[int, int]]
```

**Key functions**:
- `dijkstra(source, weights)` - returns distance to all nodes using `weights` as edge lengths
- `all_pairs_shortest_paths(weights)` - computes SP for all (u,d) pairs, used by softmin routing

**Data structures**: Adjacency list `adj[u] = [(v, capacity, edge_idx)]`, edge list `edges = [(u,v,cap)]`, edge→index mapping

---

### 2. `traffic/` - Demand Matrix Generator

**Paper mapping**: §3.1 "Generating DM sequences" - gravity model, bimodal model, sparsification

**Course concepts**:
- **CN**: Traffic matrix models (gravity = deterministic, bimodal = probabilistic heavy-tailed), sparsity
- **DSA**: Matrix operations, random sampling

**API**:
```python
def gravity_model(num_nodes: int, out_bw: np.ndarray, in_bw: np.ndarray, 
                  distances: np.ndarray) -> np.ndarray  # returns n×n DM

def bimodal_model(num_nodes: int, elephant_frac: float, 
                  out_bw: np.ndarray, in_bw: np.ndarray,
                  distances: np.ndarray) -> np.ndarray

def sparsify(DM: np.ndarray, p: float) -> np.ndarray  # keep p-fraction of pairs

def generate_cyclic_sequence(base_DMs: List[np.ndarray], length: int) -> List[np.ndarray]
def generate_iid_sequence(DM_distribution: Callable, length: int) -> List[np.ndarray]
```

**Parameters** (from paper):
- Gravity: D_{i,j} ∝ out_i × in_j / dist(i,j)²
- Bimodal: (1-elephant_frac) mice + elephant_frac elephants, elephants ~10× mice
- Sparsity p ∈ {0.3, 0.6, 0.9, 1.0}
- Bandwidths: 10s of MB to 10s of GB per vertex

---

### 3. `routing/` - Softmin Routing & Congestion Calculation

**Paper mapping**: §5 "Learning Softmin Routing" - Equations for SP_w, softmin, max-link-utilization

**Course concepts**:
- **CN**: Hop-by-hop forwarding, multicommodity flow, link utilization, congestion
- **DSA**: Flow propagation, matrix operations for multicommodity flow
- **AI**: Softmin as differentiable routing policy

**API**:
```python
def compute_shortest_path_distances(graph: NetworkGraph, weights: np.ndarray) -> np.ndarray:
    # Returns sp[u, d] = shortest distance u→d using weights as edge lengths
    # Shape: (n, n)

def compute_sp_via_neighbor(graph: NetworkGraph, weights: np.ndarray, sp: np.ndarray) -> np.ndarray:
    # SP_w(u, v, d) = w_{u,v} + sp[v, d]
    # Returns sp_via[u, v, d] for all u, v∈Γ(u), d
    # Shape: (n, n, n) but only valid for neighbors

def softmin_routing(sp_via: np.ndarray, gamma: float = 2.0) -> np.ndarray:
    # For each (u, d): ratios = softmax(-gamma * sp_via[u, :, d])
    # Returns splitting_ratios[u, v, d] ∈ [0,1], sums to 1 over v
    # Shape: (n, n, n)

def compute_multicommodity_flow(graph: NetworkGraph, 
                                splitting_ratios: np.ndarray, 
                                DM: np.ndarray) -> np.ndarray:
    # Returns flow[e] = total traffic on edge e
    # Iterative: flow = 0; for each s,t: inject DM[s,t] at s, propagate via splitting_ratios
    # Shape: (num_edges,)

def max_link_utilization(graph: NetworkGraph, flow: np.ndarray) -> float:
    # max_e flow[e] / capacity[e]
```

**Key equations**:
- SP_w(u,v,d) = w_{u,v} + min_{path v→d} Σ w_e
- R_{u,d}(v) = exp(-γ·SP_w(u,v,d)) / Σ_{v'} exp(-γ·SP_w(u,v',d))
- f_e = Σ_{s,t} DM_{s,t} × fraction of (s,t) flow on e
- U = max_e f_e / c_e

---

### 4. `Learning-To-Route/baselines/` - Classical TE Baselines

**Paper mapping**: §5 "We benchmark against three alternative non-ML-based approaches"

**Course concepts**:
- **CN**: Traditional TE approaches (reactive, average-based, oblivious)
- **AI**: Baseline policies for RL comparison

**API**:
```python
class PrevBaseline:
    # Optimize softmin weights for single most recent DM
    def compute_weights(self, graph: NetworkGraph, DM: np.ndarray) -> np.ndarray

class AvgKBaseline:
    # Optimize softmin weights for average of k recent DMs
    def compute_weights(self, graph: NetworkGraph, DM_history: List[np.ndarray]) -> np.ndarray

class ObliviousBaseline:
    # Optimal oblivious routing [Azar et al. 2003] - fixed strategy
    # Precompute once for graph, then apply to any DM
    def __init__(self, graph: NetworkGraph):
    def compute_splitting_ratios(self) -> np.ndarray  # fixed, no DM dependence
    def evaluate(self, graph: NetworkGraph, DM: np.ndarray) -> float  # returns U
```

**Implementation notes**:
- Prev / Avg_k: Solve for weights w that minimize max utilization for given DM(s) - use gradient descent or LP
- Oblivious: Polynomial-time LP from [Azar et al. STOC 2003] - simplified implementation using uniform splitting or precomputed optimal oblivious tree

---

### 5. `Learning-To-Route/agent/` - RL Agent (PPO for Continuous Control)

**Paper mapping**: §4 "Reinforcement Learning Approach", §5 "Learning Softmin Routing" - state=k-history DMs, action=|E| weights, 3-layer FCN

**Course concepts**:
- **AI**: PPO (policy gradient, clipped surrogate objective), actor-critic, GAE, experience buffer
- **CN**: State representation (k×n×n DM history), action interpretation (edge weights)
- **DSA**: Tensor operations for batching

**Network Architecture** (from paper: 3-layer FCN):
```
Input: flattened (k, n, n) → k·n²
Hidden 1: 128 units, ReLU
Hidden 2: 128 units, ReLU
Output: |E| units, linear (raw weights w_e)
```

**API**:
```python
class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_dim: int = 128, lr: float = 3e-4,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_eps: float = 0.2, epochs: int = 4, batch_size: int = 64):
    
    def act(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Returns (action, log_prob, value)
    
    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        # PPO clipped surrogate update
        # Returns loss dict
```

**RolloutBuffer** (on-policy, matches TRPO's trajectory collection):
```python
class RolloutBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
    def add(self, state, action, reward, next_state, done, log_prob, value):
    def get_batches(self, batch_size: int) -> Iterator[Dict]:
    def clear(self):
```

**State preprocessing**: Flatten k DMs → (k·n²,) vector, normalize by max bandwidth

---

### 6. `Learning-To-Route/train/` - Training Loop

**Paper mapping**: §5 "Evaluation" - training on 7 sequences × 60 DMs, 50 windows per sequence, k=10

**Course concepts**:
- **AI**: RL training loop, epoch/iteration tracking, reward computation, logging
- **CN**: Epoch = fixed time interval δt, DM revealed after routing decision

**API**:
```python
class Trainer:
    def __init__(self, graph: NetworkGraph, agent: PPOAgent, 
                 train_sequences: List[List[np.ndarray]],
                 test_sequences: List[List[np.ndarray]],
                 k: int = 10, gamma_softmin: float = 2.0):
    
    def compute_reward(self, weights: np.ndarray, DM: np.ndarray) -> float:
        # splitting = softmin_routing(weights)
        # flow = compute_multicommodity_flow(splitting, DM)
        # U = max_link_utilization(flow)
        # OPT = solve_lp_optimal(DM)  # or approximate
        # return -U / OPT
    
    def train_epoch(self) -> Dict[str, float]:
        # For each sequence, each window of k DMs:
        #   state = history[k]
        #   action = agent.act(state)
        #   reward = compute_reward(action, next_DM)
        #   buffer.add(...)
        # agent.update(buffer)
        # return metrics
    
    def evaluate(self) -> Dict[str, float]:
        # Run agent on test sequences, compute avg congestion ratio vs baselines
```

**Optimal congestion (OPT)**: Solve LP for min max utilization
- Variables: f_{s,t,e} ≥ 0 for each commodity (s,t) and edge e
- Constraints: Flow conservation at each node
- Objective: Minimize U s.t. Σ_{s,t} f_{s,t,e} ≤ U·c_e ∀e
- Use `scipy.optimize.linprog` or `cvxpy` for exact; approximate via multicommodity flow LP

---

### 7. `eval/` - Evaluation & Plotting

**Paper mapping**: §5 Figure 2 - "congestion ratio vs learning epochs, proposed vs Prev/Avg_k/Oblivious"

**Course concepts**:
- **AI**: Learning curves, statistical comparison, baseline benchmarking
- **CN**: Congestion ratio metric (U / OPT), TE evaluation methodology

**API**:
```python
def evaluate_all(methods: Dict[str, Callable], 
                 test_sequences: List[List[np.ndarray]],
                 graph: NetworkGraph,
                 k: int = 10) -> Dict[str, List[float]]:
    # Returns {method_name: [congestion_ratio_per_epoch]}

def plot_congestion_ratio(results: Dict[str, List[float]], 
                          save_path: str = "results/congestion_ratio.png"):
    # X-axis: learning epochs
    # Y-axis: congestion ratio (U / OPT)
    # Lines: Proposed (PPO), Prev, Avg_k, Oblivious
    # Matches Figure 2 style
```

**Metrics to track per epoch**:
- Mean congestion ratio on test sequences
- Std across test sequences
- Comparison to baselines (same test sequences)

---

## Data Flow Summary

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  traffic/   │────▶│   graph/    │◀───│  routing/   │
│  DM seqs    │     │  topology   │     │ softmin +   │
└─────────────┘     │  + caps     │     │ congestion  │
                    └──────┬──────┘     └──────┬──────┘
                           │                   │
                           ▼                   ▼
                    ┌─────────────┐     ┌─────────────┐
                    │   agent/    │     │  baselines/ │
                    │  PPO policy │     │ Prev/Avg_k/ │
                    │  (k·n²→|E|) │     │ Oblivious   │
                    └──────┬──────┘     └──────┬──────┘
                           │                   │
                           ▼                   ▼
                    ┌─────────────────────────────────┐
                    │           train/                │
                    │  Loop: state→action→reward→update│
                    │  Reward = -U/OPT via routing/    │
                    └─────────────────┬───────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │           eval/                 │
                    │  Congestion ratio vs epochs     │
                    │  Plot: Proposed vs 3 baselines  │
                    └─────────────────────────────────┘
```

---

## Dependencies

| Module | Depends On |
|--------|------------|
| `graph/` | `networkx` (optional, for viz), `numpy` |
| `traffic/` | `graph/` (for distances), `numpy` |
| `routing/` | `graph/`, `numpy` |
| `baselines/` | `graph/`, `routing/`, `scipy.optimize` (for LP) |
| `agent/` | `torch`, `numpy` |
| `train/` | All above |
| `eval/` | `train/`, `matplotlib` |

---

## File Structure

```
Deep-Learning-Traffic-Engineering/
├── docs/
│   ├── PAPER_ANALYSIS.md
│   └── ARCHITECTURE.md
├── graph/
│   ├── __init__.py
│   ├── network.py
│   └── test_graph.py
├── traffic/
│   ├── __init__.py
│   ├── generator.py
│   └── test_traffic.py
├── routing/
│   ├── __init__.py
│   ├── softmin.py
│   └── test_routing.py
├── baselines/
│   ├── __init__.py
│   ├── classical.py
│   └── test_baselines.py
├── agent/
│   ├── __init__.py
│   ├── ppo.py
│   └── test_agent.py
├── train/
│   ├── __init__.py
│   ├── loop.py
│   └── test_train.py
├── eval/
│   ├── __init__.py
│   ├── evaluate.py
│   └── plot.py
├── main.py           # Entry point
└── requirements.txt
```

Each module has a `test_*.py` for standalone sanity checks (Phase 2 requirement).
