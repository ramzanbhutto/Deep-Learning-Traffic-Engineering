# Architecture: SDN-DDPG Method (Kim, Kim & Lim, IEEE Access 2022)

*Second method in the Deep Learning Traffic Engineering project. Lives beside -
never inside - the verified Learning-To-Route stack.*

---

## 1. Package Placement

New top-level package **`sdn_ddpg/`**, sibling to the existing modules:

```
sdn_ddpg/
├── queue_model/           M/M/1/K per-switch analytics           (paper Eqs. 1-3)
│   └── delay_model.py + test
├── forwarding/            weighted shortest-path flow routing + ATVM (Eqs. 9-10)
│   └── routing_sdn.py + test
├── reward/                r_d, r_p, combined R                   (Eqs. 12-14)
│   └── reward.py + test
├── agent/                 DDPG actor/critic, targets, OU, replay (Eqs. 18-21)
│   └── ddpg.py + test
├── environment/           modeled-network step function          (Alg. 1 line 10)
│   └── environment.py + test
├── train/                 offline training + evaluation loop     (Alg. 1)
│   └── trainer.py + test
└── comparison/
    └── compare.py         five-method harness (PPO+DDPG checkpoints +
                            Prev/Avg_k/Oblivious on shared streams through
                            one common M/M/1/K projection)
```

`routing_sdn.py` added beyond the brief's minimum list: the paper's forwarding
(single-path weighted Dijkstra per flow) and its ATVM state builder are the
heart of the method and deserve their own tested module - bundling them into
the trainer would repeat the monolith mistake the Learning-To-Route layout
avoids.

## 2. Reuse Plan - Zero Modifications to Existing Code

| Existing asset | Used as-is for | Touch required |
|---|---|---|
| `graph.network.NetworkGraph` | topology, `dijkstra(source, weights)` distance rows | **none** |
| `traffic.generator.DemandMatrixGenerator` + sequences | identical demand streams fed to both methods | **none** |
| `routing.optimal.optimal_congestion_lp` / cache | U* reference for the cross-method U/U* column | **none** (read-only import) |
| `Learning-To-Route/baselines/classical.py` results | Prev / Avg_k / Oblivious columns of the joint chart | **none** |

**Shortest-path *reconstruction* without modifying `graph/`:** the paper needs
actual next-hops/paths (ATVM indicators x_ij^k), but `NetworkGraph.dijkstra`
returns distances only - by design. Rather than change verified code,
`routing_sdn.py` reconstructs paths by greedy descent: from node u, repeatedly
step to the neighbor v minimizing `w(u,v) + dist[v]`. This walks the exact
shortest-path DAG and terminates at the destination; cost O(path length).
Justification: provably equivalent to parent-pointer reconstruction, zero risk
to existing modules, and independently unit-tested.

## 3. Topology Decision

**Primary: reuse the existing 12-node/34-edge graph** for the headline
comparison - apples-to-apples with Learning-To-Route is the entire point of
Phase 3. The paper's smallest topology (InternetMCI, 19 nodes/33 links) may be
added later as a scale-out sanity check via a factory function in
`sdn_ddpg/topologies.py`; not needed for the core deliverable.

## 4. Traffic Convention (units mapping)

Paper operates in packets: flow rates λ ∈ [10,300] pkt/s, μ = 3000 pkt/s,
K = 10 000 pkt. Mapping adopted:

- Each nonzero DM entry (s,t,demand) becomes **one flow** with Poisson rate
  **λ^k = demand** (packet/s) - our demand matrices are already scaled to a
  regime (cut lower bound ≈ 0.5) whose per-switch aggregates land in the same
  10²-10³ pkt/s band as the paper's.
- Switch parameters copied verbatim from paper Table 2: **μ = 3000 pkt/s,
  K = 10 000 pkt**, homogeneous across switches (config knobs
  `service_rate`, `buffer_capacity` kept for experiments).

Smoke-test gate: verify hotspot utilization ρ lands in [0.3, 1.4]; if a regime
saturates (>90 % of switches at ρ > 1) the service-rate knob absorbs it rather
than distorting the reward.

## 5. Module Specifications

### 5.1 `delay_model.py` - M/M/1/K analytics

```python
def blocking_probability(rho: float, capacity: int) -> float          # Eq. (2)
def expected_queue_occupation(rho: float, capacity: int) -> float    # Eq. (3)
def expected_switch_delay(rho_or_load..., service_rate, capacity) -> float   # Eq. (1)
```

Numerical guards (unit-tested):
- ρ = 0 → P_b = 0, E[N] = 0, E[d] = 0
- |ρ − 1| < 1e-9 → E[N] = K/2 branch of Eq. (3)
- ρ > 1 still well-defined for Eq. (2)/(3) (geometric series ratio ρ^{K+1});
  delay formula remains finite because (1−P_b) shrinks as losses grow
- overflow-safe: powers computed as exp(K·log ρ), never raw ρ**K

### 5.2 `routing_sdn.py` - single-path routing + ATVM

```python
def route_flows(graph, weights, flows) -> list[FlowRoute]
    FlowRoute = (flow_id, src, dst, rate, node_sequence)

def aggregate_arrival_rates(graph, routes, service_rate, capacity,
                            refinement_passes=2) -> np.ndarray      # λ_n, with residual-rate feedback
def build_atvm(graph, routes, arrival_rates, service_rate) -> np.ndarray   # s_t, Eq. (9)-(10)
```

Interpretation choices (documented deviations):
- A flow's rate enters the queue of **every switch on its path including the
  origin** (the edge switch is a real processing point).
- Upstream loss factor of node i multiplies (1−P_b) of all path nodes
  **strictly before** i.
- λ_n ↔ P_b circularity within a path is resolved by forward pass in path
  order; across paths sharing switches, `refinement_passes` iterations of
  (aggregate → recompute P_b) - monotone and empirically converged at 2.

### 5.3 `reward.py`

```python
def delay_reward(mean_delay, worst_path_drain_time) -> float        # Eq. (12)
def loss_reward(total_lost_rate, total_offered_rate) -> float       # Eq. (13)
def combined_reward(r_d, r_p, alpha) -> float                       # Eq. (14)
```

Degenerate cases pinned by tests: zero traffic → both rewards = 1;
all-queues-full → r_d, r_p → their floors; outputs clamped to [0,1].

### 5.4 `ddpg.py`

```python
class Actor(nn.Module)      state_dim -> 400 -> 300 -> action_dim, ReLU hidden,
                            final layer scaled: w_min + (w_max-w_min)*(tanh(x)+1)/2
class Critic(nn.Module)     (state_dim + action_dim) -> 400 -> 300 -> 1
classOUNoise                theta=0.15, sigma=0.2, mu=0; reset() per episode
class ReplayBuffer          numpy-backed ring buffer, capacity 50k
class DDPGAgent
    act(state, noise=True)                    Alg. 1 line 9
    update(batch_size=100, gamma=0.99)        Eqs. (18)-(19), then soft update
    soft_update(net, target, tau=0.005)       Eqs. (20)-(21)
```

Defaults follow the paper where stated (hidden 400/300, batch 100, warmup 100
steps, γ=0.99, Adam); where the paper is silent or impractical, explicit
choices are logged in §Deviations (τ=0.005 following the standard DDPG
reference implementation the paper builds on; learning rate raised from the
paper's 1e-5 to 3e-4 because our compute budget is ~10³ iterations, not 10⁴+
- documented, not silent).

### 5.5 `environment.py` - the modeled network as an RL step

```python
class ModeledNetworkEnv
    __init__(graph, service_rate, buffer_capacity, alpha, w_min, w_max)
    reset(sequence, start_idx) -> state            bind a demand sequence
    step(weights) -> (next_state, reward, info)    Alg. 1 line 10
```

Step semantics (matches paper Sec. III-C.1):
1. demand dm_t is revealed;
2. ATVM state s_t computed under **previous** action's weights;
3. agent's new weights re-route all flows; delay/loss/reward R(s_t,a_t)
   evaluated on dm_t;
4. info dict carries mean_delay_ms, loss_fraction, throughput, plus **edge
   loads and U/U\*** (via imported LP cache) for the joint comparison -
   the SDN paper never reports U/U*, we compute it so both methods can be
   shown on one honest chart.

Temporal coupling: consecutive windows share the ATVM lineage (state inherits
last action's routing) exactly like the paper's iterations; episode ends at
sequence end (done=True), consistent with DDPG's off-policy bootstrapping.

### 5.6 `trainer.py`

Parallel in spirit to `train/trainer.py`, different internals:

```python
@dataclass DDPGConfig   (topology/service/alpha/w-range/net sizes/lr/tau/
                         buffer/batch/warmup/epochs/eval_every/seed/outdir)
class SdnDDPGTrainer
    __init__(graph, config)                 builds env, agent, shares traffic seeds
    train_epoch() -> dict                   Alg. 1 outer loop over windows of all
                                            7 train sequences; updates after warmup
    evaluate() -> dict                      deterministic policy (noise OFF) on the
                                            3 test sequences: reward, delay, loss,
                                            throughput, U/U*
    run() -> history                        saves metrics.json + checkpoint.pt
```

Same output contract as the PPO side (`metrics.json`, `checkpoint.pt`,
identical field naming where concepts coincide: `train_reward_mean`,
`test_agent_*`) so `eval/` can consume both without branching logic.

## 6. `eval/` Extension (additive, no rewrites)

New functions appended to the existing modules - existing signatures untouched:

```python
eval/evaluate.py
    load_metrics(path)                      works for both trainers' JSON (same schema)
    summarize_sdn_run(data)                 delay/loss/throughput/U-U* summary

eval/plot.py
    plot_five_method_comparison(curves_by_method, title, save_path)
        one figure per traffic regime: PPO curve, DDPG curve, three hlines
    plot_metric_panel(results, save_path)
        grouped bars: U/U* | mean delay | loss fraction - both learned methods
        + three classical baselines side by side
```

`main.py` gains a `--method {ppo,sdn-ddpg}` flag and a `--compare` mode that
loads both methods' metrics and emits the joint figures; existing CLI behavior
preserved when the flag is omitted.

## 7. Test Plan (per-module gates before anything depends on it)

| Gate file | Proves |
|---|---|
| `test_delay_model.py` | Eqs. (1)-(3) against a hand-worked toy switch (ρ=0.5, K=4: P_b=41/1272? - worked exactly in the test), ρ=0 and ρ=1 branches, monotonicity in ρ, overflow safety at K=10⁴ |
| `test_reward.py` | rewards ∈ [0,1]; zero-traffic → 1; α interpolation endpoints (α=1 pure delay, α=0 pure loss); pathological inputs clamp |
| `test_routing_sdn.py` | reconstructed paths ARE shortest (distance check vs `dijkstra`); ATVM mass conservation: Σ_j t_ij = λ_i minus downstream handoffs on a line graph; loss feedback reduces downstream rates |
| `test_ddpg.py` | actor output range = [w_min,w_max]; critic shape; soft-update arithmetic exact for τ=1 and τ=0; OU process mean-reverts, finite variance, reproducible after reset(); one gradient step reduces TD error on a synthetic bandit |
| `test_environment.py` | step returns finite reward/state; uniform weights reproduce hop-count routing; heavier demand ⇒ non-increasing r_p |
| `test_trainer.py` | mini-run (tiny net, few epochs) ends finite, writes artifacts, caches like the PPO trainer does |

## 8. Data Flow

```
graph/network.py ──────────────┐
traffic/generator.py ──────────┤
                               ▼
                  sdn_ddpg/routing_sdn.py ──► sdn_ddpg/delay_model.py
                          │                        │
                          ▼                        ▼
                  ATVM state  ──►  ddpg.DDPGAgent.act  ──► weights
                          ▲                                │
                          │        ┌───────────────────────┘
                          └────────┤
                                   ▼
                    environment.step → reward + info(U/U*, delay, loss)
                                   │
                                   ▼
                     trainer.py → metrics.json / checkpoint.pt
                                   │
        routing/optimal.py (read-only: U*) ──► eval/ five-method charts
```

---

*Gate: implementation starts only after approval of this document.*
