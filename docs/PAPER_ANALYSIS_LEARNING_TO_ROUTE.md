# Paper Analysis: "Learning To Route" (Valadarsky et al., HotNets 2017)

## 1. Network Model

**Graph**: Directed capacitated graph G = (V, E, c)
- V: vertices (routers/switches)
- E: edges (links)
- c: E → ℝ⁺, capacity function assigning capacity c_e to each edge e ∈ E

**Routing Strategy R**: For each vertex v ∈ V and source-destination pair (s, t) ∈ V × V, a mapping
R_{v,(s,t)}: Γ(v) → [0, 1] where Γ(v) are v's neighbors.

Constraints:
- ∀s,t ∈ V, v ≠ t: Σ_{u∈Γ(v)} R_{v,(s,t)}(u) = 1 (traffic conserved at non-destinations)
- ∀s,t ∈ V: Σ_{u∈Γ(t)} R_{t,(s,t)}(u) = 0 (traffic absorbed at destination)

**Demand Matrix (DM)**: n × n matrix D where D_{i,j} specifies traffic demand from source i to destination j. n = |V|.

**Induced Flow**: A DM D and routing strategy R induce a multicommodity flow. Traffic from s to t splits at s per R_{s,(s,t)}, then at each intermediate v per R_{v,(s,t)}, etc.

**Objective (Classical TE)**: Minimize maximum link utilization (congestion):
```
U(f) = max_{e∈E} (f_e / c_e)
```
where f_e is total flow on edge e under flow f.

---

## 2. Reward / Objective Function

**Reward at epoch t**:
```
r^{(t)} = - U^{(t)} / OPT^{(t)}
```
- U^{(t)}: max link utilization achieved by routing strategy R^{(t)} for demand matrix D^{(t)}
- OPT^{(t)}: optimal (minimum possible) max link utilization for D^{(t)} (computed via LP)
- Negative sign: minimizing utilization = maximizing reward
- Ratio: normalizes by optimal achievable performance for that DM

**Discounted Return** (standard RL):
```
G_t = Σ_{k=0}^∞ γ^k r_{t+k}
```
γ = discount factor (paper uses TRPO which optimizes expected discounted reward)

---

## 3. Softmin Routing (Key Innovation)

Instead of learning splitting ratios directly (|V|²·|E| or |V|·|E| parameters), learn **per-edge weights** w = {w_e}_{e∈E} (only |E| outputs).

**From weights to splitting ratios**:
1. For each vertex u, destination d, neighbor v ∈ Γ(u), compute shortest-path distance from u to d via v under weights w:
   SP_w(u, v, d) = w_{u,v} + shortest path distance from v to d using w as edge lengths

2. Apply **softmin** with parameter γ_soft (γ=2 in paper):
   For vector α = (α_1, ..., α_r), softmin_γ(α)_i = e^{-γ·α_i} / Σ_j e^{-γ·α_j}
   
   The splitting ratio at u for destination d to neighbor v:
   ```
   R_{u,d}(v) = softmin_γ({SP_w(u, v', d)}_{v'∈Γ(u)})_v
   ```
   Higher γ → closer to shortest-path routing (hard min).

---

## 4. RL Formulation

| Component | Specification |
|-----------|---------------|
| **State s_t** | k-history of demand matrices: (D^{(t-k)}, ..., D^{(t-1)}) - k=10 in experiments |
| **Action a_t** | Per-edge weights w = {w_e}_{e∈E} ∈ ℝ^{|E|} |
| **Reward r_t** | -U^{(t)} / OPT^{(t)} (congestion ratio, negative) |
| **Policy π** | Neural network mapping state → action (3-layer FCN) |
| **Algorithm** | TRPO (Trust Region Policy Optimization) - continuous control |
| **Output processing** | Softmax (per-vertex) → splitting ratios → multicommodity flow → max utilization |

**Key insight**: Output dimension reduced from |V|²·|E| to |E| by learning weights + softmin.

---

## 5. Baselines (Section 5, Figure 2)

1. **Prev**: Optimize softmin routing w.r.t. **most recent DM only** (D^{(t-1)})
   - Solve for weights that minimize max utilization for D^{(t-1)}
   
2. **Avg_k**: Optimize softmin routing w.r.t. **k most recent DMs** (D^{(t-k)}, ..., D^{(t-1)})
   - Minimize max utilization over the average of these k DMs
   
3. **Oblivious**: **Optimal oblivious routing** [Azar et al., STOC 2003]
   - Fixed routing strategy independent of DM history
   - Optimized for worst-case over all possible DMs
   - Polynomial-time computable

---

## 6. Training Loop (Pseudocode - TRPO in Paper)

```python
# Paper's TRPO approach (we substitute DQN/PPO - see substitution rationale below)

initialize policy network π_θ (3-layer FCN, input: k×|V|×|V|, output: |E|)
initialize baseline network (for advantage estimation) - TRPO detail

for learning_epoch in 1..N:
    # Each learning epoch = full traversal of training DM sequences
    for each training DM sequence S of length L:
        # Generate 50 overlapping windows of k=10 consecutive DMs
        for window_start in 0..L-k-1:
            state = S[window_start : window_start+k]  # k history
            next_DM = S[window_start + k]
            
            # TRPO: collect batch of trajectories by running current policy
            action = π_θ(state)  # per-edge weights
            splitting_ratios = softmin_routing(action, γ=2)
            flow = compute_multicommodity_flow(splitting_ratios, next_DM)
            U = max_link_utilization(flow)
            OPT = solve_lp_optimal_utilization(next_DM)
            reward = -U / OPT
            
            # Store transition for TRPO update
            store(state, action, reward, next_state)
    
    # TRPO update step (constrained policy optimization)
    θ ← TRPO_update(θ, collected_trajectories)
    
    # Evaluate on test sequences periodically
```

**Paper's experimental setup**:
- Topology: 12 nodes, 32 edges (from Internet Topology Zoo [26])
- DMs: gravity model + bimodal model, sparsity p ∈ {0.3, 0.6, 0.9, 1.0}
- Training: 7 sequences × 60 DMs each; Test: 3 sequences
- k = 10 history length
- Optimal congestion via CPLEX LP solver

---

## 7. TRPO → DQN/PPO Substitution Rationale

| Aspect | Paper (TRPO) | Our Implementation (DQN/PPO) | Faithfulness |
|--------|--------------|------------------------------|--------------|
| **Action space** | Continuous (ℝ^{|E|}) | Continuous (PPO) or Discretized (DQN) | PPO: exact match; DQN: approximate |
| **Algorithm type** | Policy gradient (on-policy) | PPO (on-policy, clipped) / DQN (off-policy) | PPO: same paradigm |
| **Key mechanisms** | Trust region constraint, KL penalty | PPO: clipped surrogate objective; DQN: experience replay, target network | PPO: modern TRPO successor |
| **Output layer** | Linear (raw weights) | Same | Exact |
| **Reward signal** | -U/OPT (dense, per-step) | Same | Exact |
| **State representation** | k-history of DMs | Same | Exact |

**Why DQN/PPO instead of TRPO?**
1. **TRPO implementation complexity**: Requires conjugate gradient, line search, Hessian-vector products - fragile to implement from scratch
2. **PPO is TRPO's direct successor** (Schulman et al., 2017): Same trust-region idea via clipped objective, simpler, more stable, standard in PyTorch
3. **DQN reference papers (Mnih et al. 2013/2015)**: Provide experience replay + target network mechanics for off-policy learning; we can adapt to continuous actions via discretization or use PPO for true continuous control
4. **Paper's RL framing is algorithm-agnostic**: State/action/reward definition, softmin routing, baselines - all preserved. Only the policy optimizer changes.

**Decision**: Use **PPO** (Proximal Policy Optimization) for continuous action space - closest to TRPO's philosophy, well-supported in PyTorch, simpler than TRPO. DQN used only as reference for experience replay/target network concepts if we discretize.

---

## 8. DM Generation (Section 3.1)

**Gravity Model** (deterministic):
- D_{i,j} ∝ out_bandwidth_i × in_bandwidth_j / distance(i,j)²
- Captures "communication proportional to endpoint bandwidths"

**Bimodal Model** (probabilistic):
- Mixture of "mice" (small flows) and "elephants" (large flows)
- Percentage of elephant flows varied: 20%, 40%, 60%
- Captures heavy-tailed traffic distributions

**Sparsification**:
- Select p-fraction of communicating pairs uniformly at random
- Remove demands for other pairs
- p ∈ {0.3, 0.6, 0.9, 1.0}

**DM Sequences**:
- Class I: Cyclic (deterministic) - DM repeats every q epochs
- Class II: IID draws from distribution - no temporal correlation

---

## 9. Implementation Notes (this repo)

Implemented as designed above, with documented deviations collected in
`README.md` §Deviations - most notably: PPO replaces TRPO (§7 rationale),
Avg_k optimizes against the element-wise mean of the k recent DMs, and
Oblivious is the competitive-ratio LP restricted to a sampled scenario family.
Exact per-DM optimum U* comes from a HiGHS multicommodity-flow LP with
byte-keyed caching. Final reproduction numbers and Figure-2-style plots:
`results/` and `README.md` §Results.