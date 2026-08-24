# Paper Analysis: "Deep Reinforcement Learning-Based Routing on Software-Defined Networks"
### Kim, Kim & Lim - IEEE Access, vol. 10, 2022 (DOI 10.1109/ACCESS.2022.3151081)

*Second method implemented in the Deep Learning Traffic Engineering project.
Companion document to `docs/PAPER_ANALYSIS_LEARNING_TO_ROUTE.md`.*

---

## 1. Network Model

**Setting**: A backbone network of N SDN-enabled switches V = [v₁,…,v_N],
links E, G = (V, E). Edge switches are the departure/arrival points of the
AS. Each switch n has a **service rate μ_n** and a finite buffer/system
capacity **K_n** (paper: μ = 3000 pkt/s, K = 10 000 pkt for all switches).

**Traffic**: At step t there are M_t flows f_t^k (a flow = a source-destination
pair). Flow k generates Poisson arrivals at rate λ_t^k. Sources/destinations
are re-randomized and rates re-drawn every iteration (λ ∈ [10, 300] pkt/s).

**Forwarding**: **Single-path**, centralized SDN style. The controller computes,
for every flow, the **weighted shortest path** p*_{i,j} under link weights
w_m ∈ [w_min, w_max] (paper uses Dijkstra; weights ∈ [1, 5]) and installs
flow rules. *All* of a flow's volume rides one path - unlike Learning To
Route, where every node splits traffic multiplicatively via softmin.

**The modeling-based twist**: the DRL agent never touches the live network.
It trains against an **M/M/1/K queue-based model** of the network built from
SDN-controller information, using synthetic random demands. Exploration risk
on the data plane = 0, iterations are untethered from real time.

---

## 2. M/M/1/K Queue Model (the reward machinery)

Per switch n at time t, with aggregate arrival rate λ_n(t) and traffic
intensity ρ_n(t) = λ_n(t)/μ_n:

| Quantity | Formula | Paper Eq. |
|---|---|---|
| Blocking (loss) prob. | P_b^n = (1−ρ)(ρ^{K}) / (1−ρ^{K+1}) | (2) |
| Queue occupation | E[N_n] = ρ/(1−ρ) − ((K+1)ρ^{K+1})/(1−ρ^{K+1}) if ρ<1; K/2 if ρ=1 | (3) |
| Expected delay | E[d_n] = E[N_n] / (λ_n(1−P_b^n)) (Little's law, loss-adjusted) | (1) |
| End-to-end delay of flow k | D_e2e^k = Σ_{n ∈ path(p*)} E[d_n] | (4) |
| Average delay | D̄_e2e = (1/M_t) Σ_k D_e2e^k | (5) |
| Per-switch lost traffic | E[L_n] = λ_n · P_b^n | (6) |
| Total lost traffic | E[L_tot] = Σ_n λ_n P_b^n | (7) |

Propagation delay assumed negligible.

---

## 3. State Representation: ATVM

**ATVM** T = [t_{i,j}]^{N,N}: traffic volume arriving at switch i whose
next hop is j - i.e., load *aggregated onto links by the current routing*,
not source-destination pairs (that would be TDM, the paper's ablation).

From Eq. (9)-(10):

```
s_t^{i,j} = min(1, (1/µ_max) · Σ_k λ_t^k(i) · x_ij^k(t))
λ_t^k(i)   = λ_t^k · Π_{l ∈ path(src(k) → i)} (1 − P_b^l(t))
```

- x_ij^k(t): indicator that link (i,j) lies on flow k's current shortest path
- λ_t^k(i): flow k's *residual* arrival rate after upstream losses - so the
  state already encodes queueing feedback
- min(1, ·) normalizes state to [0, 1]; μ_max = max switch service rate
- s_t^{i,j} ↔ t_{i,j} of the ATVM scaled by μ_max

Key property: **ATVM depends on the action** (weights → paths → aggregation),
so the agent observes how its last decision shaped the network. TDM cannot.

---

## 4. Reward Function

Two sub-rewards, both in [0, 1]:

```
r_d(t) = 1 − D̄_e2e(t) / Σ_{n∈path(p_max)} K_n/µ_n          (12)  delay reward
r_p(t) = 1 − L_tot(t) / Σ_n λ_n(t)                          (13)  loss reward
```

- p_max = the maximum-hop path currently used - the denominator of r_d is the
  delay that path would suffer if every switch on it were saturated
  (K_n/μ_n = time to drain a full buffer), i.e., a worst-case normalizer.
- r_p = fraction of offered traffic actually delivered (throughput-like).

Combined reward (α = 0.9 in all paper experiments):

```
R(s_t, a_t) = α·r_d(t) + (1−α)·r_p(t)  ∈ [0, 1]             (14)
```

Objective: maximize expected discounted return R_t^π = R(s_t,a_t) +
Σ_i γ^i R(s_{t+i}, a_{t+i}), γ = 0.99 (Eqs. 15-17).

---

## 5. Action Space

```
a_t = a_t¹ × a_t² × … × a_t^{|E|},   a_t^m ∈ [w_min, w_max] = [1, 5]    (11)
```

Same output shape as Learning To Route's per-edge weight vector - but the
downstream semantics differ:

| | Learning To Route | This paper |
|---|---|---|
| Forwarding | softmin probabilistic splitting across neighbors | hard single shortest path per flow |
| Weight role | distances shaping split ratios | pure Dijkstra edge costs |
| Path diversity | inherent (multipath) | none (single path) |

---

## 6. DDPG Specifics (Sec. IV)

Off-policy, model-free actor-critic for continuous actions:

| Component | Detail | Eq. |
|---|---|---|
| Actor | μ(s \|θ^μ): state → action | - |
| Critic | Q(s,a\|θ^Q): (state, action) → value | (16)-(17) |
| Critic loss | L = (1/H)Σ (y_i − Q(s_i,a_i))², y_i = R_i + γ·Q′(s_{i+1}, μ′(s_{i+1})) | (18) |
| Policy gradient | ∇J ≈ (1/H)Σ ∇_a Q(s,a)\|_{a=μ(s_i)} ∇_θ μ(s_i) | (19) |
| Target soft update | θ^Q′ ← ε_c·θ^Q + (1−ε_c)·θ^Q′ ; same for actor with ε_a | (20)-(21) |
| Exploration | Ornstein-Uhlenbeck noise added to the deterministic action | Alg. 1 line 9 |
| Replay buffer B | stores (s, a, R, s′); off-policy reuse emphasized for pre-training | Alg. 1 line 11 |

Paper hyper-parameters: two hidden layers 400 & 300 units, ReLU, Adam,
lr = 1e-5 for both networks, γ = 0.99, batch H = 100, warmup = 100 steps,
OU noise for exploration. Implementation originally via stable-baselines3.

Training loop (Alg. 1): act = μ(s)+N on the *modeled* network → store →
sample H transitions → update critic, actor, then soft-update targets;
repeat until convergence; refresh controller-reported network info every T
cycles.

---

## 7. The Paper's Own Baselines vs Ours

| Paper's baseline | What it is | Relation to our existing baselines |
|---|---|---|
| **Hop-count (naive)** | uniform weights ⇒ fewest-hops Dijkstra routing | closest classical cousin: ≈ unit-weight shortest path; similar in spirit to what softmin approaches as γ→∞, but single-path (no ECMP splitting) |
| **DDPG-TDM** | same DDPG, but state = raw source-destination demand matrix instead of ATVM | an *ablation*, not a TE baseline - no analog among Prev/Avg_k/Oblivious |

Notably absent from the paper: adaptive re-optimization (our Prev/Avg_k) and
worst-case robust routing (our Oblivious). So the final comparison chart adds
value neither paper could produce alone: one learned method per reward model,
plus three classical TEs, on identical traffic.

---

## 8. Topologies & Experimental Details Worth Matching

| Topology | Nodes | Full-duplex links |
|---|---|---|
| 5×5 grid | 25 | 40 |
| GEANT | 40 | 61 |
| InternetMCI | 19 | 33 |

Fixed parameters across all: μ = 3000 pkt/s, K = 10 000 pkt, λ_k ~ U[10, 300]
pkt/s re-drawn per iteration, α = 0.9, w ∈ [1, 5].

Reported metrics: reward curve (500-step moving average); average end-to-end
delay, packet loss per second, throughput - vs iterations and vs number of
flows (100→150). Findings: ATVM-state beats TDM-state beats hop-count on
delay/loss/throughput; gains shrink when traffic is light (fewest-hop routing
is then near-optimal).

**Decision deferred to Phase 1** (per brief): reuse our 12-node/34-edge
topology for apples-to-apples comparison vs adding GEANT/InternetMCI. Leaning
toward reusing ours for the headline table and optionally adding InternetMCI
(19 nodes - smallest paper topology) as a scale-out sanity check.

---

## 9. Mapping Onto Our Existing Infrastructure (preview)

- `graph/network.py` supplies G=(V,E,c) unchanged (capacities unused by the
  M/M/1/K model - service rates/buffers are per-switch here).
- `traffic/generator.py` DMs map naturally to flow sets: entry (s,t,demand)
  becomes flow k with Poisson rate λ^k (packets/s) after a packets-per-unit
  convention is fixed.
- Shortest-path-per-flow under weights: `graph.dijkstra` reused directly.
- Aggregating per-switch λ_n from routed flows = lightweight bookkeeping on
  top of path enumeration.
- The softmin `routing/` stack stays untouched; the SDN evaluator is
  *single-path*, implemented inside the new package.

## 10. Deviations - final list

1. **Own DDPG implementation** instead of stable-baselines3 (repo constraint:
   minimal dependencies, self-contained code; algorithms follow paper Eqs.
   18-21 exactly: critic MSE with target bootstrapping, deterministic policy
   gradient through ∇ₐQ, soft target updates θ′←εθ+(1−ε)θ′, OU exploration,
   replay buffer).
2. **Topology**: reused this repo's 12-node/34-edge graph instead of GEANT /
   grid / InternetMCI - required for apples-to-apples comparison against
   Learning To Route on identical traffic streams.
3. **Traffic coupling**: paper re-randomizes λ and src/dst pairs every
   iteration; we draw flows from the shared `traffic/` sequences (incl.
   cyclic regimes) so both methods see identical demands.
4. **Deterministic modeled network**: given (weights, demands) the evaluator
   is noiseless; stochasticity enters only via the synthetic demand stream -
   consistent with the paper's model-based design philosophy.
5. **Learning rates**: paper's 1e-5 (both networks) raised to 3e-4 - their
   runs cover ≥10⁴ iterations, ours ~10⁴ steps total across 80 epochs;
   without the raise nothing moves within budget. Logged, not silent.
6. **Hidden sizes** 400/300 → 256/256 and update cadence every-2-steps with
   gradient clipping (max-norm 10): compute-budget parity choices, paper
   silent on update cadence/clipping.
7. **τ = 0.005**: paper gives no value for ε_c/ε_a beyond "small positive";
   we adopt the standard DDPG reference value.
8. **Flow-rate calibration**: paper fixes λ∈[10,300] pkt/s vs μ=3000; our
   demand-scaled flows are calibrated once per experiment so the hotspot
   switch sits at ρ ≈ 0.9 under mid-range weights (`target_hotspot_rho`),
   reproducing the paper's operating regime rather than an arbitrary one.
   Verified by `test_calibration_brings_hotspot_into_regime`.
9. **Common queue projection for comparisons**: delay/loss for ALL five
   methods (including PPO/Prev/Avg_k/Oblivious, which never had queueing
   semantics in their own papers) are computed by one shared post-hoc
   converter - per-switch offered rate = sum of outgoing edge loads fed to
   M/M/1/K - so the joint table applies identical measurement machinery to
   everyone. The SDN agent's richer path-aware internal delay is still
   logged separately in its own metrics.json.
10. **ATVM loss-feedback interpretation**: upstream loss factor multiplies
    (1−P_b) of path nodes strictly before the current hop; origin switch
    processes its own departing traffic; λ↔P_b circularity resolved by 2
    forward refinement passes (monotone fixed-point, empirically converged).

## 10b. Divergence Investigation and TD3 Extension (measured)

Plain single-critic DDPG (Eqs. 18-21, faithfully implemented) diverges on
gravity_cyclic - and everywhere else, at varying drift rates. Raw evidence:

- lr = 3e-4 (our default): critic MSE grows exponentially from 74 (epoch 24)
  to 1.87e9 (epoch 180); rewards bounded in [0,1] mean any correct Q must
  lie in [0,100], so this is pure value explosion. U/U* drifted
  monotonically 1.66 -> 1.84 across epochs 36-180 while delay and loss both
  IMPROVED - the reward does not see what congestion sees.
- lr = 1e-5 (paper value), full paper budget test: 714 epochs x
  280 transitions = 199,920 steps (~200,000 iterations of Figure 5).
  Critic loss still explodes: 0.22 -> 854,692 (3.8e6x growth). The policy
  barely moves (U/U* flat at ~1.68), so "stable" here also means
  "barely learning".
- Verdict: not our learning-rate deviation and not overfitting (train and
  test rewards improve together). It is the documented plain-DDPG
  single-critic bootstrap failure mode; the original authors either did not
  train into it or their exact configuration avoided it by chance.

Extension (explicitly NOT part of the reproduced method): twin critics with
target-action smoothing (TD3, Fujimoto et al. 2018) - y uses
min(Q1', Q2') over smoothed target actions. Enabled via `--td3`
(checkpoint dirs `<regime>_sdn_td3`); plain DDPG remains the default and
its artifacts stay untouched for comparison.

## 11. Additions beyond the source plan (self-directed)

- `sdn_ddpg/compare.py`: five-method comparison harness that reloads both
  trained checkpoints, rebuilds bit-identical test streams (asserted via
  allclose between both methods' regenerated sequences), re-evaluates
  Prev/Avg_k/Oblivious, projects every method through the same M/M/1/K
  measurement, and emits `comparison.json` + joint figures. Without this the
  promised five-column table would have mixed incompatible evaluation paths.
- `sdn_ddpg/routing_sdn.py` split out from the trainer (path reconstruction
  via destination-rooted reverse-graph Dijkstra + greedy descent, ATVM
  builder, rate aggregation) - the method's core deserves its own tests;
  reverse-graph trick means zero changes to the frozen `graph/` module.
- Rate calibration + its regression test (see deviation 8).
- Six standalone test files totalling ~40 assertions, mirroring the
  Learning-To-Route discipline: hand-computed toy switch values (P_b=1/31,
  E[N]=26/31, E[d]=26/15 at ρ=0.5,K=4), τ-arithmetic exactness, OU
  reproducibility + mean-reversion, ring-buffer eviction, path-cost equality
  vs Dijkstra over random weight draws.
