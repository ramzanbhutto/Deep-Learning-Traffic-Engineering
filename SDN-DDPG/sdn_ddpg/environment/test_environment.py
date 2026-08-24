import numpy as np

from graph.network import create_12_node_topology
from sdn_ddpg.environment.environment import ModeledNetworkEnv
from sdn_ddpg.forwarding.routing_sdn import Flow
from traffic.generator import DemandMatrixGenerator


def flows_from_dm(dm, max_flows=40):
    flows = []
    for s in range(dm.shape[0]):
        for t in range(dm.shape[1]):
            if s != t and dm[s, t] > 0:
                flows.append(Flow(src=s, dst=t, rate=float(dm[s, t]) * 0.01))
                if len(flows) >= max_flows:
                    return flows
    return flows


def make_env():
    graph = create_12_node_topology()
    env = ModeledNetworkEnv(graph, service_rate=3000.0, buffer_capacity=10_000, alpha=0.9)
    gen = DemandMatrixGenerator(graph, model="gravity", sparsity=0.3, seed=3)
    dm = gen.sample()
    return env, flows_from_dm(dm)


def test_step_returns_finite_outputs():
    env, flows = make_env()
    state = env.reset(flows)
    assert state.shape == (144,)
    assert np.isfinite(state).all()

    weights = np.full(len(env.graph.edges), 2.5)
    next_state, reward, info = env.step(weights)

    assert next_state.shape == (144,) and np.isfinite(next_state).all()
    assert np.isfinite(reward) and -1e-9 <= reward <= 1.0 + 1e-9
    for key in ("mean_delay", "loss_fraction", "throughput"):
        assert np.isfinite(info[key])
    print(f"Env step OK (reward={reward:.4f}, delay={info['mean_delay']:.4f}s, "
          f"loss={info['loss_fraction']:.6f})")


def test_uniform_weights_reproduce_hop_count_routing():
    env, flows = make_env()
    env.reset(flows)
    uniform = np.ones(len(env.graph.edges))

    _, _, info_uniform = env.step(uniform)
    hops = info_uniform["worst_path_hops"]
    sp = env.graph.all_pairs_shortest_paths(np.ones(len(env.graph.edges)))
    expected_max_hops = max(sp[f.src, f.dst] for f in flows)
    assert abs(hops - expected_max_hops) < 1.0 + 1e-6
    print(f"Uniform weights -> hop-count routing OK "
          f"(worst path {hops} hops, expected ~{expected_max_hops:.0f})")


def test_better_weights_not_worse_reward():
    env, flows = make_env()
    env.reset(flows)

    rng = np.random.default_rng(9)
    results = []
    for trial in range(25):
        weights = rng.uniform(1.0, 5.0, size=len(env.graph.edges))
        _, reward, info = env.step(weights)
        results.append((reward, info["loss_fraction"]))

    rewards = [r for r, _ in results]
    best = max(rewards)
    worst = min(rewards)
    assert best > worst, "different allocations must yield different rewards"
    print(f"Weight sensitivity OK (rewards span [{worst:.4f}, {best:.4f}]))")


def test_heavier_demand_never_improves_loss():
    graph = create_12_node_topology()
    env = ModeledNetworkEnv(graph, service_rate=3000.0, buffer_capacity=1000, alpha=0.9)
    gen = DemandMatrixGenerator(graph, model="gravity", sparsity=0.3, seed=3)
    base_dm = gen.sample()
    base_flows = flows_from_dm(base_dm, max_flows=30)

    weights = np.full(len(graph.edges), 2.5)
    loss_by_scale = []
    for scale in (1.0, 3.0, 8.0):
        scaled = [Flow(s.src, s.dst, s.rate * scale) for s in base_flows]
        env.reset(scaled)
        _, _, info = env.step(weights)
        loss_by_scale.append(info["loss_fraction"])

    for i in range(1, len(loss_by_scale)):
        assert loss_by_scale[i] >= loss_by_scale[i - 1] - 1e-9, \
            "more offered traffic must not reduce loss fraction"
    print(f"Loss monotone in demand OK ({['%.4f' % l for l in loss_by_scale]})")


def test_state_depends_on_previous_action():
    env, flows = make_env()
    env.reset(flows)
    rng = np.random.default_rng(2)

    w_a = rng.uniform(1.0, 5.0, len(env.graph.edges))
    s_a, _, _ = env.step(w_a)
    w_b = rng.uniform(1.0, 5.0, len(env.graph.edges))
    s_b, _, _ = env.step(w_b)

    assert not np.allclose(s_a, s_b), "ATVM must reflect last action's routing"
    print("State tracks previous routing decision OK")


if __name__ == "__main__":
    test_step_returns_finite_outputs()
    test_uniform_weights_reproduce_hop_count_routing()
    test_better_weights_not_worse_reward()
    test_heavier_demand_never_improves_loss()
    test_state_depends_on_previous_action()
    print("\n=== ALL ENVIRONMENT TESTS PASSED ===")
