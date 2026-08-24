import numpy as np

from graph.network import NetworkGraph, create_12_node_topology
from sdn_ddpg.forwarding.routing_sdn import (Flow, route_flows, reconstruct_path, aggregate_arrival_rates, build_atvm)
from sdn_ddpg.queue_model.delay_model import blocking_probability


def make_line_graph():
    g = NetworkGraph(4, [(0, 1, 10.0), (1, 2, 10.0), (2, 3, 10.0)])
    weights = np.ones(3)
    flows = [Flow(src=0, dst=3, rate=100.0)]
    return g, weights, flows


def test_paths_are_true_shortest_paths():
    g = create_12_node_topology()
    rng = np.random.default_rng(5)

    for trial in range(10):
        weights = rng.uniform(1.0, 9.0, size=len(g.edges))
        flows = []
        for _ in range(15):
            s, t = rng.choice(12, size=2, replace=False)
            flows.append(Flow(int(s), int(t), float(rng.uniform(10, 300))))

        routes = route_flows(g, weights, flows)
        for flow, route in zip(flows, routes):
            assert route.nodes[0] == flow.src and route.nodes[-1] == flow.dst
            path_cost = sum(weights[g.edge_index(route.nodes[i], route.nodes[i + 1])]
                            for i in range(len(route.nodes) - 1))
            dist = g.dijkstra(flow.src, weights)[flow.dst]
            assert abs(path_cost - dist) < 1e-6, \
                f"reconstructed path cost {path_cost} != dijkstra {dist}"
            assert len(set(route.nodes)) == len(route.nodes), "no loops"
    print("Reconstructed paths match Dijkstra distances (10 trials x 15 flows) OK")


def test_atvm_mass_conservation_line_graph():
    g, weights, flows = make_line_graph()
    routes = route_flows(g, weights, flows)

    rates, loss_probs = aggregate_arrival_rates(g, routes, service_rate=1e9, capacity=10_000_000)
    assert loss_probs.max() < 1e-12, "huge service rate => no losses"

    atvm = build_atvm(g, routes, loss_probs, max_service_rate=1e9)
    assert atvm[0, 1] == 100.0 / 1e9
    assert atvm[1, 2] == 100.0 / 1e9
    assert atvm[2, 3] == 100.0 / 1e9

    for node in range(3):
        inflow = sum(atvm[u, node] for u in range(4)) if node > 0 else flows[0].rate / 1e9
        outflow = atvm[node].sum()
        assert abs(outflow - flows[0].rate / 1e9) < 1e-12
    print("ATVM mass conservation on line graph OK")


def test_loss_feedback_reduces_downstream_rates():
    g, weights, _ = make_line_graph()
    flows = [Flow(src=0, dst=3, rate=5000.0)]

    routes = route_flows(g, weights, flows)
    rates, loss_probs = aggregate_arrival_rates(g, routes, service_rate=5000.0, capacity=8)

    assert rates[0] >= rates[-1] - 1e-6, \
        "downstream switches must see no more than upstream after losses"
    assert loss_probs[0] > 0.01, "hot origin switch must experience blocking"
    print(f"Loss feedback OK (origin lambda={rates[0]:.0f} -> "
          f"sink lambda={rates[-1]:.0f}, Pb(origin)={loss_probs[0]:.4f})")


def test_atvm_clamped_to_unit_range():
    g, weights, _ = make_line_graph()
    flows = [Flow(src=0, dst=3, rate=50_000.0)]
    routes = route_flows(g, weights, flows)
    _, loss_probs = aggregate_arrival_rates(g, routes, service_rate=3000.0, capacity=10_000)
    atvm = build_atvm(g, routes, loss_probs, max_service_rate=3000.0)
    assert atvm.max() <= 1.0 + 1e-12 and atvm.min() >= 0.0
    print(f"ATVM normalized to [0,1] OK (max={atvm.max():.4f})")


def test_unreachable_flow_degrades_gracefully():
    g = NetworkGraph(4, [(0, 1, 10.0)])
    flows = [Flow(src=0, dst=3, rate=100.0)]
    weights = np.ones(1)
    routes = route_flows(g, weights, flows)
    assert routes[0].nodes == [0], "unreachable -> traffic stays at source"
    print("Unreachable destination handled gracefully OK")


def test_uniform_weights_give_hop_count_routing():
    g = create_12_node_topology()
    weights = np.ones(len(g.edges))
    sp = g.all_pairs_shortest_paths(weights)

    rng = np.random.default_rng(11)
    for _ in range(20):
        s, t = rng.choice(12, size=2, replace=False)
        s, t = int(s), int(t)
        route = route_flows(g, weights, [Flow(s, t, 1.0)])[0]
        assert len(route.nodes) - 2 <= sp[s, t] + 1e-6
        path_cost = sum(1 for i in range(len(route.nodes) - 1))
        assert abs(path_cost - sp[s, t]) < 1e-6
    print("Uniform weights reproduce hop-count routing OK")


if __name__ == "__main__":
    test_paths_are_true_shortest_paths()
    test_atvm_mass_conservation_line_graph()
    test_loss_feedback_reduces_downstream_rates()
    test_atvm_clamped_to_unit_range()
    test_unreachable_flow_degrades_gracefully()
    test_uniform_weights_give_hop_count_routing()
    print("\n=== ALL ROUTING_SDN TESTS PASSED ===")
