import numpy as np

from graph.network import NetworkGraph, create_12_node_topology
from routing.softmin import (softmin_splitting_ratios, compute_multicommodity_flow,
                             max_link_utilization, compute_shortest_path_distances)


def make_line_graph() -> NetworkGraph:
    edges = [(0, 1, 10.0), (1, 2, 10.0), (2, 3, 10.0)]
    return NetworkGraph(4, edges)


def make_diamond_graph() -> NetworkGraph:
    edges = [
        (0, 1, 100.0), (0, 2, 100.0),
        (1, 3, 100.0), (2, 3, 100.0),
    ]
    return NetworkGraph(4, edges)


def test_ratios_sum_to_one():
    g = create_12_node_topology()
    rng = np.random.default_rng(0)
    weights = rng.uniform(1.0, 5.0, size=len(g.edges)).astype(np.float32)

    ratios = softmin_splitting_ratios(g, weights, gamma=2.0)
    assert ratios.shape == (12, 12, 12)

    for u in range(12):
        nbrs = set(g.neighbors(u))
        for d in range(12):
            if d == u:
                assert np.allclose(ratios[u, :, d], 0.0), "destination must absorb"
                continue
            total = ratios[u, :, d].sum()
            assert abs(total - 1.0) < 1e-4, f"row sum {total} at u={u}, d={d}"
            for v in range(12):
                if v not in nbrs:
                    assert ratios[u, v, d] == 0.0, f"non-neighbor {v} got flow"
                else:
                    assert ratios[u, v, d] >= 0.0
    print("Softmin ratios sum to 1 over neighbors OK")


def test_softmin_prefers_shortest_paths():
    g = create_12_node_topology()
    rng = np.random.default_rng(1)
    weights = rng.uniform(1.0, 10.0, size=len(g.edges)).astype(np.float32)

    ratios = softmin_splitting_ratios(g, weights, gamma=500.0)

    sp = compute_shortest_path_distances(g, weights)
    checked = 0
    for u in range(12):
        nbrs = g.neighbors(u)
        for d in range(12):
            if d == u:
                continue
            dists = [sp[v, d] + weights[g.edge_index(u, v)] for v in nbrs]
            best = min(dists)
            best_nbrs = {v for v, dv in zip(nbrs, dists) if abs(dv - best) < 1e-4}
            shares = {v: ratios[u, v, d] for v in nbrs}
            argmax_nbr = max(shares, key=shares.get)
            assert argmax_nbr in best_nbrs, \
                f"gamma=500 argmax should be a shortest hop at u={u}, d={d}"
            for v in nbrs:
                if v not in best_nbrs:
                    assert shares[v] < 0.02, \
                        f"gamma=500 should avoid non-shortest hop {u}->{v}->{d}"
            checked += 1
    print(f"High-gamma approaches shortest-path routing OK ({checked} pairs checked)")


def test_line_graph_exact_flow():
    g = make_line_graph()
    weights = np.ones(3, dtype=np.float32)
    ratios = softmin_splitting_ratios(g, weights, gamma=100.0)

    dm = np.zeros((4, 4))
    dm[0, 3] = 10.0

    edge_flow, delivered = compute_multicommodity_flow(g, ratios, dm)

    assert delivered > 0.999, f"line graph should deliver everything, got {delivered}"
    np.testing.assert_allclose(edge_flow, [10.0, 10.0, 10.0], rtol=1e-3)
    u = max_link_utilization(g, edge_flow)
    assert abs(u - 1.0) < 1e-3
    print("Line graph exact propagation OK")


def test_diamond_symmetric_split():
    g = make_diamond_graph()
    weights = np.ones(4, dtype=np.float32)
    ratios = softmin_splitting_ratios(g, weights, gamma=2.0)

    r01 = ratios[0, 1, 3]
    r02 = ratios[0, 2, 3]
    assert abs(r01 - 0.5) < 1e-4 and abs(r02 - 0.5) < 1e-4

    dm = np.zeros((4, 4))
    dm[0, 3] = 10.0
    edge_flow, delivered = compute_multicommodity_flow(g, ratios, dm)

    assert delivered > 0.999
    np.testing.assert_allclose(edge_flow, [5.0, 5.0, 5.0, 5.0], rtol=1e-3)
    print("Diamond symmetric 50/50 split OK")


def test_softmin_tilts_split_by_weights():
    g = make_diamond_graph()
    edge_idx_01 = g.edge_index(0, 1)
    edge_idx_02 = g.edge_index(0, 2)
    weights = np.ones(4, dtype=np.float32)
    weights[edge_idx_02] = 2.0

    ratios = softmin_splitting_ratios(g, weights, gamma=2.0)
    r01 = ratios[0, 1, 3]

    sp_via_1 = weights[g.edge_index(0, 1)] + weights[g.edge_index(1, 3)]
    sp_via_2 = weights[g.edge_index(0, 2)] + weights[g.edge_index(2, 3)]
    expected = np.exp(-2 * sp_via_1) / (np.exp(-2 * sp_via_1) + np.exp(-2 * sp_via_2))
    assert abs(r01 - expected) < 1e-4, f"expected {expected:.4f}, got {r01:.4f}"

    dm = np.zeros((4, 4))
    dm[0, 3] = 10.0
    edge_flow, _ = compute_multicommodity_flow(g, ratios, dm)
    assert edge_flow[edge_idx_01] > edge_flow[edge_idx_02]
    print(f"Weighted split tilts correctly (via 0->1: {edge_flow[edge_idx_01]:.3f})")


def test_utilization_multiple_commodities():
    g = make_diamond_graph()
    weights = np.ones(4, dtype=np.float32) * 100
    ratios = softmin_splitting_ratios(g, weights, gamma=100.0)

    dm = np.zeros((4, 4))
    dm[0, 3] = 60.0
    dm[1, 3] = 30.0

    edge_flow, _ = compute_multicommodity_flow(g, ratios, dm)
    u = max_link_utilization(g, edge_flow)
    assert abs(u - 0.6) < 1e-3, f"edge (1,3) should carry 60/100, got U={u}"
    np.testing.assert_allclose(edge_flow, [30.0, 30.0, 60.0, 30.0], rtol=1e-3)
    print(f"Multi-commodity utilization OK (U={u:.3f})")


def test_unreachable_destination_guard():
    edges = [(0, 1, 10.0), (1, 0, 10.0)]
    g = NetworkGraph(4, edges)
    weights = np.ones(2, dtype=np.float32)
    ratios = softmin_splitting_ratios(g, weights, gamma=2.0)

    assert np.isfinite(ratios).all(), "no NaN/inf even when destination unreachable"
    dm = np.zeros((4, 4))
    dm[0, 3] = 5.0
    edge_flow, delivered = compute_multicommodity_flow(g, ratios, dm)
    assert delivered < 0.01, "traffic to unreachable node must not be delivered"
    print("Unreachable destination handled gracefully OK")


if __name__ == "__main__":
    test_ratios_sum_to_one()
    test_softmin_prefers_shortest_paths()
    test_line_graph_exact_flow()
    test_diamond_symmetric_split()
    test_softmin_tilts_split_by_weights()
    test_utilization_multiple_commodities()
    test_unreachable_destination_guard()
    print("\n=== ALL ROUTING TESTS PASSED ===")
