import numpy as np
from graph.network import NetworkGraph, create_12_node_topology


def test_graph_basic():
    g = create_12_node_topology()
    assert g.num_nodes == 12
    assert len(g.edges) == 34
    print(f"Nodes: {g.num_nodes}, Edges: {len(g.edges)}")

    for u in range(g.num_nodes):
        nbrs = g.neighbors(u)
        assert len(nbrs) > 0
        for v in nbrs:
            cap = g.capacity(u, v)
            assert cap > 0
            idx = g.edge_index(u, v)
            assert 0 <= idx < len(g.edges)
    print("Neighbors and capacities OK")


def test_dijkstra_simple():
    edges = [
        (0, 1, 10.0), (1, 0, 10.0),
        (0, 2, 10.0), (2, 0, 10.0),
        (1, 3, 10.0), (3, 1, 10.0),
        (2, 3, 10.0), (3, 2, 10.0),
    ]
    g = NetworkGraph(4, edges)
    weights = np.ones(len(edges), dtype=np.float32)

    dist = g.dijkstra(0, weights)
    expected = np.array([0.0, 1.0, 1.0, 2.0], dtype=np.float32)
    np.testing.assert_allclose(dist, expected, rtol=1e-5)
    print("Dijkstra simple graph OK")


def test_dijkstra_weighted():
    edges = [
        (0, 1, 10.0), (1, 0, 10.0),
        (0, 2, 10.0), (2, 0, 10.0),
        (1, 3, 10.0), (3, 1, 10.0),
        (2, 3, 10.0), (3, 2, 10.0),
    ]
    g = NetworkGraph(4, edges)
    weights = np.array([1.0, 1.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    dist = g.dijkstra(0, weights)
    assert dist[1] == 1.0
    assert dist[3] == 2.0
    assert dist[2] == 3.0, f"expected 0->1->3->2 cost 3, got {dist[2]}"
    print("Dijkstra weighted graph OK")


def test_all_pairs_shortest_paths():
    g = create_12_node_topology()
    weights = np.ones(len(g.edges), dtype=np.float32)
    sp = g.all_pairs_shortest_paths(weights)

    assert sp.shape == (12, 12)
    for i in range(12):
        assert sp[i, i] == 0.0
        for j in range(12):
            if i != j:
                assert sp[i, j] > 0
                assert sp[i, j] == sp[j, i]
    print("All-pairs shortest paths OK")


def test_12_node_topology_connectivity():
    g = create_12_node_topology()
    weights = np.ones(len(g.edges), dtype=np.float32)
    sp = g.all_pairs_shortest_paths(weights)

    for i in range(12):
        for j in range(12):
            assert sp[i, j] < np.inf, f"No path from {i} to {j}"
    print("12-node topology fully connected OK")


def test_edge_index_mapping():
    g = create_12_node_topology()
    num_edges = len(g.edges)
    for u, v, _ in g._edges:
        idx = g.edge_index(u, v)
        assert 0 <= idx < num_edges
        eu, ev = g.edges[idx]
        assert eu == u and ev == v
    print("Edge index mapping OK")


if __name__ == "__main__":
    test_graph_basic()
    test_dijkstra_simple()
    test_dijkstra_weighted()
    test_all_pairs_shortest_paths()
    test_12_node_topology_connectivity()
    test_edge_index_mapping()
    print("\n=== ALL GRAPH TESTS PASSED ===")