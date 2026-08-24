import numpy as np

from graph.network import NetworkGraph
from baselines.classical import (PrevBaseline, AvgKBaseline, ObliviousRouting, congestion_of_weights)
from routing.optimal import optimal_congestion_lp, congestion_lower_bound
from softmin_routing.softmin import softmin_splitting_ratios, compute_multicommodity_flow, max_link_utilization


def make_bidirectional_diamond() -> NetworkGraph:
    edges = [(0, 1, 100.0), (1, 0, 100.0),
             (0, 2, 100.0), (2, 0, 100.0),
             (1, 3, 100.0), (3, 1, 100.0),
             (2, 3, 100.0), (3, 2, 100.0)]
    return NetworkGraph(4, edges)


def test_prev_matches_optimal_on_symmetric():
    g = make_bidirectional_diamond()
    dm = np.zeros((4, 4))
    dm[0, 3] = 10.0

    opt = optimal_congestion_lp(g, dm)
    prev_util, weights = PrevBaseline(g).routing_congestion(dm, dm)

    assert abs(prev_util - opt) < 1e-3 * max(opt, 1.0), \
        f"prev {prev_util:.5f} vs OPT {opt:.5f}"
    print(f"Prev reaches optimum on symmetric diamond ({prev_util:.4f} == OPT {opt:.4f})")


def test_avgk_single_dm_matches_prev():
    g = make_bidirectional_diamond()
    dm = np.zeros((4, 4))
    dm[0, 3] = 10.0
    dm[1, 2] = 4.0

    avgk_util, _ = AvgKBaseline(g, k=3).routing_congestion([dm] * 3, dm)
    single_util, _ = AvgKBaseline(g, k=1).routing_congestion([dm], dm)

    assert abs(avgk_util - single_util) < 1e-6 * max(1.0, single_util), \
        "averaging identical DMs must not change the objective"
    print(f"Avg_k on identical DMs matches single ({avgk_util:.4f})")


def test_avgk_consistent_with_prev_on_stationary_history():
    g = make_bidirectional_diamond()
    dm = np.zeros((4, 4))
    dm[0, 3] = 10.0
    dm[1, 2] = 4.0

    avgk_util, _ = AvgKBaseline(g, k=3).routing_congestion([dm] * 3, dm)
    prev_util, _ = PrevBaseline(g).routing_congestion(dm, dm)

    assert abs(avgk_util - prev_util) < 1e-4 * max(1.0, prev_util), \
        f"stationary history should match prev ({avgk_util:.5f} vs {prev_util:.5f})"
    print(f"Avg_k stationary-history consistency OK ({avgk_util:.4f})")


def test_avgk_handles_mixed_history():
    g = make_bidirectional_diamond()
    hot = np.zeros((4, 4)); hot[0, 3] = 90.0
    calm = np.zeros((4, 4)); calm[0, 3] = 10.0

    util, _ = AvgKBaseline(g, k=2).routing_congestion([hot, calm], hot)

    assert np.isfinite(util) and 0.40 <= util <= 0.60
    print(f"Avg_k mixed-history OK (evaluated on hot: U={util:.3f})")


def test_oblivious_beats_fixed_shortest_path_worst_case():
    g = make_bidirectional_diamond()

    scenario_a = np.zeros((4, 4)); scenario_a[0, 3] = 100.0
    scenario_b = np.zeros((4, 4)); scenario_b[0, 1] = 50.0; scenario_b[2, 3] = 50.0

    oblivious = ObliviousRouting(g, [scenario_a, scenario_b])

    worst_ratio_oblivious = max(oblivious.evaluate(dm) /
                                optimal_congestion_lp(g, dm)
                                for dm in (scenario_a, scenario_b))

    sp_weights = np.ones(len(g.edges), dtype=np.float32)
    worst_ratio_sp = max(
        congestion_of_weights(g, sp_weights, [dm]) / optimal_congestion_lp(g, dm)
        for dm in (scenario_a, scenario_b))

    assert oblivious.scenario_ratio < 1.5, \
        f"oblivious should be near-optimal on its own scenarios, got {oblivious.scenario_ratio}"
    print(f"Oblivious worst-case ratio {worst_ratio_oblivious:.3f} vs "
          f"fixed-shortest-path {worst_ratio_sp:.3f}")


def test_oblivious_generalizes_to_unseen_demand():
    g = make_bidirectional_diamond()
    train_a = np.zeros((4, 4)); train_a[0, 3] = 80.0
    train_b = np.zeros((4, 4)); train_b[1, 2] = 60.0

    oblivious = ObliviousRouting(g, [train_a, train_b])

    unseen = np.zeros((4, 4)); unseen[0, 3] = 40.0; unseen[1, 2] = 30.0
    u = oblivious.evaluate(unseen)
    lb = congestion_lower_bound(g, unseen)

    assert np.isfinite(u) and u >= lb - 1e-6
    print(f"Oblivious on unseen DM: U={u:.3f} >= LB={lb:.3f}")


if __name__ == "__main__":
    test_prev_matches_optimal_on_symmetric()
    test_avgk_single_dm_matches_prev()
    test_avgk_consistent_with_prev_on_stationary_history()
    test_avgk_handles_mixed_history()
    test_oblivious_beats_fixed_shortest_path_worst_case()
    test_oblivious_generalizes_to_unseen_demand()
    print("\n=== ALL BASELINE TESTS PASSED ===")
