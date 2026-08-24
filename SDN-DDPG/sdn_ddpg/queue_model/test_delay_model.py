import numpy as np
from sdn_ddpg.queue_model.delay_model import (blocking_probability, expected_queue_occupation, expected_switch_delay, expected_lost_rate, network_queue_metrics)

def test_hand_computed_toy_switch():
    mu, lam, K = 1.0, 0.5, 4
    rho = lam / mu

    pb = blocking_probability(rho, K)
    assert abs(pb - 1 / 31) < 1e-12

    en = expected_queue_occupation(rho, K)
    assert abs(en - 26 / 31) < 1e-12

    d = expected_switch_delay(lam, mu, K)
    assert abs(d - 26 / 15) < 1e-10
    print(f"Hand-computed toy switch OK (Pb={pb:.6f}, E[N]={en:.6f}, "
          f"E[d]={d:.6f})")


def test_rho_one_branch():
    K = 4
    assert abs(blocking_probability(1.0, K) - K / (K + 1)) < 1e-9
    assert abs(expected_queue_occupation(1.0, K) - K / 2.0) < 1e-9
    print(f"rho=1 branch OK (Pb={blocking_probability(1.0, K):.4f} = "
          f"K/(K+1), E[N]=K/2)")


def test_zero_traffic():
    assert blocking_probability(0.0, 8) == 0.0
    assert expected_queue_occupation(0.0, 8) == 0.0
    assert expected_switch_delay(0.0, 3000.0, 10000) == 0.0
    assert expected_lost_rate(0.0, 3000.0, 10000) == 0.0
    print("Zero-traffic degenerate case OK")


def test_monotonicity_in_load():
    mus, lams, K = 3.0, np.linspace(0.3, 5.7, 20), 10000
    delays = [expected_switch_delay(lam, mus, K) for lam in lams]
    losses = [expected_lost_rate(lam, mus, K) for lam in lams]

    for i in range(1, len(delays)):
        assert delays[i] >= delays[i - 1] - 1e-9, "delay must be non-decreasing"
        assert losses[i] >= losses[i - 1] - 1e-9, "loss must be non-decreasing"
    print("Monotonicity in arrival rate OK")


def test_oversubscribed_switch_stays_finite():
    d = expected_switch_delay(6000.0, 3000.0, 10000)
    loss = expected_lost_rate(6000.0, 3000.0, 10000)
    assert np.isfinite(d) and d > 0
    assert 0 < loss <= 6000.0
    print(f"Oversubscribed switch OK (E[d]={d:.2f}s, lost={loss:.0f} pkt/s)")


def test_paper_scale_no_overflow():
    d = expected_switch_delay(2900.0, 3000.0, 10000)
    assert np.isfinite(d)
    rho_edge = blocking_probability(1.0000001, 10000)
    assert np.isfinite(rho_edge)
    print(f"Paper-scale parameters OK (near-saturation delay {d:.3f}s, "
          f"edge-case Pb finite)")


def test_network_aggregation():
    rates = np.array([500.0, 1500.0, 0.0])
    metrics = network_queue_metrics(rates, service_rate=3000.0, capacity=10000)

    assert set(metrics) >= {"mean_delay", "loss_fraction", "throughput"}
    assert metrics["mean_delay"] > 0
    assert 0 <= metrics["loss_fraction"] < 1
    assert abs(metrics["throughput"] -
               (rates.sum() - metrics["loss_fraction"] * rates.sum())) < 1e-6

    zero = network_queue_metrics(np.zeros(4), 3000.0, 10000)
    assert zero["mean_delay"] == 0.0 and zero["loss_fraction"] == 0.0
    print(f"Network aggregation OK (delay={metrics['mean_delay']:.4f}, "
          f"loss={metrics['loss_fraction']:.6f})")


if __name__ == "__main__":
    test_hand_computed_toy_switch()
    test_rho_one_branch()
    test_zero_traffic()
    test_monotonicity_in_load()
    test_oversubscribed_switch_stays_finite()
    test_paper_scale_no_overflow()
    test_network_aggregation()
    print("\n=== ALL DELAY MODEL TESTS PASSED ===")
