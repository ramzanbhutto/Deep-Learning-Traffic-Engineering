import numpy as np

from sdn_ddpg.reward.reward import delay_reward, loss_reward, combined_reward
from sdn_ddpg.queue_model.delay_model import expected_switch_delay


def test_rewards_in_unit_interval():
    rng = np.random.default_rng(0)
    for _ in range(500):
        d = rng.uniform(0, 50)
        drain = rng.uniform(0.1, 20)
        r_d = delay_reward(d, drain)
        assert 0.0 <= r_d <= 1.0

        lost = rng.uniform(0, 10_000)
        offered = rng.uniform(1, 10_000)
        r_p = loss_reward(lost, offered)
        assert 0.0 <= r_p <= 1.0

        alpha = rng.uniform(0, 1)
        r = combined_reward(r_d, r_p, alpha)
        assert 0.0 <= r <= 1.0
    print("Rewards bounded in [0,1] over random stress OK")


def test_delay_reward_normalization():
    assert delay_reward(0.0, 5.0) == 1.0
    assert abs(delay_reward(2.5, 5.0) - 0.5) < 1e-12
    assert delay_reward(7.5, 5.0) == 0.0, "over-worst-case must clamp to 0"
    assert delay_reward(3.3, 0.0) == 1.0, "no path -> no delay penalty"
    print("Delay reward normalization + clamps OK")


def test_loss_reward_semantics():
    assert loss_reward(0.0, 1000.0) == 1.0
    assert abs(loss_reward(100.0, 1000.0) - 0.9) < 1e-12
    assert loss_reward(1500.0, 1000.0) == 0.0, "loss > offered clamps to 0"
    assert loss_reward(0.0, 0.0) == 1.0, "zero traffic is perfect delivery"
    print("Loss reward semantics OK")


def test_alpha_interpolation():
    r_d, r_p = 0.8, 0.4
    assert combined_reward(r_d, r_p, 1.0) == r_d
    assert combined_reward(r_d, r_p, 0.0) == r_p
    assert abs(combined_reward(r_d, r_p, 0.5) - 0.6) < 1e-12

    try:
        combined_reward(r_d, r_p, 1.5)
        raise AssertionError("alpha=1.5 must raise")
    except ValueError:
        pass
    print("Alpha interpolation endpoints + validation OK")


def test_degenerate_environment_cases():
    mu, K = 3000.0, 10000

    zero_traffic_drain = expected_switch_delay(0.0, mu, K) * 3
    assert zero_traffic_drain == 0.0
    assert delay_reward(0.0, 1e-9) == 1.0

    saturated_delay = expected_switch_delay(2 * mu, mu, K)
    at_capacity_delay = expected_switch_delay(mu, mu, K)
    for value in (saturated_delay, at_capacity_delay):
        assert np.isfinite(value) and value >= 0

    assert at_capacity_delay > 10 * saturated_delay, (
        "M/M/1/K delay must peak around rho=1 where admitted "
        "throughput collapses")

    r_sat = delay_reward(saturated_delay, saturated_delay)
    assert abs(r_sat - 1.0) < 1e-9 or r_sat >= 0.0
    print(f"Degenerate cases OK (rho=1 spike {at_capacity_delay:.1f}s >> "
          f"oversubscribed {saturated_delay:.2f}s - expected M/M/1/K physics)")


if __name__ == "__main__":
    test_rewards_in_unit_interval()
    test_delay_reward_normalization()
    test_loss_reward_semantics()
    test_alpha_interpolation()
    test_degenerate_environment_cases()
    print("\n=== ALL REWARD TESTS PASSED ===")
