import numpy as np
import torch

from agent.ppo import PPOAgent, RolloutBuffer, ActorCritic


def test_network_shapes():
    net = ActorCritic(state_dim=100, action_dim=34, hidden_dim=64)
    state = torch.randn(5, 100)
    dist, value = net(state)
    assert dist.mean.shape == (5, 34)
    assert value.shape == (5,)
    sample = dist.sample()
    assert sample.shape == (5, 34)
    log_prob = dist.log_prob(sample).sum(dim=-1)
    assert log_prob.shape == (5,)
    print("ActorCritic shapes OK")


def test_act_modes():
    agent = PPOAgent(state_dim=100, action_dim=34, hidden_dim=64, seed=0)
    state = np.random.randn(100).astype(np.float32)

    action, log_prob, value = agent.act(state)
    assert action.shape == (34,)
    assert np.isfinite(action).all()
    assert np.isfinite(log_prob) and np.isfinite(value)

    a1, _, _ = agent.act(state, deterministic=True)
    a2, _, _ = agent.act(state, deterministic=True)
    assert np.allclose(a1, a2), "deterministic mode must be repeatable"
    print("act() stochastic + deterministic modes OK")


def test_buffer_gae_known_values():
    buffer = RolloutBuffer()
    rewards = [1.0, 1.0, 1.0]
    values = [0.0, 0.0, 0.0]
    dones = [False, False, True]
    for r, v, d in zip(rewards, values, dones):
        buffer.add(np.zeros(4), np.zeros(2), 0.0, r, v, d)

    advantages, returns = buffer.compute_gae(last_value=0.0, gamma=0.9, gae_lambda=1.0)
    assert len(advantages) == 3 and len(returns) == 3

    assert abs(returns[2] - 1.0) < 1e-6
    expected_t1 = 1.0 + 0.9 * 1.0
    assert abs(returns[1] - expected_t1) < 1e-6
    expected_t0 = 1.0 + 0.9 * (1.0 + 0.9 * 1.0)
    assert abs(returns[0] - expected_t0) < 1e-6

    done_masked = advantages[2] - (rewards[2] + 0.9 * 0.0 - values[2])
    assert abs(done_masked) < 1e-6
    print(f"GAE returns OK ({returns.round(3)})")


def test_update_runs_and_improves_logprob():
    probe = PPOAgent(state_dim=8, action_dim=3, hidden_dim=32, seed=1)
    state = np.zeros(8, dtype=np.float32)
    with torch.no_grad():
        dist_init, _ = probe.policy(torch.as_tensor(state).unsqueeze(0))
    init_mean_sum = float(dist_init.mean.sum())

    agent2 = PPOAgent(state_dim=8, action_dim=3, hidden_dim=32, seed=1, update_epochs=10)

    buffer = RolloutBuffer()
    for _ in range(64):
        action, log_prob, value = agent2.act(state)
        reward = 1.0 if float(action.sum()) > 0 else -1.0
        buffer.add(state, action, log_prob, reward, value, True)

    stats = agent2.update(buffer)
    assert all(np.isfinite(v) for v in stats.values())
    assert len(buffer) == 0

    with torch.no_grad():
        dist_trained, _ = agent2.policy(torch.as_tensor(state).unsqueeze(0))
        trained_mean_sum = float(dist_trained.mean.sum())
        assert trained_mean_sum > init_mean_sum + 0.05, \
            f"policy mean should shift toward rewards ({init_mean_sum:.3f} -> {trained_mean_sum:.3f})"
    print(f"PPO update OK (policy_loss={stats['policy_loss']:.4f}, "
          f"value_loss={stats['value_loss']:.4f}, mean sum "
          f"{init_mean_sum:.3f} -> {trained_mean_sum:.3f})")


def test_buffer_clear():
    buffer = RolloutBuffer()
    for i in range(5):
        buffer.add(np.zeros(4), np.zeros(2), 0.0, float(i), 0.0, False)
    assert len(buffer) == 5
    buffer.clear()
    assert len(buffer) == 0
    print("Buffer clear OK")


if __name__ == "__main__":
    test_network_shapes()
    test_act_modes()
    test_buffer_gae_known_values()
    test_update_runs_and_improves_logprob()
    test_buffer_clear()
    print("\n=== ALL AGENT TESTS PASSED ===")
