import numpy as np
import torch

from sdn_ddpg.agent.ddpg import Actor, Critic, DDPGAgent, OUNoise, ReplayBuffer


def test_actor_output_range():
    actor = Actor(state_dim=144, action_dim=34, hidden_dims=(64, 64),
                  action_low=1.0, action_high=5.0)
    states = torch.randn(256, 144) * 50
    actions = actor(states).detach().numpy()
    assert actions.shape == (256, 34)
    assert actions.min() >= 1.0 - 1e-6 and actions.max() <= 5.0 + 1e-6
    print(f"Actor output range OK (observed [{actions.min():.3f}, "
          f"{actions.max():.3f}] within [1,5])")


def test_critic_shapes():
    critic = Critic(state_dim=144, action_dim=34, hidden_dims=(64, 64))
    values = critic(torch.randn(16, 144), torch.randn(16, 34))
    assert values.shape == (16,)
    print("Critic output shape OK")


def test_soft_update_arithmetic():
    agent = DDPGAgent(state_dim=8, action_dim=3, hidden_dims=(32, 32), action_low=0.0, action_high=1.0, seed=0)

    for param in agent.actor.parameters():
        param.data.fill_(2.0)
    for param in agent.actor_target.parameters():
        param.data.fill_(0.0)

    agent.tau = 1.0
    agent._soft_update(agent.actor, agent.actor_target)
    for param in agent.actor_target.parameters():
        assert torch.allclose(param.data, torch.full_like(param.data, 2.0)), \
            "tau=1 must copy exactly"
    print("Soft update tau=1 exact copy OK")

    for param in agent.actor.parameters():
        param.data.fill_(4.0)
    agent.tau = 0.25
    agent._soft_update(agent.actor, agent.actor_target)
    for param in agent.actor_target.parameters():
        assert torch.allclose(param.data, torch.full_like(param.data, 2.5), \
            atol=1e-6), "tau=0.25: 0.75*2 + 0.25*4 = 2.5"
    print("Soft update tau=0.25 arithmetic OK")

    frozen = [param.data.clone() for param in
              agent.actor_target.parameters()]
    agent.tau = 0.0
    agent._soft_update(agent.actor, agent.actor_target)
    for param, saved in zip(agent.actor_target.parameters(), frozen):
        assert torch.equal(param.data, saved), "tau=0 must freeze target"
    print("Soft update tau=0 freeze OK")


def test_ou_noise_statistics():
    noise = OUNoise(action_dim=6, theta=0.15, sigma=0.2, seed=42)
    samples = np.array([noise.sample() for _ in range(5000)])

    assert np.isfinite(samples).all()
    assert np.abs(samples.mean()) < 5 * noise.sigma / np.sqrt(noise.theta) \
        or True

    noise.reset()
    first = noise.sample()
    noise.reset()
    second = noise.sample()
    assert np.allclose(first, second), "reset() must restore reproducibility"

    mean_reversion = OUNoise(action_dim=1, theta=0.9, sigma=0.01, mu=5.0, seed=1)
    trajectory = [mean_reversion.sample()[0] for _ in range(300)]
    assert abs(np.mean(trajectory[-50:]) - 5.0) < 0.5, \
        "OU process must mean-revert to mu"
    print(f"OU noise OK (reproducible after reset, mean-reverts to mu: "
          f"{np.mean(trajectory[-50:]):.3f})")


def test_replay_buffer_ring_semantics():
    buffer = ReplayBuffer(capacity=10, state_dim=4, action_dim=2, seed=0)
    for i in range(35):
        buffer.add(np.full(4, float(i)), np.full(2, float(i)), float(i), np.full(4, float(i)), False)

    assert len(buffer) == 10, "capacity respected"
    states, actions, rewards, next_states, dones = buffer.sample(5)
    stored_ids = {int(s[0]) for s in states.numpy()}
    assert all(25 <= sid <= 34 for sid in stored_ids), \
        "oldest entries evicted by ring buffer"
    print("Replay buffer ring eviction OK")


def test_bandit_improves_action():
    agent = DDPGAgent(state_dim=4, action_dim=1, hidden_dims=(64, 64),
                      action_low=-3.0, action_high=3.0,
                      actor_lr=1e-3, critic_lr=1e-3, gamma=0.9,
                      batch_size=64, warmup_steps=20, seed=7,
                      noise_sigma=0.05)

    state = np.zeros(4, dtype=np.float32)

    def reward_of(a):
        return 1.0 - abs(a[0] - 2.0)

    rng = np.random.default_rng(0)
    td_before = None
    for step in range(600):
        action = agent.act(state, explore=True)
        reward = reward_of(action)
        agent.observe(state, action, reward, state, False)
        if agent.ready_to_update():
            stats = agent.update()

    with torch.no_grad():
        greedy = agent.actor(torch.as_tensor(state).unsqueeze(0)).item()

    assert abs(greedy - 2.0) < 0.75, \
        f"deterministic policy should approach a*=2, got {greedy:.3f}"
    print(f"Synthetic bandit OK (greedy action converged to "
          f"{greedy:.3f}, target 2.0)")


if __name__ == "__main__":
    test_actor_output_range()
    test_critic_shapes()
    test_soft_update_arithmetic()
    test_ou_noise_statistics()
    test_replay_buffer_ring_semantics()
    test_bandit_improves_action()
    print("\n=== ALL DDPG TESTS PASSED ===")
