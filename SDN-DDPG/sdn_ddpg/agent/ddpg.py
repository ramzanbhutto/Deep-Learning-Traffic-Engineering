from collections import deque
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def hidden_block(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.ReLU(),
    )


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: Tuple[int, int] = (400, 300),
                 action_low: float = 1.0, action_high: float = 5.0):
        super().__init__()
        self.action_low = action_low
        self.action_high = action_high

        self.net = nn.Sequential(
            hidden_block(state_dim, hidden_dims[0]),
            hidden_block(hidden_dims[0], hidden_dims[1]),
            nn.Linear(hidden_dims[1], action_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        raw = self.net(state)
        mid = (self.action_high + self.action_low) / 2.0
        half_range = (self.action_high - self.action_low) / 2.0
        return mid + half_range * raw


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, int] = (400, 300)):
        super().__init__()
        self.net = nn.Sequential(
            hidden_block(state_dim + action_dim, hidden_dims[0]),
            hidden_block(hidden_dims[0], hidden_dims[1]),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([state, action], dim=-1)
        return self.net(combined).squeeze(-1)


class OUNoise:
    def __init__(self, action_dim: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2, seed: int | None = None):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)
        self._rng_state = self.rng.bit_generator.state
        self.reset()

    def reset(self):
        self.state = np.full(self.action_dim, self.mu, dtype=np.float64)
        self.rng.bit_generator.state = self._rng_state

    def sample(self) -> np.ndarray:
        drift = self.theta * (self.mu - self.state)
        diffusion = self.sigma * self.rng.standard_normal(self.action_dim)
        self.state = self.state + drift + diffusion
        return self.state.copy()


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int, seed: int | None = None):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = random.Random(seed)

    def add(self, state, action, reward, next_state, done):
        idx = self.position
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        indices = self.rng.sample(range(self.size), batch_size)
        return (
            torch.as_tensor(self.states[indices]),
            torch.as_tensor(self.actions[indices]),
            torch.as_tensor(self.rewards[indices]),
            torch.as_tensor(self.next_states[indices]),
            torch.as_tensor(self.dones[indices]),
        )

    def __len__(self):
        return self.size


class DDPGAgent:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 action_low: float = 1.0,
                 action_high: float = 5.0,
                 hidden_dims: Tuple[int, int] = (400, 300),
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 buffer_capacity: int = 50_000,
                 batch_size: int = 100,
                 warmup_steps: int = 100,
                 noise_sigma: float = 0.2,
                 noise_theta: float = 0.15,
                 seed: int | None = None,
                 use_twin_critics: bool = False,
                 target_policy_noise: float = 0.2,
                 target_noise_clip: float = 0.5):
        if seed is not None:
            torch.manual_seed(seed)

        self.actor = Actor(state_dim, action_dim, hidden_dims, action_low, action_high)
        self.critic = Critic(state_dim, action_dim, hidden_dims)

        self.actor_target = Actor(state_dim, action_dim, hidden_dims, action_low, action_high)
        self.critic_target = Critic(state_dim, action_dim, hidden_dims)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.use_twin_critics = use_twin_critics
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip

        if use_twin_critics:
            self.critic2 = Critic(state_dim, action_dim, hidden_dims)
            self.critic2_target = Critic(state_dim, action_dim, hidden_dims)
            self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        critic_params = list(self.critic.parameters())
        if use_twin_critics:
            critic_params += list(self.critic2.parameters())
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=critic_lr)

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.total_steps = 0

        self.noise = OUNoise(action_dim, theta=noise_theta, sigma=noise_sigma, seed=seed)
        self.buffer = ReplayBuffer(buffer_capacity, state_dim, action_dim, seed=seed)

        self.action_low = action_low
        self.action_high = action_high

    def act(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_t).squeeze(0).numpy()
        if explore and self.total_steps >= self.warmup_steps:
            action = action + self.noise.sample()
        elif explore:
            action = action + self.noise.sample()
        return np.clip(action, self.action_low, self.action_high)

    def observe(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)
        self.total_steps += 1

    def ready_to_update(self) -> bool:
        return len(self.buffer) >= max(self.batch_size, self.warmup_steps)

    def update(self) -> dict:
        states, actions, rewards, next_states, dones = \
            self.buffer.sample(self.batch_size)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            if self.use_twin_critics and self.target_policy_noise > 0:
                noise = (torch.randn_like(next_actions)
                         * self.target_policy_noise).clamp(
                    -self.target_noise_clip, self.target_noise_clip)
                next_actions = (next_actions + noise).clamp(
                    self.action_low, self.action_high)
            target_q = self.critic_target(next_states, next_actions)
            if self.use_twin_critics:
                target_q2 = self.critic2_target(next_states, next_actions)
                target_q = torch.min(target_q, target_q2)
            y = rewards + self.gamma * (1.0 - dones) * target_q

        current_q = self.critic(states, actions)
        critic_loss = nn.functional.mse_loss(current_q, y)
        if self.use_twin_critics:
            twin_loss = nn.functional.mse_loss(self.critic2(states, actions), y)
            critic_loss = critic_loss + twin_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        policy_loss_value = None
        if self.total_steps >= self.warmup_steps:
            predicted_actions = self.actor(states)
            policy_loss = -self.critic(states, predicted_actions).mean()

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.actor_optimizer.step()
            policy_loss_value = float(policy_loss.item())

        self._soft_update(self.critic, self.critic_target)
        if self.use_twin_critics:
            self._soft_update(self.critic2, self.critic2_target)
        self._soft_update(self.actor, self.actor_target)

        with torch.no_grad():
            q_abs_mean = float(
                self.critic(states, actions).abs().mean().item())
        return {"critic_loss": float(critic_loss.item()),
                "q_abs_mean": q_abs_mean,
                "policy_loss": policy_loss_value}

    def _soft_update(self, net: nn.Module, target: nn.Module):
        for param, target_param in zip(net.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - self.tau)
            target_param.data.add_(self.tau * param.data)

    def save(self, path: str):
        payload = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "use_twin_critics": self.use_twin_critics,
        }
        if self.use_twin_critics:
            payload["critic2"] = self.critic2.state_dict()
            payload["critic2_target"] = self.critic2_target.state_dict()
        torch.save(payload, path)

    def load(self, path: str):
        payload = torch.load(path)
        self.use_twin_critics = payload.get("use_twin_critics", False)
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.actor_target.load_state_dict(payload["actor_target"])
        self.critic_target.load_state_dict(payload["critic_target"])
        if self.use_twin_critics:
            self.critic2.load_state_dict(payload["critic2"])
            self.critic2_target.load_state_dict(payload["critic2_target"])
