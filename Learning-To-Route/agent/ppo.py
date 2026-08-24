from typing import Dict, Iterator, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, state: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        features = self.body(state)
        mu = self.mu_head(features)
        std = self.log_std.clamp(-3.0, 1.0).exp().expand_as(mu)
        return Normal(mu, std), self.value_head(features).squeeze(-1)


class RolloutBuffer:
    def __init__(self):
        self.states, self.actions, self.log_probs = [], [], []
        self.rewards, self.values, self.dones = [], [], []

    def add(self, state, action, log_prob, reward, value, done):
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(bool(done))

    def __len__(self):
        return len(self.states)

    def compute_gae(self, last_value: float, gamma: float,
                    gae_lambda: float) -> Tuple[np.ndarray, np.ndarray]:
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones)

        advantages = np.zeros(len(rewards))
        gae = 0.0
        for t in reversed(range(len(rewards))):
            mask = 0.0 if dones[t] else 1.0
            delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[t] = gae
        returns = advantages + np.array(self.values)
        return advantages.astype(np.float32), returns.astype(np.float32)

    def clear(self):
        self.__init__()


class PPOAgent:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 128,
                 lr: float = 3e-4,
                 gamma: float = 0.95,
                 gae_lambda: float = 0.95,
                 clip_eps: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 update_epochs: int = 4,
                 batch_size: int = 64,
                 max_grad_norm: float = 0.5,
                 seed: int | None = None):
        if seed is not None:
            torch.manual_seed(seed)
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

    def act(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist, value = self.policy(state_t)
            if deterministic:
                action = dist.mean
                log_prob = dist.log_prob(action).sum(dim=-1)
            else:
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
        return (action.squeeze(0).numpy(),
                float(log_prob.item()),
                float(value.item()))

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, values = self.policy(states)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, values, entropy

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        last_value = buffer.values[-1]
        advantages, returns = buffer.compute_gae(last_value, self.gamma, self.gae_lambda)

        states = torch.as_tensor(np.stack(buffer.states))
        actions = torch.as_tensor(np.stack(buffer.actions))
        old_log_probs = torch.as_tensor(np.array(buffer.log_probs))
        advantages_t = torch.as_tensor(advantages)
        returns_t = torch.as_tensor(returns)

        advantages_t = (advantages_t - advantages_t.mean()) / \
            (advantages_t.std() + 1e-8)

        num_samples = len(buffer)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        update_count = 0

        for _ in range(self.update_epochs):
            perm = torch.randperm(num_samples)
            for start in range(0, num_samples, self.batch_size):
                idx = perm[start:start + self.batch_size]
                log_probs, values, entropy = self.evaluate_actions(states[idx], actions[idx])

                ratio = (log_probs - old_log_probs[idx]).exp()
                surr1 = ratio * advantages_t[idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (returns_t[idx] - values).pow(2).mean()
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + \
                    self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                for key, val in (("policy_loss", policy_loss),
                                 ("value_loss", value_loss),
                                 ("entropy", entropy.mean())):
                    stats[key] += float(val.item())
                update_count += 1

        buffer.clear()
        return {key: val / max(update_count, 1) for key, val in stats.items()}
