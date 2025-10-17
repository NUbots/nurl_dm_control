import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, x):
        return self.actor(x), self.critic(x)
    def act(self, x):
        logits = self.actor(x)
        dist = torch.distributions.Normal(logits, 1.0)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob
    def get_value(self, x):
        return self.critic(x)

def compute_gae(rewards, values, masks, gamma=0.99, lam=0.95):
    values = values + [0]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

class PPO:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, lam=0.95, clip_eps=0.2, epochs=10, batch_size=64):
        self.model = ActorCritic(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
    def update(self, obs, actions, log_probs, returns, advantages):
        obs = torch.tensor(obs, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        old_log_probs = torch.tensor(log_probs, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        for _ in range(self.epochs):
            idx = np.random.permutation(len(obs))
            for start in range(0, len(obs), self.batch_size):
                end = start + self.batch_size
                mb_idx = idx[start:end]
                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]
                logits, values = self.model(mb_obs)
                dist = torch.distributions.Normal(logits, 1.0)
                new_log_probs = dist.log_prob(mb_actions).sum(-1)
                ratio = (new_log_probs - mb_old_log_probs).exp()
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (mb_returns - values.squeeze(-1)).pow(2).mean()
                loss = actor_loss + 0.5 * critic_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()