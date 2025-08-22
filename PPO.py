from H780_Train import  H780MuJoCoEnv
import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
from typing import Dict, Tuple, Any, Optional
import xml.etree.ElementTree as ET

# tiny_ppo.py
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.distributions import Normal

# simple_ppo_h780.py
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.distributions import Normal
from torch.distributions import MultivariateNormal

from H780_Train import H780MuJoCoEnv  # <-- uses your uploaded environment

# GPU detection
def get_device():
    """Automatically detect and return the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


# ----- Simple DNNs -----
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # diagonal std (log)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = torch.tanh(self.mean(x))          # keep actions in [-1, 1]
        std = torch.exp(self.log_std)            # (A,)
        return mean, std

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.value(x)                     # (B,1)

# ----- PPO -----
class PPO:
    def __init__(self, state_dim, action_dim, lr_actor=1e-4, lr_critic=3e-4, gamma=0.99,
                 lambda_gae=0.95, clip_ratio=0.2, epochs=4, batch_size=32,
                 max_timesteps_per_episode=1000, device=None):
        # Auto-detect device if not specified
        if device is None:
            device = get_device()
        
        self.device = device
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_timesteps_per_episode = max_timesteps_per_episode
        
        # Enable mixed precision training if using GPU
        if device.type == 'cuda':
            self.scaler = torch.amp.GradScaler('cuda')
            print("Enabled mixed precision training for GPU acceleration")
        else:
            self.scaler = None

    @torch.no_grad()
    def select_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,S)
        mean, std = self.actor(state)                 # mean: (1,A), std: (A,)
        cov = torch.diag(std**2)                      # (A,A)
        dist = MultivariateNormal(mean.squeeze(0), covariance_matrix=cov)
        action = dist.sample()                        # (A,)
        action = torch.clamp(action, -1.0, 1.0)       # keep in range
        log_prob = dist.log_prob(action)              # scalar
        return action.cpu().numpy(), float(log_prob.cpu())

    def compute_gae(self, rewards, values, next_value):
        """
        No 'done' flags: treat episode as a fixed-length trajectory and bootstrap
        from value of the final state.
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            v_t = values[t]
            v_next = next_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * v_next - v_t
            gae = delta + self.gamma * self.lambda_gae * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def update(self, trajectory):
        states = np.array(trajectory['states'], dtype=np.float32)
        actions = np.array(trajectory['actions'], dtype=np.float32)
        old_log_probs = np.array(trajectory['log_probs'], dtype=np.float32)
        rewards = np.array(trajectory['rewards'], dtype=np.float32)

        with torch.no_grad():
            v = self.critic(torch.as_tensor(states, device=self.device)).squeeze(-1).cpu().numpy()
            next_v = self.critic(torch.as_tensor(trajectory['next_state'], dtype=torch.float32, device=self.device).unsqueeze(0)).item()

        advantages, returns = self.compute_gae(rewards, v, next_v)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.as_tensor(states, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device)
        old_logp_t = torch.as_tensor(old_log_probs, device=self.device)
        adv_t = torch.as_tensor(advantages, device=self.device)
        ret_t = torch.as_tensor(returns, device=self.device)

        N = len(states)
        for _ in range(self.epochs):
            idx = np.arange(N)
            np.random.shuffle(idx)
            for start in range(0, N, self.batch_size):
                b = idx[start:start + self.batch_size]
                bs, ba = states_t[b], actions_t[b]
                bold = old_logp_t[b]
                badv, bret = adv_t[b], ret_t[b]

                # Use mixed precision training if available
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        mean, std = self.actor(bs)                       # mean: (B,A), std: (A,)
                        cov = torch.diag_embed(std**2)                   # (B,A,A)
                        dist = MultivariateNormal(mean, covariance_matrix=cov)
                        new_logp = dist.log_prob(ba)                     # (B,)
                        entropy = dist.entropy().mean()

                        ratio = torch.exp(new_logp - bold)
                        surr1 = ratio * badv
                        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * badv
                        actor_loss = -(torch.min(surr1, surr2).mean()) - 0.005 * entropy

                    self.optimizer_actor.zero_grad()
                    self.scaler.scale(actor_loss).backward()
                    self.scaler.unscale_(self.optimizer_actor)
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    self.scaler.step(self.optimizer_actor)
                    self.scaler.update()

                    with torch.amp.autocast('cuda'):
                        values = self.critic(bs).squeeze(-1)
                        values_clipped = torch.clamp(values, -10.0, 10.0)
                        critic_loss = nn.MSELoss()(values_clipped, bret)

                    self.optimizer_critic.zero_grad()
                    self.scaler.scale(critic_loss).backward()
                    self.scaler.unscale_(self.optimizer_critic)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.scaler.step(self.optimizer_critic)
                    self.scaler.update()
                else:
                    # Standard training for CPU
                    mean, std = self.actor(bs)                       # mean: (B,A), std: (A,)
                    cov = torch.diag_embed(std**2)                   # (B,A,A)
                    dist = MultivariateNormal(mean, covariance_matrix=cov)
                    new_logp = dist.log_prob(ba)                     # (B,)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_logp - bold)
                    surr1 = ratio * badv
                    surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * badv
                    actor_loss = -(torch.min(surr1, surr2).mean()) - 0.005 * entropy

                    self.optimizer_actor.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    self.optimizer_actor.step()

                    values = self.critic(bs).squeeze(-1)
                    values_clipped = torch.clamp(values, -10.0, 10.0)
                    critic_loss = nn.MSELoss()(values_clipped, bret)

                    self.optimizer_critic.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.optimizer_critic.step()

        # Clear GPU cache periodically to prevent memory buildup
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def save_model(self, actor_path="ppo_actor.pth", critic_path="ppo_critic.pth"):
        """Save both actor and critic models."""
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        print(f"Models saved to {actor_path} and {critic_path}")

    def load_model(self, actor_path="ppo_actor.pth", critic_path="ppo_critic.pth"):
        """Load both actor and critic models."""
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        print(f"Models loaded from {actor_path} and {critic_path}")

    def train(self, env, total_timesteps):
        timestep = 0
        episode_rewards = []
        while timestep < total_timesteps:
            # Reset: support envs returning either obs or (obs, info)
            reset_out = env.reset()
            state = reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out

            trajectory = {'states': [], 'actions': [], 'log_probs': [], 'rewards': []}
            episode_reward = 0.0

            for t in range(self.max_timesteps_per_episode):
                action, log_prob = self.select_action(state)

                # step: env returns (next_state, reward, terminated, truncated, info)
                step_result = env.step(action)
                if len(step_result) == 5:
                    next_state, reward, terminated, truncated, info = step_result
                else:
                    # Fallback for old format
                    next_state, reward = step_result
                    terminated, truncated = False, False

                trajectory['states'].append(state)
                trajectory['actions'].append(action)
                trajectory['log_probs'].append(log_prob)
                trajectory['rewards'].append(reward)

                state = next_state
                episode_reward += reward
                timestep += 1

                # Early termination
                if terminated or truncated:
                    break

            trajectory['next_state'] = state        # bootstrap from value(state) at horizon
            self.update(trajectory)

            episode_rewards.append(episode_reward)
            if len(episode_rewards) % 10 == 0:
                gpu_info = ""
                if self.device.type == 'cuda':
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3
                    gpu_info = f", GPU Memory: {gpu_memory:.2f}GB"
                
                print(f"Episode: {len(episode_rewards)}, "
                      f"Reward: {episode_reward:.2f}, "
                      f"Avg(10): {np.mean(episode_rewards[-10:]):.2f}, "
                      f"Timestep: {timestep}{gpu_info}")
    

# Usage
if __name__ == "__main__":
    env = H780MuJoCoEnv()
    ppo = PPO(
        state_dim=46, 
        action_dim=23,
        lr_actor=3e-4,      # Slightly higher learning rate
        lr_critic=1e-3,     # Higher critic learning rate
        gamma=0.99,
        lambda_gae=0.95,
        clip_ratio=0.2,
        epochs=100,
        batch_size=64,      # Larger batch size
        max_timesteps_per_episode=500,  # Shorter episodes
    )
    ppo.train(env, total_timesteps=1000000)
    ppo.save_model()  # Save both actor and critic models


