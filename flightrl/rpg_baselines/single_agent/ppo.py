import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import MultivariateNormal
import numpy as np
import os
import gym
from typing import NamedTuple

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter



class RolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class RolloutBuffer:
    def __init__(self, device, buffer_size, n_envs, obs_dim, action_dim, gamma, gae_lambda):
        self.device = device
        self.rollout_buffer_size = buffer_size
        self.n_envs = n_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reset()

    def reset(self):
        self.observations = np.zeros((self.rollout_buffer_size, self.n_envs, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.rollout_buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.rollout_buffer_size, self.n_envs), dtype=np.float32)
        self.is_terminals = np.zeros((self.rollout_buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.rollout_buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.rollout_buffer_size, self.n_envs), dtype=np.float32)

        # Update after compute_returns_and_advantage function
        self.returns = np.zeros((self.rollout_buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.rollout_buffer_size, self.n_envs), dtype=np.float32)

        self.pos = 0
        self.full = False
        self.generator_ready = False

    def add(self, obs, action, reward, done, value, log_prob):
        if len(log_prob.shape) == 0:
            log_prob = log_prob.reshape(-1, 1) # Reshape 0-d tensor to avoid error

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.is_terminals[self.pos] = np.array(done)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.rollout_buffer_size:
            self.full = True

    def compute_returns_and_advantage(self, last_values, dones):
        with torch.no_grad():
            last_values = last_values.clone().cpu().numpy().flatten()
            last_gae_lam = 0

            for step in reversed(range(self.rollout_buffer_size)):
                if step == self.rollout_buffer_size - 1:
                    next_non_terminal = 1.0 - dones
                    next_values = last_values
                else:
                    next_non_terminal = 1.0 - self.is_terminals[step + 1]
                    next_values = self.values[step + 1]
                delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step] # TD error
                last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
                self.advantages[step] = last_gae_lam

            self.returns = self.advantages + self.values

    def get(self, batch_size):
        assert self.full, ""
        indices = np.random.permutation(self.rollout_buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = ["observations",
                            "actions",
                            "values",
                            "log_probs",
                            "advantages",
                            "returns",]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.rollout_buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.rollout_buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds):
        data = (self.observations[batch_inds],
                self.actions[batch_inds],
                self.values[batch_inds].flatten(),
                self.log_probs[batch_inds].flatten(),
                self.advantages[batch_inds].flatten(),
                self.returns[batch_inds].flatten(),)
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))

    def to_torch(self, array):
        return torch.tensor(array, device=self.device)

    @staticmethod
    def swap_and_flatten(arr: np.ndarray):
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features) to [n_steps * n_envs, ...] (which maintain the order)
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])


def z_score_normalize(x):
    mean = torch.mean(x)
    std = torch.std(x)
    return (x - mean) / (std + 1e-8)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, max_action=None):
        super(ActorCritic, self).__init__()
        self.action_dim = action_dim
        self.max_action = max_action

        # Actor
        self.actor_l1 = nn.Linear(obs_dim, 256)
        self.actor_l2 = nn.Linear(256, 256)
        self.actor_mu = nn.Linear(256, action_dim)
        self.actor_std = nn.Linear(256, action_dim)
        
        # Critic
        self.critic_l1 = nn.Linear(obs_dim, 256)
        self.critic_l2 = nn.Linear(256, 256)
        self.critic_q = nn.Linear(256, 1)
        
    def forward(self):
        raise NotImplementedError
    
    def forward_actor(self, state):
        x = z_score_normalize(state)
        x = F.relu(self.actor_l1(x))
        x = F.relu(self.actor_l2(x))
        mu = torch.tanh(self.actor_mu(x)) * self.max_action
        std = F.softplus(self.actor_std(x))

        return mu, std
    
    def forward_critic(self, state):
        x = z_score_normalize(state)
        x = F.relu(self.critic_l1(x))
        x = F.relu(self.critic_l2(x))
        values = self.critic_q(x)

        return values

    def act(self, state):
        mu, std = self.forward_actor(state)
        values = self.forward_critic(state)
        cov = torch.diag_embed(std**2)
        dist = MultivariateNormal(mu, cov)
        actions = dist.sample()
        logprobs = dist.log_prob(actions)

        return actions.detach(), logprobs.detach(), values.detach()
    
    def evaluate(self, states, actions):
        mu, std = self.forward_actor(states)
        values = self.forward_critic(states)
        cov = torch.diag_embed(std**2)
        dist = MultivariateNormal(mu, cov)
        logprobs = dist.log_prob(actions)
        entropy = dist.entropy()

        return logprobs, values, entropy


class PPO:
    def __init__(self,
                 n_envs=1,
                 gamma=0.99,
                 gae_lambda=0.95,
                 rollout_length=256,
                 learning_rate=3e-4,
                 n_epochs=15,
                 ent_coef=0.00,
                 vf_coef=0.5,
                 max_grad_norm=0.5,
                 batch_size=64,
                 clip_range=0.2,
                 obs_dim=None,
                 action_dim=4,
                 max_action=3.0):
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.n_envs = n_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_epochs = n_epochs
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.clip_range = clip_range
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_action = max_action

        self.normalize_advantage = True
        self.rollout_buffer = RolloutBuffer(self.device, rollout_length, n_envs, obs_dim, action_dim, gamma, gae_lambda)

        self.policy = ActorCritic(obs_dim, action_dim, max_action).to(self.device)
        self.policy_old = ActorCritic(obs_dim, action_dim, max_action).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).float().to(self.device)
            actions, logprobs, values = self.policy_old.act(state)

        return actions.cpu().numpy().astype(np.float32), logprobs, values

    def predict_values(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).float().to(self.device)
            values = self.policy.forward_critic(state)
        return values

    def train(self, last_values, dones):
        self.rollout_buffer.compute_returns_and_advantage(last_values, dones)

        # For learning curves
        pg_losses, value_losses = [], []

        # Optimize policy for n epochs
        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                # Evaluating old actions and values
                logprobs, values, entropy = self.policy.evaluate(rollout_data.observations, rollout_data.actions)

                # match state_values tensor dimensions with rewards tensor
                values = values.flatten()

                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Ratio between old and new policy (pi_theta / pi_theta__old), should be one at the first iteration
                ratios = torch.exp(logprobs - rollout_data.old_log_prob)

                # clipped surrogate loss
                surr1 = advantages * ratios
                surr2 = advantages * torch.clamp(ratios, 1-self.clip_range, 1+self.clip_range)
                policy_loss = -torch.min(surr1, surr2).mean()
                pg_losses.append(policy_loss.item()) # Logging

                # Value loss using the TD (ga_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values)
                value_losses.append(value_loss.item()) # Logging

                # Entropy loss favor exploration
                entropy_loss = -torch.mean(entropy)

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Reset buffer
        self.rollout_buffer.reset()
        
        return np.mean(pg_losses), np.mean(value_losses)
    
    def save(self, save_path, timestep):
        filename = os.path.join(save_path, "actor_critic_{}k.pkl".format(timestep))
        torch.save(self.policy_old.state_dict(), filename)
   
    def load(self, save_path):
        self.policy_old.load_state_dict(torch.load(save_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(save_path, map_location=lambda storage, loc: storage))


class Trainer:
    def __init__(self,
                 model=None,
                 env=None,
                 n_envs=1,
                 max_training_timesteps=None,
                 max_episode_steps=None,
                 evaluation_time_steps=None,
                 update_timestep=None,
                 obs_dim=None,
                 action_dim=None,
                 max_action=None,
                 save_dir=None):
        self.model = model
        self.env = env # Parallel environment
        self.n_envs = n_envs
        self.max_training_timesteps = max_training_timesteps
        self.max_episode_steps = max_episode_steps
        self.evaluation_time_steps = evaluation_time_steps
        self.update_timestep = update_timestep
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.save_dir = os.path.join(save_dir, "model", "ppo")

        # Store current step elements
        self._last_obs = np.zeros((self.n_envs, self.obs_dim), dtype=np.float32)
        self._dones = None

        self.writer = SummaryWriter(log_dir="runs/single/ppo/")

    def learn(self, render=False):
        time_step = 0 # Total training time step
        tqdm_bar = tqdm(initial=0, desc="Training", total=self.max_training_timesteps, unit="timestep")
        accumulative_reward, n_episode, print_episode = 0.0, 0, 0
        best_score = None

        if render:
            self.env.connectUnity()

        while time_step < self.max_training_timesteps:
            n_episode += 1 # Start new episode
            print_episode += 1
            epi_step, score = 0, 0.0

            for env_i in range(self.n_envs):
                self._last_obs[env_i] = self.env[env_i].reset()
            
            while not epi_step > self.max_episode_steps:
                tqdm_bar.update(1)
                time_step += 1
                epi_step += 1

                actions, log_probs, values = self.model.select_action(self._last_obs)
                # Action clipping
                clipped_actions = np.clip(actions, -self.max_action, self.max_action)

                new_obs, rewards, new_dones, _ = self.env.step(clipped_actions)

                self.model.rollout_buffer.add(self._last_obs, actions, rewards, self._dones, values, log_probs)

                self._last_obs = new_obs
                self._dones = new_dones

                # update PPO agent
                if time_step % self.update_timestep == 0:
                    last_values = self.model.predict_values(self.new_obs)
                    actor_loss, critic_loss = self.model.train(last_values, self.new_dones)
                    self.writer.add_scalar("actor_loss", actor_loss, global_step=time_step)
                    self.writer.add_scalar("critic_loss", critic_loss, global_step=time_step)
                    
                accumulative_reward += np.mean(rewards)
                score += np.mean(rewards)
                if any(new_dones):
                    break
                
                if time_step % self.evaluation_time_steps == 0:
                    avg_reward = round(accumulative_reward / print_episode, 2)
                    accumulative_reward, print_episode = 0.0, 0
                    print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(n_episode, time_step, avg_reward))
                    
                    if best_score == None or avg_reward > best_score:
                        self.save(self.save_dir, int(time_step/1000))
                        best_score = avg_reward

            self.writer.add_scalar("score", score, global_step=time_step) # Save episodic reward

        tqdm_bar.close()
        self.env.close()
        self.writer.flush()
        self.writer.close()

        if render:
            self.env.disconnectUnity()

    def save(self, save_dir=None, timestep=None):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.model.save(save_dir, timestep)