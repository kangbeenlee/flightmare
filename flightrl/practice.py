import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys
import os
from torch.utils.tensorboard import SummaryWriter



class Actor(nn.Module):
    def __init__(self, obs_dim=None, action_dim=None):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))
        return mu

# Action value network
class Critic(nn.Module):
    def __init__(self, obs_dim=None, action_dim=None):
        super(Critic, self).__init__()
        self.fc_s = nn.Linear(obs_dim, 64)
        self.fc_a = nn.Linear(action_dim, 64)
        self.fc = nn.Linear(128, 32)
        self.fc_q = nn.Linear(32,1)

    def forward(self, s, a):
        h1 = F.relu(self.fc_s(s))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1,h2], dim=1)
        q = F.relu(self.fc(cat))
        q = self.fc_q(q)
        return q

class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

class ReplayBuffer:
    def __init__(self, N=1, obs_dim=None, action_dim=None, memory_capacity=50000):
        self.memory_capacity = memory_capacity
        self.ptr = 0
        self.size = 0
        
        # Initialize replay buffer
        self.buffer = {'obs_n': np.zeros([self.memory_capacity, N, obs_dim]),
                       'a_n': np.zeros([self.memory_capacity, N, action_dim]),
                       'r_n': np.zeros([self.memory_capacity, N, 1]),
                       'obs_prime_n': np.zeros([self.memory_capacity, N, obs_dim]),
                       'done_n': np.ones([self.memory_capacity, N, 1])
                       }

    def store(self, obs_n, a_n, r_n, obs_prime_n, done_n):
        self.buffer['obs_n'][self.ptr] = obs_n
        self.buffer['a_n'][self.ptr] = a_n
        self.buffer['r_n'][self.ptr] = r_n
        self.buffer['obs_prime_n'][self.ptr] = obs_prime_n
        self.buffer['done_n'][self.ptr] = done_n

        # Rewrite the experience from the begining like FIFO style rather than pop
        self.ptr = (self.ptr + 1) % self.memory_capacity
        self.size = min(self.size + 1, self.memory_capacity)

    def sample(self):
        idx = np.random.choice(self.size, replace=False)
        mini_batch = {}
        for key in self.buffer.keys():
            mini_batch[key] = torch.tensor(self.buffer[key][idx], dtype=torch.float32).squeeze(0)
        return mini_batch
    
    def __len__(self):
        return self.size


class Trainer:
    def __init__(self, env=None, total_timesteps=1000, rollout_steps=200, gamma=0.99, lr_q=0.001, lr_mu=5e-4,
                 training_start=2000, tau=0.005, use_hard_update=False, target_update_period=300,
                 memory_capacity=50000, device=None):
        self.device = device
        self.env = env
        # The number of parallelized drones
        self.N = env.num_envs
        self.obs_dim = env.num_obs
        self.action_dim = env.num_obs

        # Discount factor
        self.gamma = gamma

        # Training paramters
        self.total_timesteps = total_timesteps
        self.rollout_steps = rollout_steps

        # Store episode
        self.replay_buffer = ReplayBuffer(N=env.num_envs, obs_dim=env.num_obs, action_dim=env.num_acts, memory_capacity=memory_capacity)

        # Actor network and Q network
        self.mu = Actor(obs_dim=env.num_obs, action_dim=env.num_acts).to(device)
        self.target_mu = Actor(obs_dim=env.num_obs, action_dim=env.num_acts).to(device)
        self.q_network = Critic(obs_dim=env.num_obs, action_dim=env.num_acts).to(device)
        self.target_q_network = Critic(obs_dim=env.num_obs, action_dim=env.num_acts).to(device)
        self.target_mu.load_state_dict(self.mu.state_dict())
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        # Target network update
        self.train_update = 0
        self.training_start = training_start
        self.target_update_period = target_update_period
        self.use_hard_update = use_hard_update
        self.tau = tau

        # Optimizer        
        self.lr_q = lr_q
        self.lr_mu = lr_mu
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr_q)
        self.mu_optimizer = optim.Adam(self.mu.parameters(), lr=self.lr_mu)
        self.ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))
        
        # Tensorboard results
        self.writer = SummaryWriter(log_dir="runs/DDPG/")

    # Sample continuous action
    def chooseAction(self, obs):
        with torch.no_grad():
            action = self.mu(obs) + self.ou_noise()[0]
        return action.cpu().numpy()

    def train(self):
        self.train_update += 1
        mini_batch = self.replay_buffer.sample()
        
        obs_n = mini_batch['obs_n'].to(self.device) # obs_n.shape=(n_envs, obs_dim)
        a_n = mini_batch['a_n'].to(self.device) # obs_n.shape=(n_envs, action_dim)
        r_n = mini_batch['r_n'].to(self.device) # obs_n.shape=(n_envs, 1)
        obs_prime_n = mini_batch['obs_prime_n'].to(self.device) # obs_n.shape=(n_envs, obs_dim)
        done_n = mini_batch['done_n'].to(self.device) # obs_n.shape=(n_envs, 1)

        with torch.no_grad():
            td_target = r_n + self.gamma * self.target_q_network(obs_prime_n, self.target_mu(obs_prime_n)) * (1 - done_n)

        # Weights update of policy network (gradient ascent)
        mu_loss = -self.q_network(obs_n, self.mu(obs_n)).mean() # .mean() -> batch mean
        self.mu_optimizer.zero_grad()
        mu_loss.backward()
        self.mu_optimizer.step()

        # Weights update of value network (gradient descent)
        q_loss = F.smooth_l1_loss(self.q_network(obs_n, a_n), td_target.detach())
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Target update
        if self.use_hard_update:
            if self.train_update % self.target_update_period == 0:
                self.target_mu.load_state_dict(self.mu.state_dict())
                self.target_q_network.load_state_dict(self.q_network.state_dict())
        else:
            for param, target_param in zip(self.mu.parameters(), self.target_mu.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.q_network.parameters(), self.target_q_network.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return (mu_loss.data.item() + q_loss.data.item()) / 2

    def learn(self, render=False):
        step = 0

        assert self.total_timesteps % self.rollout_steps == 0, "Total time steps should be divided into rollout steps without rest"
        num_episodes = self.total_timesteps // (self.N * self.rollout_steps)

        if render:
            self.env.connectUnity()

        for episode in range(num_episodes):
            # Initialize
            score = 0
            obs_n = self.env.reset()

            for epi_step in range(self.rollout_steps):
                step += 1
                
                action_n = self.chooseAction(torch.from_numpy(obs_n).float().to(self.device))      
                obs_prime_n, reward_n, done_n, _ = self.env.step(action_n)
                
                # Reshape
                reward_n = reward_n.reshape(-1, 1)
                done_n = done_n.reshape(-1, 1)
                
                self.replay_buffer.store(obs_n, action_n, reward_n, obs_prime_n, done_n)
                obs_n = obs_prime_n

                score += reward_n.mean()
                
                # if done:
                #     print("Done step:", epi_step)
                #     break

                if len(self.replay_buffer) > self.training_start:
                    loss = self.train()
                    self.writer.add_scalar("loss", loss, global_step=step)

            print("train episode: {}, score: {:.1f}, buffer size: {}".format(episode+1, score, len(self.replay_buffer)))
            self.writer.add_scalar("score", score, global_step=episode)

        self.env.close()
        self.writer.flush()
        self.writer.close()

        if render:
            self.env.disconnectUnity()

    def save(self, save_path=None):
        filename_actor = os.path.join(save_path, "ddpg_actor.pkl")
        filename_critic = os.path.join(save_path, "ddpg_critic.pkl")
        
        with open(filename_actor, 'wb') as f:
            torch.save(self.mu.state_dict(), f)
        with open(filename_critic, 'wb') as f:
            torch.save(self.q_network.state_dict(), f)