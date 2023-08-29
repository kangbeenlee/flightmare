import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys
import os
from tqdm import tqdm

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
        mu = torch.tanh(self.fc_mu(x)) * 3 # Velocity limitation
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
    def __init__(self, obs_dim=None, action_dim=None, memory_capacity=None, batch_size=None):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size

        self.ptr = 0
        self.size = 0
        
        # Initialize replay buffer
        self.buffer = {'obs': np.zeros([self.memory_capacity, self.obs_dim]),
                       'a': np.zeros([self.memory_capacity, self.action_dim]),
                       'r': np.zeros([self.memory_capacity, 1]),
                       'obs_prime': np.zeros([self.memory_capacity, self.obs_dim]),
                       'done': np.ones([self.memory_capacity, 1])
                       }

    def store(self, obs, a, r, obs_prime, done):
        self.buffer['obs'][self.ptr] = obs
        self.buffer['a'][self.ptr] = a
        self.buffer['r'][self.ptr] = r
        self.buffer['obs_prime'][self.ptr] = obs_prime
        self.buffer['done'][self.ptr] = done

        # Rewrite the experience from the begining like FIFO style rather than pop
        self.ptr = (self.ptr + 1) % self.memory_capacity
        self.size = min(self.size + 1, self.memory_capacity)

    def sample(self):
        indices = np.random.choice(self.size, size=self.batch_size, replace=False)
        mini_batch = {}
        for key in self.buffer.keys():
            mini_batch[key] = torch.tensor(self.buffer[key][indices], dtype=torch.float32)
        return mini_batch
    
    def __len__(self):
        return self.size

class DDPG:
    def __init__(self, device=None, gamma=0.99, lr_actor=5e-4, lr_critic=0.001, tau=0.005, use_hard_update=False, target_update_period=None,
                 obs_dim=12, action_dim=4):
        self.device = device
        
        self.actor = Actor(obs_dim, action_dim).to(device)
        self.critic = Critic(obs_dim, action_dim).to(device)
        # Target network
        self.target_actor = Actor(obs_dim, action_dim).to(device)
        self.target_critic = Critic(obs_dim, action_dim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Optimizer        
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(action_dim))
        
        self.gamma = gamma
        
        self.tau = tau
        self.update = 0
        self.use_hard_update = use_hard_update
        self.target_update_period = target_update_period
        
    # Sample continuous action
    def choose_action(self, obs):
        obs = torch.from_numpy(obs).float().to(self.device)
        with torch.no_grad():
            action = self.actor(obs).cpu().numpy() + self.ou_noise()
        return action.astype(np.float32)

    def train(self, memory):
        self.update += 1
        mini_batch = memory.sample()
        
        obs = mini_batch['obs'].to(self.device)
        a = mini_batch['a'].to(self.device)
        r = mini_batch['r'].to(self.device)
        obs_prime = mini_batch['obs_prime'].to(self.device)
        done = mini_batch['done'].to(self.device)
        
        with torch.no_grad():
            td_target = r + self.gamma * self.target_critic(obs_prime, self.target_actor(obs_prime)) * (1 - done)

        # Weights update of value network (gradient descent)
        q_loss = F.smooth_l1_loss(self.critic(obs, a), td_target.detach())
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()
        
        # Weights update of policy network (gradient ascent)
        actor_loss = -self.critic(obs, self.actor(obs)).mean() # .mean() -> batch mean
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Target update
        if self.use_hard_update:
            if self.update % self.target_update_period == 0:
                self.target_actor.load_state_dict(self.actor.state_dict())
                self.target_critic.load_state_dict(self.critic.state_dict())
        else:
            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


        return (q_loss.data.item() + actor_loss.data.item()) / 2

    def save_models(self, save_path):
        filename_actor = os.path.join(save_path, "saved/ddpg_actor.pkl")
        filename_critic = os.path.join(save_path, "saved/ddpg_critic.pkl")
        
        with open(filename_actor, 'wb') as f:
            torch.save(self.actor.state_dict(), f)
        with open(filename_critic, 'wb') as f:
            torch.save(self.critic.state_dict(), f)
            
    def load_models(self, load_nn):
        filename_actor = os.path.join(load_nn, "ddpg_actor.pkl")
        filename_critic = os.path.join(load_nn, "ddpg_critic.pkl")
        self.actor.load_state_dict(torch.load(filename_actor))
        self.critic.load_state_dict(torch.load(filename_critic))

class Trainer:
    def __init__(self, model=None, env=None, num_episodes=None, max_episode_steps=None, obs_dim=None, action_dim=None, memory_capacity=None,
                 batch_size=None, training_start=None):
        self.model = model
        self.env = env
        self.num_episodes = num_episodes
        self.max_episode_steps = max_episode_steps
        self.training_start = training_start
        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, action_dim=action_dim, memory_capacity=memory_capacity, batch_size=batch_size)

        # Tensorboard results
        self.writer = SummaryWriter(log_dir="runs/DDPG/")

    def learn(self, render=False):
        score = 0.0
        epi_step = 0
        tqdm_bar = tqdm(initial=0, desc="Training", total=self.num_episodes, unit="episode")

        if render:
            self.env.connectUnity()

        for episode in range(self.num_episodes):
            # Initialize
            epi_step += 1
            tqdm_bar.update(1)
            obs = self.env.reset()

            for i in range(self.max_episode_steps):
                action = self.model.choose_action(obs)
                obs_prime, reward, done, _ = self.env.step(action)
                self.replay_buffer.store(obs, action, reward, obs_prime, done)
                obs = obs_prime

                score += reward[0] # Just for single agent
                if done:
                    break

                if len(self.replay_buffer) > self.training_start:
                    loss = self.model.train(self.replay_buffer)

            if episode % 20 == 0 and episode != 0:
                average_reward = round(score/20, 1)
                print("train episode: {}, average reward: {:.1f}, buffer size: {}".format(episode, average_reward, len(self.replay_buffer)))
                self.writer.add_scalar("score", average_reward, global_step=episode)
                self.writer.add_scalar("loss", loss, global_step=episode)
                score = 0.0 # Initialize score every 100 episodes

        tqdm_bar.close()
        self.env.close()
        self.writer.flush()
        self.writer.close()

        if render:
            self.env.disconnectUnity()

    def save(self, save_dir=None):
        self.model.save_models(save_dir)