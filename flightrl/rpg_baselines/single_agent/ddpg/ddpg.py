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
    def __init__(self, obs_dim=None, action_dim=None, max_action=None):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, action_dim)
        
        self.max_action = max_action
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.max_action * torch.tanh(self.fc_mu(x))
        return mu

# Action value network
class Critic(nn.Module):
    def __init__(self, obs_dim=None, action_dim=None):
        super(Critic, self).__init__()
        self.fc_s = nn.Linear(obs_dim, 128)
        self.fc_a = nn.Linear(action_dim, 128)
        self.fc = nn.Linear(256, 256)
        self.fc_q = nn.Linear(256, 1)

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
    def __init__(self,
                 obs_dim=None,
                 action_dim=None,
                 memory_capacity=None,
                 batch_size=None):
        
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
    def __init__(self,
                 device=None,
                 gamma=0.99,
                 lr_actor=3e-4,
                 lr_critic=3e-4,
                 tau=0.005,
                 obs_dim=12,
                 action_dim=4,
                 max_action=3.0):
        
        self.device = device
        
        self.actor = Actor(obs_dim, action_dim, max_action).to(device)
        self.critic = Critic(obs_dim, action_dim).to(device)
        # Target network
        self.target_actor = Actor(obs_dim, action_dim, max_action).to(device)
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
        critic_loss = F.smooth_l1_loss(self.critic(obs, a), td_target.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Weights update of policy network (gradient ascent)
        actor_loss = -self.critic(obs, self.actor(obs)).mean() # .mean() -> batch mean
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Target update
        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.data.item(), actor_loss.data.item()

    def save(self, save_path, episode):
        save_path = os.path.join(save_path, "saved")
        filename_actor = os.path.join(save_path, "actor_{}.pkl".format(episode))
        filename_critic = os.path.join(save_path, "critic_{}.pkl".format(episode))
        
        with open(filename_actor, 'wb') as f:
            torch.save(self.actor.state_dict(), f)
        with open(filename_critic, 'wb') as f:
            torch.save(self.critic.state_dict(), f)
            
    def load(self, load_nn_actor, load_nn_critic):
        self.actor.load_state_dict(torch.load(load_nn_actor))
        self.critic.load_state_dict(torch.load(load_nn_critic))

class Trainer:
    def __init__(self, model=None, env=None, num_episodes=None, max_episode_steps=None, obs_dim=None, action_dim=None, memory_capacity=None,
                 batch_size=None, training_start=None, save_dir=None):
        self.model = model
        self.env = env
        self.num_episodes = num_episodes
        self.max_episode_steps = max_episode_steps
        self.training_start = training_start
        self.save_dir = save_dir
        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, action_dim=action_dim, memory_capacity=memory_capacity, batch_size=batch_size)

        # Tensorboard results
        self.writer = SummaryWriter(log_dir="runs/ddpg/")

    def evaluate_policy(self, env, policy, eval_episodes=10):
        avg_reward = 0.0
        
        for _ in range(eval_episodes):
            obs, done, epi_step = env.reset(), False, 0
            while not (done or (epi_step >= self.max_episode_steps)):
                epi_step += 1
                
                action = policy.choose_action(obs)
                w_z = np.array([[0.0]])
                action = np.concatenate((action, w_z), axis=1).astype(np.float32)
                
                obs, reward, done, _ = env.step(action)
                avg_reward += reward[0]

        avg_reward /= eval_episodes
        return avg_reward

    def learn(self, render=False):
        time_step = 0 # Total training time step
        tqdm_bar = tqdm(initial=0, desc="Training", total=self.num_episodes, unit="episode")
        best_score = None

        if render:
            self.env.connectUnity()

        for episode in range(self.num_episodes):
            tqdm_bar.update(1) # Initialize
            obs, done, epi_step, score = self.env.reset(), False, 0, 0.0
            
            while not (done or (epi_step >= self.max_episode_steps)):
                time_step += 1
                epi_step += 1
                
                action = self.model.choose_action(obs)
                w_z = np.array([[0.0]])
                temp_action = np.concatenate((action, w_z), axis=1).astype(np.float32)
                obs_prime, reward, done, _ = self.env.step(temp_action)
                # obs_prime, reward, done, _ = self.env.step(action)
                
                # reward /= 10
                
                self.replay_buffer.store(obs, action, reward, obs_prime, done)
                obs = obs_prime

                score += reward[0] # Just for single agent
                if done:
                    break

                if time_step > self.training_start:
                    critic_loss, actor_loss = self.model.train(self.replay_buffer)
                    self.writer.add_scalar("critic_loss", critic_loss, global_step=time_step)
                    self.writer.add_scalar("actor_loss", actor_loss, global_step=time_step)

            self.writer.add_scalar("score", score, global_step=episode)

            if (episode + 1) % 20 == 0:
                eval_score = self.evaluate_policy(self.env, self.model)
                print(f">>> Evaluation reward at episode {episode + 1}: {eval_score:.3f}")
                
                if best_score == None or eval_score > best_score:
                    self.save(self.save_dir, episode + 1)
                    best_score = eval_score

        tqdm_bar.close()
        self.env.close()
        self.writer.flush()
        self.writer.close()

        if render:
            self.env.disconnectUnity()

    def save(self, save_dir=None, episode=None):
        self.model.save(save_dir, episode)