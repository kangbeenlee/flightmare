import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
from tqdm import tqdm
import random
import gym
from torch.utils.tensorboard import SummaryWriter



# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class Actor(nn.Module):
	def __init__(self, obs_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(obs_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, obs_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(obs_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(obs_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1

class ReplayBuffer(object):
    def __init__(self, obs_dim, action_dim, memory_capacity=int(1e6), batch_size=256):
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0
        
        self.state = np.zeros((memory_capacity, obs_dim))
        self.action = np.zeros((memory_capacity, action_dim))
        self.next_obs = np.zeros((memory_capacity, obs_dim))
        self.reward = np.zeros((memory_capacity, 1))
        self.done = np.zeros((memory_capacity, 1))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def add(self, state, action, reward, next_obs, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.memory_capacity
        self.size = min(self.size + 1, self.memory_capacity)
        
    def sample(self):
        indices = np.random.randint(0, self.size, size=self.batch_size)
        return (
            torch.FloatTensor(self.state[indices]).to(self.device),
            torch.FloatTensor(self.action[indices]).to(self.device),
            torch.FloatTensor(self.next_obs[indices]).to(self.device),
            torch.FloatTensor(self.reward[indices]).to(self.device),
            torch.FloatTensor(self.done[indices]).to(self.device)
        )
        
class TD3(object):
    def __init__(
		self,
        device,
        gamma=0.99,
        lr_actor=3e-4,
        lr_critic=3e-4,
        tau=0.005,                        
        policy_noise=0.2, # std
        noise_clip=0.5,
        policy_freq=2,                        
        obs_dim=17,
        action_dim=4,
        max_action=3.0):

        self.device = device
        self.actor = Actor(obs_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        self.critic = Critic(obs_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        
        self.total_it = 0
        self.prev_actor_loss = None

    # def choose_action(self, state):
    #     state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
    #     return self.actor(state).cpu().data.numpy()

    def choose_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()
        return action.astype(np.float32)

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer 
        state, action, next_obs, reward, done = replay_buffer.sample()

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_obs) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + self.gamma * target_Q * (1 - done)

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            # Save loss
            self.prev_actor_loss = actor_loss.data.item()
            return critic_loss.data.item(), actor_loss.data.item()
        else:
            return critic_loss.data.item(), self.prev_actor_loss

    def save(self, save_path, episode):
        save_path = os.path.join(save_path, "saved")
        filename_actor = os.path.join(save_path, "actor_{}.pkl".format(episode))
        filename_critic = os.path.join(save_path, "critic_{}.pkl".format(episode))
        
        with open(filename_actor, 'wb') as f:
            torch.save(self.actor.state_dict(), f)
        with open(filename_critic, 'wb') as f:
            torch.save(self.critic.state_dict(), f)

    def load(self, load_nn_actor, load_nn_critic):
        self.critic.load_state_dict(torch.load(load_nn_critic))
        # self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        # self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(load_nn_actor))
        # self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        # self.actor_target = copy.deepcopy(self.actor)

class Trainer:
    def __init__(
            self,
            model=None,
            env=None,
            num_episodes=None,
            max_episode_steps=None,
            obs_dim=None,
            action_dim=None,
            max_action=None,
            expl_noise=None,
            memory_capacity=None,
            batch_size=None,
            training_start=None,
            save_dir=None
        ):
        
        self.model = model
        self.env = env
        self.num_episodes = num_episodes
        self.max_episode_steps = max_episode_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.expl_noise = expl_noise
        self.training_start = training_start
        self.save_dir = save_dir
        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, action_dim=action_dim, memory_capacity=memory_capacity, batch_size=batch_size)

        # Tensorboard results
        self.writer = SummaryWriter(log_dir="runs/td3/")

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

                # Select action randomly or according to policy
                if time_step < self.training_start:
                    action = np.random.uniform(-self.max_action, self.max_action, self.action_dim).reshape(1, -1)
                else:
                    action = (self.model.choose_action(np.array(obs)) +
                              np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)).clip(-self.max_action, self.max_action).reshape(1, -1)

                w_z = np.array([[0.0]])
                temp_action = np.concatenate((action, w_z), axis=1).astype(np.float32)

                obs_prime, reward, done, _ = self.env.step(temp_action)
                self.replay_buffer.add(obs, action, reward, obs_prime, done)
                obs = obs_prime

                score += reward[0] # Just for single agent
                if done:
                    break

                if time_step > self.training_start:
                    critic_loss, actor_loss = self.model.train(self.replay_buffer)
                    if actor_loss:
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