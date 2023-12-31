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


# Trick : z score normalization
def z_score_normalize(x):
    x = (x - x.mean()) / (x.std() + 1e-8)
    return x


# Trick : orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)



# # ************************************************************************
# # ******************************** SRT ***********************************
# # ************************************************************************
# class Actor(nn.Module):
#     def __init__(self, args, obs_dim, action_dim, max_action):
#         super(Actor, self).__init__()
#         self.use_orthogonal_init = args.use_orthogonal_init
#         self.use_z_score_normalization = args.use_z_score_normalization
#         self.max_action = max_action
#         self.l1 = nn.Linear(obs_dim, 256)
#         self.l2 = nn.Linear(256, 256)
#         self.l3 = nn.Linear(256, action_dim)

#         if self.use_orthogonal_init:
#             print("------use_orthogonal_init------")
#             orthogonal_init(self.l1)
#             orthogonal_init(self.l2)
#             orthogonal_init(self.l3)

#     def forward(self, x):
#         with torch.autograd.set_detect_anomaly(True):
#             if self.use_z_score_normalization:
#                 x = z_score_normalize(x)     
#             x = F.relu(self.l1(x))
#             x = F.relu(self.l2(x))
#             return torch.tanh(self.l3(x))



# # ************************************************************************
# # ******************************** CTBR **********************************
# # ************************************************************************
# class Actor(nn.Module):
#     def __init__(self, args, obs_dim, action_dim, max_action):
#         super(Actor, self).__init__()
#         self.use_orthogonal_init = args.use_orthogonal_init
#         self.use_z_score_normalization = args.use_z_score_normalization
#         self.max_action = max_action
#         self.l1 = nn.Linear(obs_dim, 256)
#         self.l2 = nn.Linear(256, 256)
#         self.l3 = nn.Linear(256, 1)
#         self.l4 = nn.Linear(256, 3)

#         if self.use_orthogonal_init:
#             print("------use_orthogonal_init------")
#             orthogonal_init(self.l1)
#             orthogonal_init(self.l2)
#             orthogonal_init(self.l3)
#             orthogonal_init(self.l4)

#     def forward(self, x):
#         if self.use_z_score_normalization:
#             x = z_score_normalize(x)     
#         x = F.relu(self.l1(x))
#         x = F.relu(self.l2(x))
#         c = torch.sigmoid(self.l3(x)) # Collective thrust
#         w = torch.tanh(self.l4(x)) * 3.0 # Body-Rates
#         return torch.cat([c, w], dim=-1)



# ************************************************************************
# ********************************* LV ***********************************
# ************************************************************************
class Actor(nn.Module):
    def __init__(self, args, obs_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.use_orthogonal_init = args.use_orthogonal_init
        self.use_z_score_normalization = args.use_z_score_normalization
        self.max_action = max_action
        self.l1 = nn.Linear(obs_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        if self.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.l1)
            orthogonal_init(self.l2)
            orthogonal_init(self.l3)

    def forward(self, x):
        if self.use_z_score_normalization:
            x = z_score_normalize(x)     
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.max_action * torch.tanh(self.l3(x))


class Critic(nn.Module):
    def __init__(self, args, obs_dim, action_dim):
        super(Critic, self).__init__()
        self.use_orthogonal_init = args.use_orthogonal_init
        self.use_z_score_normalization = args.use_z_score_normalization

        # Q1 architecture
        self.l1 = nn.Linear(obs_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        
        # Q2 architecture
        self.l4 = nn.Linear(obs_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

        if self.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.l1)
            orthogonal_init(self.l2)
            orthogonal_init(self.l3)
            orthogonal_init(self.l4)
            orthogonal_init(self.l5)
            orthogonal_init(self.l6)

    def forward(self, s, a):
        if self.use_z_score_normalization:
            s = z_score_normalize(s)
            a = z_score_normalize(a)
        sa = torch.cat([s, a], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2
    
    def Q1(self, s, a):
        if self.use_z_score_normalization:
            s = z_score_normalize(s)
            a = z_score_normalize(a)
        sa = torch.cat([s, a], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class ReplayBuffer(object):
    def __init__(self, args, obs_dim, action_dim, memory_capacity=int(1e6), batch_size=256):
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0
        
        self.state = np.zeros((memory_capacity, obs_dim))
        self.action = np.zeros((memory_capacity, action_dim))
        self.next_obs = np.zeros((memory_capacity, obs_dim))
        self.reward = np.zeros((memory_capacity, 1))
        self.done = np.zeros((memory_capacity, 1))
        
        self.device = args.device
        
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
        args,
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

        self.device = args.device
        self.actor = Actor(args, obs_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        self.critic = Critic(args, obs_dim, action_dim).to(self.device)
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

    def select_action(self, state):
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

        # if self.use_grad_clip:
        #     torch.nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)

        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            # Save loss
            self.prev_actor_loss = actor_loss.data.item()
            return critic_loss.data.item(), self.prev_actor_loss
        else:
            return critic_loss.data.item(), self.prev_actor_loss

    def save(self, save_path, timestep):
        filename_actor = os.path.join(save_path, "actor_{}k.pkl".format(timestep))
        # filename_critic = os.path.join(save_path, "critic_{}k.pkl".format(timestep))
        torch.save(self.actor.state_dict(), filename_actor)
        # torch.save(self.critic.state_dict(), filename_critic)

    def load(self, load_nn):
        self.actor.load_state_dict(torch.load(load_nn, map_location='cuda:0'))
        # self.critic.load_state_dict(torch.load(load_nn_critic))

class Trainer:
    def __init__(
            self,
            args=None,
            model=None,
            env=None,
            max_training_timesteps=None,
            max_episode_steps=None,
            evaluation_time_steps=None,
            evaluation_times=None,
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
        self.max_training_timesteps = max_training_timesteps
        self.max_episode_steps = max_episode_steps
        self.evaluation_time_steps = evaluation_time_steps
        self.evaluation_times = evaluation_times
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.expl_noise = expl_noise
        self.training_start = training_start
        self.save_dir = os.path.join(save_dir, "model", "batch_{}_td3".format(batch_size))
        # self.save_dir = os.path.join(save_dir, "model", "td3_srt".format(batch_size))
        self.replay_buffer = ReplayBuffer(args, obs_dim=obs_dim, action_dim=action_dim, memory_capacity=memory_capacity, batch_size=batch_size)

        # Tensorboard results
        self.writer = SummaryWriter(log_dir="runs/single/batch_{}_td3/".format(batch_size))
        # self.writer = SummaryWriter(log_dir="runs/single/td3_srt/".format(batch_size))

        self.main = 0

    def evaluate_policy(self, env, policy, max_episode_steps, eval_episodes=10):
        avg_reward = 0.
        for _ in range(eval_episodes):
            obs, done, epi_step = env.reset(), False, 0
            while not (done or (epi_step > max_episode_steps)):
                epi_step += 1
                action = policy.select_action(np.array(obs))
                obs, reward, done, _ = env.step(action)
                avg_reward += reward[0]
        avg_reward = round(avg_reward / eval_episodes, 2)
        return avg_reward

    def learn(self, render=False):
        time_step = 0 # Total training time step
        tqdm_bar = tqdm(initial=0, desc="Training", total=self.max_training_timesteps, unit="timestep", dynamic_ncols=True)
        n_episode, best_score, best_timestep = 0, None, 0

        if render:
            self.env.connectUnity()

        while time_step < self.max_training_timesteps:
            n_episode += 1 # Start new episode
            obs, epi_step, score = self.env.reset(), 0, 0.0
            
            while not epi_step > self.max_episode_steps:
                tqdm_bar.update(1)
                time_step += 1
                epi_step += 1
                
                # Select action randomly or according to policy
                if time_step < self.training_start:
                    action = np.random.uniform(-self.max_action, self.max_action, self.action_dim).reshape(1, -1).astype(np.float32)
                else:
                    action = (self.model.select_action(np.array(obs)) + 
                              np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)).clip(-self.max_action, self.max_action).reshape(1, -1).astype(np.float32)
                    
                obs_prime, reward, done, _ = self.env.step(action)
                self.replay_buffer.add(obs, action, reward, obs_prime, done)
                obs = obs_prime

                score += reward[0] # Record episodic reward
                if done:
                    break

                if time_step > self.training_start:
                    critic_loss, actor_loss = self.model.train(self.replay_buffer)
                    if actor_loss:
                        self.writer.add_scalar("critic_loss", critic_loss, global_step=time_step)
                        self.writer.add_scalar("actor_loss", -actor_loss, global_step=time_step)

                if time_step % self.evaluation_time_steps == 0:
                    avg_reward = self.evaluate_policy(self.env, self.model, self.max_episode_steps, self.evaluation_times)
                    if best_score == None or avg_reward > best_score:
                        best_score = avg_reward
                        best_timestep = int(time_step/1000)

                    print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {} \t\t Best timestep : {}k".format(n_episode, time_step, avg_reward, best_timestep))
                    self.save(self.save_dir, int(time_step/1000))

            self.writer.add_scalar("score", score, global_step=time_step) # Save episodic reward

        tqdm_bar.close()
        self.env.close()
        self.writer.flush()
        self.writer.close()

    # def evaluate_policy(self, env, policy, max_episode_steps, eval_episodes=10):
    #     avg_reward = 0.
    #     for _ in range(eval_episodes):
    #         obs, done, epi_step = env.reset(), False, 0
    #         obs = obs[self.main]
    #         while not (done or (epi_step > max_episode_steps)):
    #             epi_step += 1
    #             action = policy.select_action(np.array(obs)).reshape(1, -1).astype(np.float32)

    #             pseudo_action = np.array([[0.0, 0.0, 0.0, 0.0],
    #                                       [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    #             # pseudo_action = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    #             pseudo_action = np.concatenate((action, pseudo_action), axis=0)

    #             obs, reward, done, _ = env.step(pseudo_action)
    #             obs = obs[self.main]
    #             done = done[self.main]

    #             avg_reward += reward[self.main]
    #     avg_reward = round(avg_reward / eval_episodes, 2)
    #     return avg_reward

    # def learn(self, render=False):
    #     time_step = 0 # Total training time step
    #     tqdm_bar = tqdm(initial=0, desc="Training", total=self.max_training_timesteps, unit="timestep", dynamic_ncols=True)
    #     n_episode, best_score, best_timestep = 0, None, 0

    #     if render:
    #         self.env.connectUnity()

    #     while time_step < self.max_training_timesteps:
    #         n_episode += 1 # Start new episode
    #         obs, epi_step, score = self.env.reset(), 0, 0.0
    #         obs = obs[self.main]
            
    #         while not epi_step > self.max_episode_steps:
    #             tqdm_bar.update(1)
    #             time_step += 1
    #             epi_step += 1
                
    #             # Select action randomly or according to policy
    #             if time_step < self.training_start:
    #                 action = np.random.uniform(-self.max_action, self.max_action, self.action_dim).reshape(1, -1).astype(np.float32)
    #             else:
    #                 action = (self.model.select_action(np.array(obs)) + 
    #                           np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)).clip(-self.max_action, self.max_action).reshape(1, -1).astype(np.float32)
                    
    #             pseudo_action = np.array([[0.0, 0.0, 0.0, 0.0],
    #                                       [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    #             # pseudo_action = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    #             pseudo_action = np.concatenate((action, pseudo_action), axis=0)

    #             obs_prime, reward, done, _ = self.env.step(pseudo_action)

    #             obs_prime = obs_prime[self.main]
    #             reward = reward[self.main]
    #             done = done[self.main]

    #             self.replay_buffer.add(obs, action, reward, obs_prime, done)
    #             obs = obs_prime

    #             score += reward # Record episodic reward
    #             if done:
    #                 break

    #             if time_step > self.training_start:
    #                 critic_loss, actor_loss = self.model.train(self.replay_buffer)
    #                 if actor_loss:
    #                     self.writer.add_scalar("critic_loss", critic_loss, global_step=time_step)
    #                     self.writer.add_scalar("actor_loss", -actor_loss, global_step=time_step)

    #             if time_step % self.evaluation_time_steps == 0:
    #                 avg_reward = self.evaluate_policy(self.env, self.model, self.max_episode_steps, self.evaluation_times)
    #                 if best_score == None or avg_reward > best_score:
    #                     best_score = avg_reward
    #                     best_timestep = int(time_step/1000)

    #                 print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {} \t\t Best timestep : {}k".format(n_episode, time_step, avg_reward, best_timestep))
    #                 self.save(self.save_dir, int(time_step/1000))

    #         self.writer.add_scalar("score", score, global_step=time_step) # Save

    #     tqdm_bar.close()
    #     self.env.close()
    #     self.writer.flush()
    #     self.writer.close()

    def save(self, save_dir=None, timestep=None):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.model.save(save_dir, timestep)