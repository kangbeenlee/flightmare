import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys
import os
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter



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
#     def __init__(self, args, obs_dim=None, action_dim=None, max_action=None):
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
#     def __init__(self, args, obs_dim=None, action_dim=None, max_action=None):
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
    def __init__(self, args, obs_dim=None, action_dim=None, max_action=None):
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
    def __init__(self, args, obs_dim=None, action_dim=None):
        super(Critic, self).__init__()        
        self.use_orthogonal_init = args.use_orthogonal_init
        self.use_z_score_normalization = args.use_z_score_normalization
        self.l1 = nn.Linear(obs_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        if self.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.l1)
            orthogonal_init(self.l2)
            orthogonal_init(self.l3)

    def forward(self, s, a):
        with torch.autograd.set_detect_anomaly(True):
            if self.use_z_score_normalization:
                s = z_score_normalize(s)
                a = z_score_normalize(a)
            q = F.relu(self.l1(torch.cat([s, a], 1)))
            q = F.relu(self.l2(q))
            return self.l3(q)


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
                 args,
                 gamma=0.99,
                 lr_actor=3e-4,
                 lr_critic=3e-4,
                 tau=0.005,
                 obs_dim=12,
                 action_dim=4,
                 max_action=3.0):
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.actor = Actor(args, obs_dim, action_dim, max_action).to(self.device)
        self.critic = Critic(args, obs_dim, action_dim).to(self.device)
        self.target_actor = Actor(args, obs_dim, action_dim, max_action).to(self.device)
        self.target_critic = Critic(args, obs_dim, action_dim).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Optimizer
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        self.gamma = gamma
        
        self.tau = tau
        self.update = 0
        
    # Sample continuous action
    def select_action(self, obs):
        obs = torch.from_numpy(obs).float().to(self.device)
        with torch.no_grad():
            action = self.actor(obs).cpu().numpy()
        return action.astype(np.float32)

    def train(self, memory):
        with torch.autograd.set_detect_anomaly(True):

            self.update += 1
            mini_batch = memory.sample()
            
            obs = mini_batch['obs'].to(self.device)
            a = mini_batch['a'].to(self.device)
            r = mini_batch['r'].to(self.device)
            obs_prime = mini_batch['obs_prime'].to(self.device)
            done = mini_batch['done'].to(self.device)
            
            with torch.no_grad():
                next_action = self.target_actor(obs_prime)
                next_value = self.target_critic(obs_prime, next_action)
                td_target = r + self.gamma * next_value * (1 - done)
                # td_target = r + self.gamma * self.target_critic(obs_prime, self.target_actor(obs_prime)) * (1 - done)

            # Weights update of value network (gradient descent)
            critic_loss = F.smooth_l1_loss(self.critic(obs, a), td_target.detach())

            if torch.any(torch.isnan(critic_loss)):
                print(td_target.detach())
                print("-----dataset-----")
                print(torch.any(torch.isnan(r)))
                print(r)


            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
            self.critic_optimizer.step()
            
            # Weights update of policy network (gradient ascent)
            actor_loss = -self.critic(obs, self.actor(obs)).mean() # .mean() -> batch mean
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
            self.actor_optimizer.step()

            # Target update
            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            return critic_loss.data.item(), actor_loss.data.item()

    def save(self, save_path, timestep):
        filename_actor = os.path.join(save_path, "actor_{}k.pkl".format(timestep))
        # filename_critic = os.path.join(save_path, "critic_{}k.pkl".format(timestep))
        torch.save(self.actor.state_dict(), filename_actor)
        # torch.save(self.critic.state_dict(), filename_critic)

    def load(self, load_nn):
        self.actor.load_state_dict(torch.load(load_nn))
        # self.critic.load_state_dict(torch.load(load_nn_critic))

class Trainer:
    def __init__(self,
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
                 save_dir=None):
        
        self.model = model
        self.env = env
        self.max_training_timesteps = max_training_timesteps
        self.max_episode_steps = max_episode_steps
        self.evaluation_time_steps = evaluation_time_steps
        self.evaluation_times = evaluation_times
        self.training_start = training_start
        self.save_dir = os.path.join(save_dir, "model", "batch_{}_ddpg".format(batch_size))
        # self.save_dir = os.path.join(save_dir, "model", "ddpg".format(batch_size))
        self.action_dim = action_dim
        self.max_action = max_action
        self.expl_noise = expl_noise
        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, action_dim=action_dim, memory_capacity=memory_capacity, batch_size=batch_size)

        # Tensorboard results
        self.writer = SummaryWriter(log_dir="runs/single/batch_{}_ddpg/".format(batch_size))
        # self.writer = SummaryWriter(log_dir="runs/single/ddpg/".format(batch_size))

        # Main agent id
        self.main = 2

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
                    action = self.model.select_action(np.array(obs))

                obs_prime, reward, done, _ = self.env.step(action)
                self.replay_buffer.store(obs, action, reward, obs_prime, done)
                obs = obs_prime

                score += reward[0] # Record episodic reward
                if done:
                    break

                if time_step > self.training_start:
                    critic_loss, actor_loss = self.model.train(self.replay_buffer)
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

        if render:
            self.env.disconnectUnity()


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
    #             pseudo_action = np.concatenate((pseudo_action, action), axis=0)

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
    #                 action = self.model.select_action(np.array(obs)).reshape(1, -1).astype(np.float32)


    #             pseudo_action = np.array([[0.0, 0.0, 0.0, 0.0],
    #                                       [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    #             pseudo_action = np.concatenate((pseudo_action, action), axis=0)


    #             obs_prime, reward, done, _ = self.env.step(pseudo_action)

    #             obs_prime = obs_prime[self.main]
    #             reward = reward[self.main]
    #             done = done[self.main]

    #             self.replay_buffer.store(obs, action, reward, obs_prime, done)
    #             obs = obs_prime

    #             score += reward # Record episodic reward
    #             if done:
    #                 break

    #             if time_step > self.training_start:
    #                 critic_loss, actor_loss = self.model.train(self.replay_buffer)
    #                 self.writer.add_scalar("critic_loss", critic_loss, global_step=time_step)
    #                 self.writer.add_scalar("actor_loss", -actor_loss, global_step=time_step)                    

    #             if time_step % self.evaluation_time_steps == 0:
    #                 avg_reward = self.evaluate_policy(self.env, self.model, self.max_episode_steps, self.evaluation_times)
    #                 if best_score == None or avg_reward > best_score:
    #                     best_score = avg_reward
    #                     best_timestep = int(time_step/1000)

    #                 print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {} \t\t Best timestep : {}k".format(n_episode, time_step, avg_reward, best_timestep))
    #                 self.save(self.save_dir, int(time_step/1000))

    #         self.writer.add_scalar("score", score, global_step=time_step) # Save episodic reward

    #     tqdm_bar.close()
    #     self.env.close()
    #     self.writer.flush()
    #     self.writer.close()

    #     if render:
    #         self.env.disconnectUnity()



    def save(self, save_dir=None, timestep=None):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.model.save(save_dir, timestep)