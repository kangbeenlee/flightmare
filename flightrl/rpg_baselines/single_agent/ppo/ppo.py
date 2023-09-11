import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
import numpy as np
import os

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter



class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, device, obs_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()

        self.device = device
        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # actor
        self.actor = nn.Sequential(
                        nn.Linear(obs_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, action_dim),
                        nn.Tanh())
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(obs_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        
    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):
        action_mean = self.actor(state)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)
        
        # For Single Action Environments.
        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

class PPO:
    def __init__(self,
                 device=None,
                 gamma=None,
                 obs_dim=None,
                 action_dim=None,
                 lr_actor=None,
                 lr_critic=None,
                 K_epochs=None,
                 eps_clip=None,
                 action_std_init=0.6):
        
        self.device = device
        self.gamma = gamma
        self.action_std = action_std_init
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(device, obs_dim, action_dim, action_std_init).to(device)
        self.optimizer = optim.Adam([{'params': self.policy.actor.parameters(), 'lr': lr_actor},
                                     {'params': self.policy.critic.parameters(), 'lr': lr_critic}])
        self.policy_old = ActorCritic(device, obs_dim, action_dim, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
            print("setting actor output action_std to min_action_std : ", self.action_std)
        else:
            print("setting actor output action_std to : ", self.action_std)
        self.set_action_std(self.action_std)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).float().to(self.device)
            action, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.detach().cpu().numpy().astype(np.float32)

    def train(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        #
        avg_loss = 0.0

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = (-torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy).mean()
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            avg_loss += loss.data.item()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        
        return avg_loss / self.K_epochs
    
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
                 max_training_timesteps=None,
                 max_episode_steps=None,
                 evaluation_time_steps=None,
                 update_timestep=None,
                 action_std_decay_freq=None,
                 action_std_decay_rate=None,
                 min_action_std=None,
                 save_dir=None):
        
        self.model = model
        self.env = env
        self.max_training_timesteps = max_training_timesteps
        self.max_episode_steps = max_episode_steps
        self.evaluation_time_steps = evaluation_time_steps
        self.update_timestep = update_timestep
        self.action_std_decay_freq = action_std_decay_freq
        self.action_std_decay_rate = action_std_decay_rate
        self.min_action_std = min_action_std
        self.save_dir = os.path.join(save_dir, "saved", "ppo")
        # Tensorboard results
        self.writer = SummaryWriter(log_dir="runs/ppo/")

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
            obs, epi_step, score = self.env.reset(), 0, 0.0
            
            while not epi_step > self.max_episode_steps:
                tqdm_bar.update(1)
                time_step += 1
                epi_step += 1

                action = self.model.select_action(obs)
                obs, reward, done, _ = self.env.step(action)

                # saving reward and is_terminals
                self.model.buffer.rewards.append(reward[0])
                self.model.buffer.is_terminals.append(done)

                # update PPO agent
                if time_step % self.update_timestep == 0:
                    loss = self.model.train()
                    self.writer.add_scalar("ppo_loss", -loss, global_step=time_step)
                    
                # if continuous action space; then decay action std of ouput action distribution
                if time_step % self.action_std_decay_freq == 0:
                    self.model.decay_action_std(self.action_std_decay_rate, self.min_action_std)

                accumulative_reward += reward[0] # Just for single agent
                score += reward[0] # Record episodic reward
                if done:
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