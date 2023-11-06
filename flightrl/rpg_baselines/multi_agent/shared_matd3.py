# import torch
# import torch.nn.functional as F
# import numpy as np
# import copy
# from rpg_baselines.multi_agent.networks import Actor,Critic_MATD3
# import os



# class MATD3(object):
#     def __init__(self, args):
#         self.device = args.device
#         self.N = args.N
#         self.max_action = args.max_action
#         self.action_dim = args.action_dim

#         self.lr_a = args.lr_a
#         self.lr_c = args.lr_c
#         self.gamma = args.gamma
#         self.tau = args.tau
#         self.use_grad_clip = args.use_grad_clip
#         self.policy_noise = args.policy_noise
#         self.noise_clip = args.noise_clip
#         self.policy_update_freq = args.policy_update_freq
#         self.actor_pointer = 0

#         # Create an individual actor and critic
#         self.actor = Actor(args).to(self.device)
#         self.critic = Critic_MATD3(args).to(self.device)
#         self.actor_target = copy.deepcopy(self.actor).to(self.device)
#         self.critic_target = copy.deepcopy(self.critic).to(self.device)

#         self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
#         self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

#     # Each agent selects actions based on its own local observations(add noise for exploration)
#     def choose_action(self, obs_n, noise_std):
#         obs_n = torch.tensor(obs_n, dtype=torch.float).to(self.device)
#         a_n = self.actor(obs_n).detach().cpu().numpy() # obs_n.shape=(N, obs_dim)
#         a_n = (a_n + np.random.normal(0, noise_std, size=a_n.shape)).clip(-self.max_action, self.max_action).astype(np.float32) # a_n.shape=(N, obs_dim)
#         return a_n

#     def train(self, replay_buffer):
#         with torch.autograd.set_detect_anomaly(True):
#             self.actor_pointer += 1
#             batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = replay_buffer.sample()

#             # Compute target_Q
#             with torch.no_grad():  # target_Q has no gradient
#                 # Trick 1:target policy smoothing
#                 batch_a_next_n = self.actor_target(batch_obs_next_n) # batch_a_next_n.shape=(batch, N, action_dims)
#                 noise = (torch.randn_like(batch_a_next_n) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip) # noise.shape=(batch, N, action_dims)
#                 batch_a_next_n = (batch_a_next_n + noise).clamp(-self.max_action, self.max_action)

#                 # Dimension convert: (batch, N, ?) -> (batch, N*?)
#                 batch_a_next_n_reshaped = batch_a_next_n.reshape(batch_a_next_n.size(0), -1) # batch_a_next_n_reshaped.shape=(batch, N*action_dims)
#                 batch_obs_next_n_reshaped = batch_obs_next_n.reshape(batch_obs_next_n.size(0), -1) # batch_obs_next_n_reshaped.shape=(batch, N*obs_dim)

#                 # Trick 2:clipped double Q-learning
#                 Q1_next, Q2_next = self.critic_target(batch_obs_next_n_reshaped, batch_a_next_n_reshaped) # Q1_next & Q2_next.shape=(batch_size, 1)
                
#                 # Dimension convert: (batch, 1) -> (batch, N, 1)
#                 Q1_next = Q1_next.unsqueeze(1).expand(-1, self.N, -1) 
#                 Q2_next = Q2_next.unsqueeze(1).expand(-1, self.N, -1)

#                 target_Q = batch_r_n + self.gamma * (1 - batch_done_n) * torch.min(Q1_next, Q2_next) # target_Q.shape:(batch_size, N, 1)

#             # Dimension convert: (batch, N, ?) -> (batch, N*?)
#             batch_a_n_reshaped = batch_a_n.reshape(batch_a_n.size(0), -1) # batch_a_next_n_reshaped.shape=(batch, N*action_dims)
#             batch_obs_n_reshaped = batch_obs_n.reshape(batch_obs_n.size(0), -1) # batch_obs_next_n_reshaped.shape=(batch, N*obs_dim)

#             # Compute current_Q
#             current_Q1, current_Q2 = self.critic(batch_obs_n_reshaped, batch_a_n_reshaped)  # current_Q1.shape:(batch_size, 1)

#             # Dimension convert: (batch, 1) -> (batch, N, 1)
#             current_Q1 = current_Q1.unsqueeze(1).expand(-1, self.N, -1)
#             current_Q2 = current_Q2.unsqueeze(1).expand(-1, self.N, -1)
            
#             critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
#             # Optimize the critic
#             self.critic_optimizer.zero_grad()
#             critic_loss.backward()
#             if self.use_grad_clip:
#                 torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
#             self.critic_optimizer.step()

#             # Trick 3:delayed policy updates
#             if self.actor_pointer % self.policy_update_freq == 0:
#                 # Reselect the actions of the agent
#                 batch_a_n = self.actor(batch_obs_n)

#                 # Dimension convert: (batch, N, ?) -> (batch, N*?)
#                 batch_a_n_reshaped = batch_a_n.reshape(batch_a_n.size(0), -1) # batch_a_next_n_reshaped.shape=(batch, N*action_dims)

#                 actor_loss = -self.critic.Q1(batch_obs_n_reshaped, batch_a_n_reshaped).mean()  # Only use Q1
#                 # Optimize the actor
#                 self.actor_optimizer.zero_grad()
#                 actor_loss.backward()
#                 if self.use_grad_clip:
#                     torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
#                 self.actor_optimizer.step()

#                 # Softly update the target networks
#                 for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
#                     target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

#                 for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
#                     target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

#     def save_model(self, path, total_steps):
#         path = os.path.join(path, "matd3_{}k.pth".format(int(total_steps / 1000)))
#         torch.save(self.actor.state_dict(), path)

#     def load_model(self, load_nn):
#         self.actor.load_state_dict(torch.load(load_nn))