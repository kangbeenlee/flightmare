# import torch
# import numpy as np



# class ReplayBuffer(object):
#     def __init__(self, args):
#         self.device = args.device
#         self.N = args.N
#         self.obs_dim = args.obs_dim
#         self.action_dim = args.action_dim
#         self.buffer_size = args.buffer_size
#         self.batch_size = args.batch_size
#         self.ptr = 0
#         self.current_size = 0
        
#         # Initialize buffer to store transitions.
#         self.buffer = {
#             'obs_n': np.empty([self.buffer_size, self.N, self.obs_dim], dtype=np.float32),
#             'a_n': np.empty([self.buffer_size, self.N, self.action_dim], dtype=np.float32),
#             'r_n': np.empty([self.buffer_size, self.N, 1], dtype=np.float32),
#             'obs_next_n': np.empty([self.buffer_size, self.N, self.obs_dim], dtype=np.float32),
#             'done_n': np.empty([self.buffer_size, self.N, 1], dtype=np.float32)
#         }

#     def store_transition(self, obs_n, a_n, r_n, obs_next_n, done_n):
#         if len(r_n.shape) == 1:
#             r_n = np.array(r_n).reshape(-1, 1)
#         if len(done_n.shape) == 1:
#             done_n = np.array(done_n).reshape(-1, 1)

#         self.buffer['obs_n'][self.ptr] = obs_n
#         self.buffer['a_n'][self.ptr] = a_n
#         self.buffer['r_n'][self.ptr] = r_n
#         self.buffer['obs_next_n'][self.ptr] = obs_next_n
#         self.buffer['done_n'][self.ptr] = done_n

#         self.ptr = (self.ptr + 1) % self.buffer_size
#         self.current_size = min(self.current_size + 1, self.buffer_size)

#     def sample(self):
#         index = np.random.choice(self.current_size, size=self.batch_size, replace=False)

#         # Batch-fetch transitions from buffer.
#         mini_batch = {key: torch.tensor(value[index], dtype=torch.float32).to(self.device) for key, value in self.buffer.items()}
        
#         return mini_batch['obs_n'], mini_batch['a_n'], mini_batch['r_n'], mini_batch['obs_next_n'], mini_batch['done_n']
    
#     def __len__(self):
#         return self.current_size