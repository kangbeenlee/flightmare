import torch
import numpy as np



class ReplayBuffer(object):
    def __init__(self, args):
        self.device = args.device
        self.N = args.N  # The number of agents
        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.ptr = 0
        self.current_size = 0

        # Initialize buffer to store transitions.
        self.buffer = {
            'obs_n': np.empty([self.N, self.buffer_size, self.obs_dim], dtype=np.float32),
            'a_n': np.empty([self.N, self.buffer_size, self.action_dim], dtype=np.float32),
            'r_n': np.empty([self.N, self.buffer_size, 1], dtype=np.float32),
            'obs_next_n': np.empty([self.N, self.buffer_size, self.obs_dim], dtype=np.float32),
            'done_n': np.empty([self.N, self.buffer_size, 1], dtype=np.float32)
        }

    def store_transition(self, obs_n, a_n, r_n, obs_next_n, done_n):
        for agent_id in range(self.N):
            self.buffer['obs_n'][agent_id][self.ptr] = obs_n[agent_id]
            self.buffer['a_n'][agent_id][self.ptr] = a_n[agent_id]
            self.buffer['r_n'][agent_id][self.ptr] = r_n[agent_id]
            self.buffer['obs_next_n'][agent_id][self.ptr] = obs_next_n[agent_id]
            self.buffer['done_n'][agent_id][self.ptr] = done_n[agent_id]

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(self):
        index = np.random.choice(self.current_size, size=self.batch_size, replace=False)
        batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = [], [], [], [], []

        for agent_id in range(self.N):
            batch_obs_n.append(torch.tensor(self.buffer['obs_n'][agent_id][index], dtype=torch.float32).to(self.device))
            batch_a_n.append(torch.tensor(self.buffer['a_n'][agent_id][index], dtype=torch.float32).to(self.device))
            batch_r_n.append(torch.tensor(self.buffer['r_n'][agent_id][index], dtype=torch.float32).to(self.device))
            batch_obs_next_n.append(torch.tensor(self.buffer['obs_next_n'][agent_id][index], dtype=torch.float32).to(self.device))
            batch_done_n.append(torch.tensor(self.buffer['done_n'][agent_id][index], dtype=torch.float32).to(self.device))

        return batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n