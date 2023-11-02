import torch
import torch.nn as nn
import torch.nn.functional as F



def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.max_action = args.max_action
        self.fc1 = nn.Linear(args.obs_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.action_dim)
        self.activation = nn.Tanh()
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        a = self.max_action * torch.tanh(self.fc3(x))
        return a


class Critic_MADDPG(nn.Module):
    def __init__(self, args):
        super(Critic_MADDPG, self).__init__()
        self.fc1 = nn.Linear(args.critic_input_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)
        self.activation = nn.Tanh()
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s_n, a_n):
        # s_n.shape=(batch, N*obs_dim)
        # a_n.shape=(batch, N*action_dim)
        s_a_n = torch.cat([s_n, a_n], dim=-1) # s_a_n.shape=(batch, N*(obs_dim + action_dim))
        q = self.activation(self.fc1(s_a_n))
        q = self.activation(self.fc2(q))
        q = self.fc3(q)
        return q


class Critic_MATD3(nn.Module):
    def __init__(self, args):
        super(Critic_MATD3, self).__init__()
        self.fc1 = nn.Linear(args.critic_input_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

        self.fc4 = nn.Linear(args.critic_input_dim, args.hidden_dim)
        self.fc5 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc6 = nn.Linear(args.hidden_dim, 1)

        self.activation = nn.Tanh()
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)
            orthogonal_init(self.fc4)
            orthogonal_init(self.fc5)
            orthogonal_init(self.fc6)

    def forward(self, s_n, a_n):
        # s_n.shape=(batch, N*obs_dim)
        # a_n.shape=(batch, N*action_dim)
        s_a_n = torch.cat([s_n, a_n], dim=-1) # s_a_n.shape=(batch, N*(obs_dim + action_dim))
        q1 = self.activation(self.fc1(s_a_n))
        q1 = self.activation(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = self.activation(self.fc4(s_a_n))
        q2 = self.activation(self.fc5(q2))
        q2 = self.fc6(q2)
        return q1, q2

    def Q1(self, s_n, a_n):
        s_a_n = torch.cat([s_n, a_n], dim=-1)
        q1 = self.activation(self.fc1(s_a_n))
        q1 = self.activation(self.fc2(q1))
        q1 = self.fc3(q1)
        return q1