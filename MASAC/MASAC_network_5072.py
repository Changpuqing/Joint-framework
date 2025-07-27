import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.max_action = args.max_action
        self.l1 = nn.Linear(args.obs_dim_n[agent_id], args.hidden_dim1)
        self.l2 = nn.Linear(args.hidden_dim1, args.hidden_dim2)
        self.mean_layer = nn.Linear(args.hidden_dim2, args.action_dim_n[agent_id])
        self.log_std_layer = nn.Linear(args.hidden_dim2, args.action_dim_n[agent_id])

    def forward(self, x, deterministic=False, with_logprob=True):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)  # We output the log_std to ensure that std=exp(log_std)>0
        log_std = torch.clamp(log_std, -1, 1)
        std = torch.exp(log_std)

        dist = Normal(mean, std)  # Generate a Gaussian distribution
        if deterministic:  # When evaluating，we use the deterministic policy
            a = mean
        else:
            a = dist.rsample()  # reparameterization trick: mean+std*N(0,1)

        if with_logprob:  # The method refers to Open AI Spinning up, which is more stable.
            log_pi = dist.log_prob(a).sum(dim=1, keepdim=True)
            log_pi -= (2 * (np.log(2) - a - F.softplus(-2 * a))).sum(dim=1, keepdim=True)
        else:
            log_pi = None

        a = (self.max_action * torch.sigmoid(a)).clip(0.1, self.max_action)  # Use tanh to compress the unbounded Gaussian distribution into a bounded action interval.

        return a, log_pi

class Critic(nn.Module):  # According to (s,a), directly calculate Q(s,a)
    def __init__(self, args):
        super(Critic, self).__init__()
        # Q1
        self.l1 = nn.Linear(sum(args.obs_dim_n) + sum(args.action_dim_n), args.hidden_dim1) #直接用args就行
        self.l2 = nn.Linear(args.hidden_dim1, args.hidden_dim2)
        self.l3 = nn.Linear(args.hidden_dim2, 1)
        # Q2
        self.l4 = nn.Linear(sum(args.obs_dim_n) + sum(args.action_dim_n), args.hidden_dim1)
        self.l5 = nn.Linear(args.hidden_dim1, args.hidden_dim2)
        self.l6 = nn.Linear(args.hidden_dim2, 1)

    def forward(self, s, a):
        s = torch.cat(s, dim=1)
        a = torch.cat(a, dim=1)
        s_a = torch.cat([s, a], dim=1)  # (1024,18)==有错误
        # s_a = torch.cat([s, a], 1)
        q1 = F.relu(self.l1(s_a))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(s_a))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2