import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
import numpy as np
import copy
import os, shutil
from normalization import Normalization, RewardScaling  # Trick 中的标准化
from MASAC_network_5072 import Actor, Critic

class MASAC(object):
    def __init__(self, args, agent_id):
        self.agent_id = agent_id
        self.max_action = args.max_action
        self.hidden_width = 256  # The number of neurons in hidden layers of the neural network
        self.batch_size = 256  # batch size
        self.N = args.N
        self.GAMMA = 0.99  # discount factor
        self.TAU = 0.005  # Softly update the target network
        self.lr = 3e-4  # learning rate
        self.adaptive_alpha = True  # Whether to automatically learn the temperature alpha
        self.use_grad_clip = args.use_grad_clip
        if self.adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = -args.action_dim_n[self.agent_id]
            # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)
            if agent_id == 0:
                # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
                self.log_alpha_0 = torch.zeros(1, requires_grad=True)
                self.alpha0 = self.log_alpha_0.exp()
                self.alpha_optimizer_0 = torch.optim.Adam([self.log_alpha_0], lr=self.lr)
            elif agent_id == 1:
                self.log_alpha_1 = torch.zeros(1, requires_grad=True)
                self.alpha1 = self.log_alpha_1.exp()
                self.alpha_optimizer_1 = torch.optim.Adam([self.log_alpha_1], lr=self.lr)
            elif agent_id == 2:
                self.log_alpha_2 = torch.zeros(1, requires_grad=True)
                self.alpha2 = self.log_alpha_2.exp()
                self.alpha_optimizer_2 = torch.optim.Adam([self.log_alpha_2], lr=self.lr)
            elif agent_id == 3:
                self.log_alpha_3 = torch.zeros(1, requires_grad=True)
                self.alpha3 = self.log_alpha_3.exp()
                self.alpha_optimizer_3 = torch.optim.Adam([self.log_alpha_3], lr=self.lr)
            elif agent_id == 4:
                self.log_alpha_4 = torch.zeros(1, requires_grad=True)
                self.alpha4 = self.log_alpha_4.exp()
                self.alpha_optimizer_4 = torch.optim.Adam([self.log_alpha_4], lr=self.lr)
        else:
            self.alpha0 = args.alpha0
            self.alpha1 = args.alpha1
            self.alpha2 = args.alpha2
            self.alpha3 = args.alpha3
            self.alpha4 = args.alpha4
            # self.alpha = 0.2

        self.actor = self.actor = Actor(args, agent_id)
        self.critic = Critic(args)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr) #这里李志的MATD3代码里面没按智能体分开
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def choose_action(self, s, deterministic=False):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a, _ = self.actor(s, deterministic, False)  # When choosing actions, we do not need to compute log_pi
        return a.data.numpy().flatten()

    def learn(self, replay_buffer, agent_n):
        batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = replay_buffer.sample()  # Sample a batch

        with torch.no_grad():
            batch_a_next_n = []
            log_pi_ = []
            for i in range(self.N):
                batch_a_next, log_pi_value = agent_n[i].actor(batch_obs_next_n[i])
                batch_a_next_n.append(batch_a_next)
                log_pi_.append(log_pi_value)  # a' from the current policy
            # Compute target Q
            target_Q1, target_Q2 = self.critic_target(batch_obs_next_n, batch_a_next_n)
            # target_Q = batch_r + self.GAMMA * (1 - batch_dw) * (torch.min(target_Q1, target_Q2) - self.alpha * log_pi_)
            if self.agent_id == 0:
                target_Q = batch_r_n[self.agent_id] + self.GAMMA * (1 - batch_done_n[self.agent_id]) * (torch.min(target_Q1, target_Q2) - self.alpha0 * log_pi_[self.agent_id])  # shape:(batch_size,1)
            elif self.agent_id == 1:
                target_Q = batch_r_n[self.agent_id] + self.GAMMA * (1 - batch_done_n[self.agent_id]) * (torch.min(target_Q1, target_Q2) - self.alpha1 * log_pi_[self.agent_id])  # shape:(batch_size,1)
            elif self.agent_id == 2:
                target_Q = batch_r_n[self.agent_id] + self.GAMMA * (1 - batch_done_n[self.agent_id]) * (torch.min(target_Q1, target_Q2) - self.alpha2 * log_pi_[self.agent_id])
            elif self.agent_id == 3:
                target_Q = batch_r_n[self.agent_id] + self.GAMMA * (1 - batch_done_n[self.agent_id]) * (torch.min(target_Q1, target_Q2) - self.alpha3 * log_pi_[self.agent_id])
            elif self.agent_id == 4:
                target_Q = batch_r_n[self.agent_id] + self.GAMMA * (1 - batch_done_n[self.agent_id]) * (torch.min(target_Q1, target_Q2) - self.alpha4 * log_pi_[self.agent_id])


        # Compute current Q
        current_Q1, current_Q2 = self.critic(batch_obs_n, batch_a_n)
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) #这里应该不用分开
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        # Freeze critic networks so you don't waste computational effort
        for params in self.critic.parameters():
            params.requires_grad = False

        # Compute actor loss
        batch_a_n[self.agent_id], log_pi = self.actor(batch_obs_n[self.agent_id])
        Q1, Q2 = self.critic(batch_obs_n, batch_a_n)
        Q = torch.min(Q1, Q2)
        # actor_loss = (self.alpha * log_pi - Q).mean()
        if self.agent_id == 0:
            actor_loss = (self.alpha0 * log_pi - Q).mean()
        elif self.agent_id == 1:
            actor_loss = (self.alpha1 * log_pi - Q).mean()
        elif self.agent_id == 2:
            actor_loss = (self.alpha2 * log_pi - Q).mean()
        elif self.agent_id == 3:
            actor_loss = (self.alpha3 * log_pi - Q).mean()
        elif self.agent_id == 4:
            actor_loss = (self.alpha4 * log_pi - Q).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True

        # Update alpha
        # if self.adaptive_alpha:
        #     # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
        #     alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()
        #     self.alpha_optimizer.zero_grad()
        #     alpha_loss.backward()
        #     self.alpha_optimizer.step()
        #     self.alpha = self.log_alpha.exp()

        if self.adaptive_alpha:
            if self.agent_id == 0:
                # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
                alpha_loss_0 = -(self.log_alpha_0.exp() * (log_pi + self.target_entropy).detach()).mean() #看看这里的熵对吗
                self.alpha_optimizer_0.zero_grad()
                alpha_loss_0.backward()
                self.alpha_optimizer_0.step()
                self.alpha0 = self.log_alpha_0.exp()
            elif self.agent_id == 1:
                alpha_loss_1 = -(self.log_alpha_1.exp() * (log_pi + self.target_entropy).detach()).mean()  # 看看这里的熵对吗
                self.alpha_optimizer_1.zero_grad()
                alpha_loss_1.backward()
                self.alpha_optimizer_1.step()
                self.alpha1 = self.log_alpha_1.exp()
            elif self.agent_id == 2:
                alpha_loss_2 = -(self.log_alpha_2.exp() * (log_pi + self.target_entropy).detach()).mean()  # 看看这里的熵对吗
                self.alpha_optimizer_2.zero_grad()
                alpha_loss_2.backward()
                self.alpha_optimizer_2.step()
                self.alpha2 = self.log_alpha_2.exp()
            elif self.agent_id == 3:
                alpha_loss_3 = -(self.log_alpha_3.exp() * (log_pi + self.target_entropy).detach()).mean()  # 看看这里的熵对吗
                self.alpha_optimizer_3.zero_grad()
                alpha_loss_3.backward()
                self.alpha_optimizer_3.step()
                self.alpha3 = self.log_alpha_3.exp()
            elif self.agent_id == 4:
                alpha_loss_4 = -(self.log_alpha_4.exp() * (log_pi + self.target_entropy).detach()).mean()  # 看看这里的熵对吗
                self.alpha_optimizer_4.zero_grad()
                alpha_loss_4.backward()
                self.alpha_optimizer_4.step()
                self.alpha4 = self.log_alpha_4.exp()

        # Softly update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)