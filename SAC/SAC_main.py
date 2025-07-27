import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
from normalization import Normalization
import os, shutil
from env_516 import EnvMove
import utils_516_SAC
import argparse

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.mean_layer = nn.Linear(hidden_width, action_dim)
        self.log_std_layer = nn.Linear(hidden_width, action_dim)

    def forward(self, x, deterministic=False, with_logprob=True):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)  # We output the log_std to ensure that std=exp(log_std)>0
        log_std = torch.clamp(log_std, -20, 2)
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

        a = (self.max_action * torch.sigmoid(a)).clip(0.05, self.max_action)  # Use tanh to compress the unbounded Gaussian distribution into a bounded action interval.

        return a, log_pi


class Critic(nn.Module):  # According to (s,a), directly calculate Q(s,a)
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Critic, self).__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, 1)
        # Q2
        self.l4 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l5 = nn.Linear(hidden_width, hidden_width)
        self.l6 = nn.Linear(hidden_width, 1)

    def forward(self, s, a):
        s_a = torch.cat([s, a], 1)
        q1 = F.relu(self.l1(s_a))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(s_a))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim):
        self.max_size = int(1e6)
        self.count = 0
        self.size = 0
        self.s = np.zeros((self.max_size, state_dim))
        self.a = np.zeros((self.max_size, action_dim))
        self.r = np.zeros((self.max_size, 1))
        self.s_ = np.zeros((self.max_size, state_dim))
        self.dw = np.zeros((self.max_size, 1))

    def store(self, s, a, r, s_, dw):
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.count = (self.count + 1) % self.max_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.size = min(self.size + 1, self.max_size)  # Record the number of  transitions

    def sample(self, batch_size):
        index = np.random.choice(self.size, size=batch_size)  # Randomly sampling
        batch_s = torch.tensor(self.s[index], dtype=torch.float)
        batch_a = torch.tensor(self.a[index], dtype=torch.float)
        batch_r = torch.tensor(self.r[index], dtype=torch.float)
        batch_s_ = torch.tensor(self.s_[index], dtype=torch.float)
        batch_dw = torch.tensor(self.dw[index], dtype=torch.float)

        return batch_s, batch_a, batch_r, batch_s_, batch_dw


class SAC(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.max_action = max_action
        self.hidden_width = 256  # The number of neurons in hidden layers of the neural network
        self.batch_size = 256  # batch size
        self.GAMMA = 0.99  # discount factor
        self.TAU = 0.005  # Softly update the target network
        self.lr = 3e-4  # learning rate
        self.adaptive_alpha = True  # Whether to automatically learn the temperature alpha
        if self.adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = -action_dim
            # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)
        else:
            self.alpha = 0.2

        self.actor = Actor(state_dim, action_dim, self.hidden_width, max_action)
        self.critic = Critic(state_dim, action_dim, self.hidden_width)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def choose_action(self, s, deterministic=False):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a, _ = self.actor(s, deterministic, False)  # When choosing actions, we do not need to compute log_pi
        return a.data.numpy().flatten()

    def learn(self, relay_buffer):
        batch_s, batch_a, batch_r, batch_s_, batch_dw = relay_buffer.sample(self.batch_size)  # Sample a batch

        with torch.no_grad():
            batch_a_, log_pi_ = self.actor(batch_s_)  # a' from the current policy
            # Compute target Q
            target_Q1, target_Q2 = self.critic_target(batch_s_, batch_a_)
            target_Q = batch_r + self.GAMMA * (1 - batch_dw) * (torch.min(target_Q1, target_Q2) - self.alpha * log_pi_)

        # Compute current Q
        current_Q1, current_Q2 = self.critic(batch_s, batch_a)
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Freeze critic networks so you don't waste computational effort
        for params in self.critic.parameters():
            params.requires_grad = False

        # Compute actor loss
        a, log_pi = self.actor(batch_s)
        Q1, Q2 = self.critic(batch_s, a)
        Q = torch.min(Q1, Q2)
        actor_loss = (self.alpha * log_pi - Q).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True

        # Update alpha
        if self.adaptive_alpha:
            # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
            alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        # Softly update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)


def evaluate_policy(env, agent):
    times = 3  # Perform three evaluations and calculate the average
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        done = False
        episode_reward = 0
        while not done:
            a = agent.choose_action(s, deterministic=True)  # We use the deterministic policy during the evaluating
            s_, r, done, _ = env.step(a)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return int(evaluate_reward / times)


def reward_adapter(r, env_index):
    if env_index == 0:  # Pendulum-v1
        r = (r + 8) / 8
    elif env_index == 1:  # BipedalWalker-v3
        if r <= -100:
            r = -1
    return r



class Runner:
    def __init__(self, args, number, seed):
        self.args = args
        self.seed = seed
        self.number = number
        self.args.N = 5  # The number of agents
        self.args.obs_dim_n = 3  # obs dimensions of N agents
        self.args.action_dim_n = 20  # actions dimensions of N agents

        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Create a tensorboard
        # self.writer = SummaryWriter(log_dir='runs/{}/{}_env_{}_number_{}_seed_{}'.format(self.args.algorithm, self.args.algorithm, self.env_name, self.number, self.seed))

        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0
        self.noise_std = self.args.noise_std_init  # Initialize noise_std

    def run(self, ):
    # env_name = ['BipedalWalker-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2']
    # env_index = 0
    # env = gym.make(env_name[env_index])
    # env_evaluate = gym.make(env_name[env_index])  # When evaluating the policy, we need to rebuild an environment
    # number = 1
    # seed = 0
    # # Set random seed
    # env.seed(seed)
    # # env.action_space.seed(seed)
    # env_evaluate.seed(seed)
    # # env_evaluate.action_space.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    #
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape[0]
    # max_action = float(env.action_space.high[0])
    # max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    # print("env={}".format(env_name[env_index]))
    # print("state_dim={}".format(state_dim))
    # print("action_dim={}".format(action_dim))
    # print("max_action={}".format(max_action))
    # print("max_episode_steps={}".format(max_episode_steps))
    #
    # agent = SAC(state_dim, action_dim, max_action)
    # replay_buffer = ReplayBuffer(state_dim, action_dim)
    # # Build a tensorboard
    # writer = SummaryWriter(log_dir='runs/SAC/SAC_env_{}_number_{}_seed_{}'.format(env_name[env_index], number, seed))
    #
    # max_train_steps = 3e6  # Maximum number of training steps
    # random_steps = 25e3  # Take the random actions in the beginning for the better exploration
    # evaluate_freq = 5e3  # Evaluate the policy every 'evaluate_freq' steps
    # evaluate_num = 0  # Record the number of evaluations
    # evaluate_rewards = []  # Record the rewards during the evaluating
    # total_steps = 0  # Record the total steps during the training
    #
    # while total_steps < max_train_steps:
    #     s = env.reset()
    #     episode_steps = 0
    #     done = False
    #     while not done:
    #         episode_steps += 1
    #         if total_steps < random_steps:  # Take the random actions in the beginning for the better exploration
    #             a = env.action_space.sample()
    #         else:
    #             a = agent.choose_action(s)
    #         s_, r, done, _ = env.step(a)
    #         r = reward_adapter(r, env_index)  # Adjust rewards for better performance
    #         # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
    #         # dw means dead or win,there is no next state s';
    #         # but when reaching the max_episode_steps,there is a next state s' actually.
    #         if done and episode_steps != max_episode_steps:
    #             dw = True
    #         else:
    #             dw = False
    #         replay_buffer.store(s, a, r, s_, dw)  # Store the transition
    #         s = s_
    #
    #         if total_steps >= random_steps:
    #             agent.learn(replay_buffer)
    #
    #         # Evaluate the policy every 'evaluate_freq' steps
    #
    #         total_steps += 1

    # max_train_steps = 6000
    # LEARNING_WINDOW = 2000
    # BAND_MBS = 40
    # BAND_SBS = 10
    # COMP_MBS = 30
    # COMP_SBS = 20


        writepath = 'SAC_main_519'
        if os.path.exists(writepath): shutil.rmtree(writepath)
        if not os.path.exists(writepath): os.mkdir(writepath)
        writer = SummaryWriter(log_dir=writepath)

        agent = SAC(3, 20, 0.95)

        replay_buffer = ReplayBuffer(3, 20)
        state_norm = Normalization(shape=3)
        latency_average_lst, tp_lst, = [], []  #
        qoe_lst = []
        reward_lst = []
        utility_lst = []
        utility_set_lst = []
        qoe_ave_t_lst = []
        action_lst = []
        latency_average_t_lst = []
        pkt_jiayou = []
        env = EnvMove()
        env.countReset()  # 回头仔细想想别漏了
        env.ue_position()
        a_n = np.random.uniform(0.05, 0.95, 20)
        env.evolutionary_init(args)
        env.activity()

        env.band_ser_cat[1] = np.round(a_n[0] * args.BAND_MBS * 10 ** 6)  # 采取动作
        env.band_ser_cat[0] = np.round(((1 - a_n[0]) * a_n[1]) * args.BAND_MBS * 10 ** 6)  # 采取动作
        env.band_ser_cat[2] = np.round(((1 - a_n[0]) * (1 - a_n[1])) * args.BAND_MBS * 10 ** 6)
        env.fer_ser_cat[1] = np.round(a_n[2] * args.COMP_MBS * 10 ** 9)
        env.fer_ser_cat[0] = np.round(((1 - a_n[2]) * a_n[3]) * args.COMP_MBS * 10 ** 9)
        env.fer_ser_cat[2] = np.round(((1 - a_n[2]) * (1 - a_n[3])) * args.COMP_MBS * 10 ** 9)

        env.band_ser_cat[4] = np.round(a_n[4] * args.BAND_SBS * 10 ** 6)  # 采取动作
        env.band_ser_cat[3] = np.round(((1 - a_n[4]) * a_n[5]) * args.BAND_SBS * 10 ** 6)  # 采取动作
        env.band_ser_cat[5] = np.round(((1 - a_n[4]) * (1 - a_n[5])) * args.BAND_SBS * 10 ** 6)
        env.fer_ser_cat[4] = np.round(a_n[6] * args.COMP_SBS * 10 ** 9)
        env.fer_ser_cat[3] = np.round(((1 - a_n[6]) * a_n[7]) * args.COMP_SBS * 10 ** 9)
        env.fer_ser_cat[5] = np.round(((1 - a_n[6]) * (1 - a_n[7])) * args.COMP_SBS * 10 ** 9)

        env.band_ser_cat[7] = np.round(a_n[8] * args.BAND_SBS * 10 ** 6)  # 采取动作
        env.band_ser_cat[6] = np.round(((1 - a_n[8]) * a_n[9]) * args.BAND_SBS * 10 ** 6)  # 采取动作
        env.band_ser_cat[8] = np.round(((1 - a_n[8]) * (1 - a_n[9])) * args.BAND_SBS * 10 ** 6)
        env.fer_ser_cat[7] = np.round(a_n[10] * args.COMP_SBS * 10 ** 9)
        env.fer_ser_cat[6] = np.round(((1 - a_n[10]) * a_n[11]) * args.COMP_SBS * 10 ** 9)
        env.fer_ser_cat[8] = np.round(((1 - a_n[10]) * (1 - a_n[11])) * args.COMP_SBS * 10 ** 9)

        env.band_ser_cat[10] = np.round(a_n[12] * args.BAND_SBS * 10 ** 6)  # 采取动作
        env.band_ser_cat[9] = np.round(((1 - a_n[12]) * a_n[13]) * args.BAND_SBS * 10 ** 6)  # 采取动作
        env.band_ser_cat[11] = np.round(((1 - a_n[12]) * (1 - a_n[13])) * args.BAND_SBS * 10 ** 6)
        env.fer_ser_cat[10] = np.round(a_n[14] * args.COMP_SBS * 10 ** 9)
        env.fer_ser_cat[9] = np.round(((1 - a_n[14]) * a_n[15]) * args.COMP_SBS * 10 ** 9)
        env.fer_ser_cat[11] = np.round(((1 - a_n[14]) * (1 - a_n[15])) * args.COMP_SBS * 10 ** 9)

        env.band_ser_cat[13] = np.round(a_n[16] * args.BAND_SBS * 10 ** 6)  # 采取动作
        env.band_ser_cat[12] = np.round(((1 - a_n[16]) * a_n[17]) * args.BAND_SBS * 10 ** 6)  # 采取动作
        env.band_ser_cat[14] = np.round(((1 - a_n[16]) * (1 - a_n[17])) * args.BAND_SBS * 10 ** 6)
        env.fer_ser_cat[13] = np.round(a_n[18] * args.COMP_SBS * 10 ** 9)
        env.fer_ser_cat[12] = np.round(((1 - a_n[18]) * a_n[19]) * args.COMP_SBS * 10 ** 9)
        env.fer_ser_cat[14] = np.round(((1 - a_n[18]) * (1 - a_n[19])) * args.COMP_SBS * 10 ** 9)
        for i_subframe in range(args.LEARNING_WINDOW):  # 第 i 个子帧

            env.scheduling()  # 分配资源块
            env.provisioning()  # 计算数据速率以及清理buffer
            if i_subframe < args.LEARNING_WINDOW - 1:
                env.activity()
        pkt = env.get_state()
        observe = state_norm(pkt)
        for i_iter in range(args.max_train_steps):
            ue_list = []
            env.countReset()
            # env.ue_position()
            # env.user_move()

            env.activity()
            s = observe
            a_n = agent.choose_action(s)
            # for _ in range(5):
            #     a_n[_] = (a_n[_] + 1) / 2
            env.band_ser_cat[1] = np.round(a_n[0] * args.BAND_MBS * 10 ** 6)  # 采取动作
            env.band_ser_cat[0] = np.round(((1 - a_n[0]) * a_n[1]) * args.BAND_MBS * 10 ** 6)  # 采取动作
            env.band_ser_cat[2] = np.round(((1 - a_n[0]) * (1 - a_n[1])) * args.BAND_MBS * 10 ** 6)
            env.fer_ser_cat[1] = np.round(a_n[2] * args.COMP_MBS * 10 ** 9)
            env.fer_ser_cat[0] = np.round(((1 - a_n[2]) * a_n[3]) * args.COMP_MBS * 10 ** 9)
            env.fer_ser_cat[2] = np.round(((1 - a_n[2]) * (1 - a_n[3])) * args.COMP_MBS * 10 ** 9)

            env.band_ser_cat[4] = np.round(a_n[4] * args.BAND_SBS * 10 ** 6)  # 采取动作
            env.band_ser_cat[3] = np.round(((1 - a_n[4]) * a_n[5]) * args.BAND_SBS * 10 ** 6)  # 采取动作
            env.band_ser_cat[5] = np.round(((1 - a_n[4]) * (1 - a_n[5])) * args.BAND_SBS * 10 ** 6)
            env.fer_ser_cat[4] = np.round(a_n[6] * args.COMP_SBS * 10 ** 9)
            env.fer_ser_cat[3] = np.round(((1 - a_n[6]) * a_n[7]) * args.COMP_SBS * 10 ** 9)
            env.fer_ser_cat[5] = np.round(((1 - a_n[6]) * (1 - a_n[7])) * args.COMP_SBS * 10 ** 9)

            env.band_ser_cat[7] = np.round(a_n[8] * args.BAND_SBS * 10 ** 6)  # 采取动作
            env.band_ser_cat[6] = np.round(((1 - a_n[8]) * a_n[9]) * args.BAND_SBS * 10 ** 6)  # 采取动作
            env.band_ser_cat[8] = np.round(((1 - a_n[8]) * (1 - a_n[9])) * args.BAND_SBS * 10 ** 6)
            env.fer_ser_cat[7] = np.round(a_n[10] * args.COMP_SBS * 10 ** 9)
            env.fer_ser_cat[6] = np.round(((1 - a_n[10]) * a_n[11]) * args.COMP_SBS * 10 ** 9)
            env.fer_ser_cat[8] = np.round(((1 - a_n[10]) * (1 - a_n[11])) * args.COMP_SBS * 10 ** 9)

            env.band_ser_cat[10] = np.round(a_n[12] * args.BAND_SBS * 10 ** 6)  # 采取动作
            env.band_ser_cat[9] = np.round(((1 - a_n[12]) * a_n[13]) * args.BAND_SBS * 10 ** 6)  # 采取动作
            env.band_ser_cat[11] = np.round(((1 - a_n[12]) * (1 - a_n[13])) * args.BAND_SBS * 10 ** 6)
            env.fer_ser_cat[10] = np.round(a_n[14] * args.COMP_SBS * 10 ** 9)
            env.fer_ser_cat[9] = np.round(((1 - a_n[14]) * a_n[15]) * args.COMP_SBS * 10 ** 9)
            env.fer_ser_cat[11] = np.round(((1 - a_n[14]) * (1 - a_n[15])) * args.COMP_SBS * 10 ** 9)

            env.band_ser_cat[13] = np.round(a_n[16] * args.BAND_SBS * 10 ** 6)  # 采取动作
            env.band_ser_cat[12] = np.round(((1 - a_n[16]) * a_n[17]) * args.BAND_SBS * 10 ** 6)  # 采取动作
            env.band_ser_cat[14] = np.round(((1 - a_n[16]) * (1 - a_n[17])) * args.BAND_SBS * 10 ** 6)
            env.fer_ser_cat[13] = np.round(a_n[18] * args.COMP_SBS * 10 ** 9)
            env.fer_ser_cat[12] = np.round(((1 - a_n[18]) * a_n[19]) * args.COMP_SBS * 10 ** 9)
            env.fer_ser_cat[14] = np.round(((1 - a_n[18]) * (1 - a_n[19])) * args.COMP_SBS * 10 ** 9)

            for i_subframe in range(args.LEARNING_WINDOW):

                env.scheduling()  # 分配资源块
                env.provisioning()  # 计算数据速率以及清理buffer
                if i_subframe < args.LEARNING_WINDOW - 1:
                    env.activity()
            for i in range(self.args.N):
                UE_num = env.find_ue_num(i)
                ue_list.append(UE_num)
            pkt = env.get_state()

            # observe = [utilsthree.gen_states(pkt) for agent, pkt in zip(self.agent_n, pkt_total)]
            observe = state_norm(pkt)
            s_ = observe
            tp, qoe, latency_average = env.get_reward(args)
            latency_average_lst.append(latency_average)  # 本来就是列表，应该不需要tolist
            tp_lst.append(tp)
            qoe_lst.append(qoe)

            utility, reward = utils_516_SAC.calc__reward(latency_average, tp, qoe)
            reward_lst.append(reward)
            utility_lst.append(utility.tolist())

            print('\nStep-%d' % i_iter)
            print('uility: ', utility)
            print('latency: ', latency_average)
            print('QoE: ', qoe)
            print('throughput: ', tp)
            print('reward: ', reward)
            print('action:', a_n)
            print('UE_num: ', ue_list)
            print('pkt ', pkt)

            terminal = False
            done_n = False
            replay_buffer.store(s, a_n, reward, s_, done_n)  # Store the transition  # Store the transition
            self.total_steps += 1

            if self.total_steps > self.args.batch_size:
                agent.learn(replay_buffer)

            # if (i_iter + 1) % 500 == 0:
            #     with open(LOG_TRAIN, 'a+') as f:
            #         for i in range(len(tp_lst)):
            #             print(
            #                 'Reward: %.4f, utility: %4f, TP: %.4f,  QoS_embb: %.4f, QoS_urllc: %.4f, latency_ave: %.4f' % (
            #                     reward_lst[i], utility_lst[i], tp_lst[i], qoe_lst[i][0], qoe_lst[i][1],latency_average[i]),
            #                     file=f)

            writer.add_scalar('utility', utility, global_step=i_iter + 1)
            writer.add_scalar('QoS_0', qoe[0], global_step=i_iter + 1)
            writer.add_scalar('QoS_1', qoe[1], global_step=i_iter + 1)
            writer.add_scalar('QoS_2', qoe[2], global_step=i_iter + 1)
            writer.add_scalar('Reward', reward, global_step=i_iter + 1)
            writer.add_scalar('TP_total', tp, global_step=i_iter + 1)
            writer.add_scalar('LatencyaveM0', latency_average[0], global_step=i_iter + 1)
            writer.add_scalar('LatencyaveM1', latency_average[1], global_step=i_iter + 1)
            writer.add_scalar('LatencyaveM2', latency_average[2], global_step=i_iter + 1)
            writer.add_scalar('Action_M0', a_n[0], global_step=i_iter + 1)
            writer.add_scalar('Action_M1', a_n[1], global_step=i_iter + 1)
            writer.add_scalar('Action_M2', a_n[2], global_step=i_iter + 1)
            writer.add_scalar('Action_M3', a_n[3], global_step=i_iter + 1)

            if (i_iter + 1) % 10 == 0:
                # 12 项
                np.save('{}/{}.npy'.format(writepath, "Utility"), np.array(utility_lst))
                np.save('{}/{}.npy'.format(writepath, "Reward"), np.array(reward_lst))
                np.save('{}/{}.npy'.format(writepath, "ave_latency"), np.array(latency_average_lst))
                np.save('{}/{}.npy'.format(writepath, "TP"), np.array(tp_lst))
                np.save('{}/{}.npy'.format(writepath, "QoS"), np.array(qoe_lst))




if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MADDPG and MATD3 in MPE environment")
    parser.add_argument("--max_train_steps", type=int, default=int(6000), help=" Maximum number of training steps")
    # parser.add_argument("--episode_limit", type=int, default=25, help="Maximum number of steps per episode")
    # parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    # parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")
    parser.add_argument("--max_action", type=float, default=0.95, help="Max action")
    parser.add_argument("--algorithm", type=str, default="MATD3", help="MADDPG or MATD3")
    parser.add_argument("--buffer_size", type=int, default=int(6000), help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=128, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--hidden_dim1", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--noise_std_init", type=float, default=0.25, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_std_min", type=float, default=0.05, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_decay_steps", type=float, default=3000,help="How many steps before the noise_std decays to the minimum")
    parser.add_argument("--use_noise_decay", type=bool, default=True, help="Whether to decay the noise_std")
    parser.add_argument("--lr_a", type=float, default=0.001, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=0.0015, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Softly update the target network")
    parser.add_argument("--use_orthogonal_init", type=bool, default=False, help="Orthogonal initialization")
    parser.add_argument("--use_grad_clip", type=bool, default=False, help="Gradient clip")
    # --------------------------------------MATD3--------------------------------------------------------------------
    parser.add_argument("--epsilon", type=float, default=1.0, help="Target policy smoothing")
    parser.add_argument("--epsilon_init", type=float, default=1.0, help="Target policy smoothing")
    parser.add_argument("--epsilon_min", type=float, default=0, help="Target policy smoothing")
    parser.add_argument("--epsilon_step", type=float, default=1500, help="Target policy smoothing")
    parser.add_argument("--policy_noise", type=float, default=0.1, help="Target policy smoothing")
    parser.add_argument("--noise_clip", type=float, default=0.5, help="Clip noise")
    parser.add_argument("--policy_update_freq", type=int, default=2, help="The frequency of policy updates")

    parser.add_argument("--LEARNING_WINDOW", type=int, default=2000, help="the number of timeslot")
    parser.add_argument("--BAND_MBS", type=int, default=40, help="total band of MBS")
    parser.add_argument("--BAND_SBS", type=int, default=10, help="total band of SBS")
    parser.add_argument("--COMP_MBS", type=int, default=35, help="total band of SBS")
    parser.add_argument("--COMP_SBS", type=int, default=25, help="total band of SBS")
    # parser.add_argument("--CACH_MBS", type=int, default=5, help="total band of SBS")
    # parser.add_argument("--CACH_SBS", type=int, default=2, help="total band of SBS")
    parser.add_argument("--UE_NUMS", type=int, default=100, help="total band of SBS")
    parser.add_argument("--game_epsilon", type=float, default=0.9, help="total band of SBS")
    parser.add_argument("--adopt_mtr", type=float, default=0.05, help="total band of SBS")

    LOG_TRAIN = 'TD3 219.txt'

    args = parser.parse_args()
    args.noise_std_decay = (args.noise_std_init - args.noise_std_min) / args.noise_decay_steps

    runner = Runner(args, number=1, seed=7)
    runner.run()


