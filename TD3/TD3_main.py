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
        self.l3 = nn.Linear(hidden_width, action_dim)

    def forward(self, s):
        s = F.relu(self.l1(s))
        s = F.relu(self.l2(s))
        a = self.max_action * torch.sigmoid(self.l3(s))  # [-max,max]
        return a


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

    def Q1(self, s, a):
        s_a = torch.cat([s, a], 1)
        q1 = F.relu(self.l1(s_a))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        return q1


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


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.max_action = max_action
        self.hidden_width = 256  # The number of neurons in hidden layers of the neural network
        self.batch_size = 256  # batch size
        self.GAMMA = 0.99  # discount factor
        self.TAU = 0.005  # Softly update the target network
        self.lr = 0.0001  # learning rate
        self.policy_noise = 0.2 * max_action  # The noise for the trick 'target policy smoothing'
        self.noise_clip = 0.5 * max_action  # Clip the noise
        self.policy_freq = 2  # The frequency of policy updates
        self.actor_pointer = 0

        self.actor = Actor(state_dim, action_dim, self.hidden_width, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(state_dim, action_dim, self.hidden_width)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a = self.actor(s).data.numpy().flatten()
        return a

    def learn(self, relay_buffer):
        self.actor_pointer += 1
        batch_s, batch_a, batch_r, batch_s_, batch_dw = relay_buffer.sample(self.batch_size)  # Sample a batch

        # Compute the target Q
        with torch.no_grad():  # target_Q has no gradient
            # Trick 1:target policy smoothing
            # torch.randn_like can generate random numbers sampled from N(0,1)，which have the same size as 'batch_a'
            noise = (torch.randn_like(batch_a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(batch_s_) + noise).clamp(-self.max_action, self.max_action)

            # Trick 2:clipped double Q-learning
            target_Q1, target_Q2 = self.critic_target(batch_s_, next_action)
            target_Q = batch_r + self.GAMMA * (1 - batch_dw) * torch.min(target_Q1, target_Q2)

        # Get the current Q
        current_Q1, current_Q2 = self.critic(batch_s, batch_a)
        # Compute the critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Trick 3:delayed policy updates
        if self.actor_pointer % self.policy_freq == 0:
            # Freeze critic networks so you don't waste computational effort
            for params in self.critic.parameters():
                params.requires_grad = False

            # Compute actor loss
            actor_loss = -self.critic.Q1(batch_s, self.actor(batch_s)).mean()  # Only use Q1
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Unfreeze critic networks
            for params in self.critic.parameters():
                params.requires_grad = True

            # Softly update the target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)


def evaluate_policy(env, agent):
    times = 3  # Perform three evaluations and calculate the average
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        done = False
        episode_reward = 0
        while not done:
            a = agent.choose_action(s)  # We do not add noise when evaluating
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

        writepath = 'TD3_main_911_10w'
        if os.path.exists(writepath): shutil.rmtree(writepath)
        if not os.path.exists(writepath): os.mkdir(writepath)
        writer = SummaryWriter(log_dir=writepath)
        max_action = 0.8
        agent = TD3(3, 15, 0.80)
        # noise_std = 0.1 * max_action  # the std of Gaussian noise for exploration

        replay_buffer = ReplayBuffer(3, 15)
        state_norm = Normalization(shape=3)
        latency_average_lst, energy_evergy_lst = [], []  #
        energy_que_con_lst = []
        reward_lst = []
        env = EnvMove()
        env.countReset()  # 回头仔细想想别漏了
        a_n = np.random.uniform(0.05, 0.95, 15)
        env.activity()

        env.scheduling(a_n)  # 分配资源块
        env.provisioning(a_n)  # 计算数据速率以及清理buffer

        state = env.get_state()
        observe = state_norm(state)
        for i_iter in range(args.max_train_steps):
            env.countReset()
            env.activity()
            s = observe
            a_n = agent.choose_action(s)
            a_n = (a_n + np.random.normal(0, self.noise_std, size=15)).clip(0.01, 0.97)
            # for _ in range(5):
            #     a_n[_] = (a_n[_] + 1) / 2
            env.scheduling(a_n)  # 分配资源块
            env.provisioning(a_n)  # 计算数据速率以及清理buffer

            state = env.get_state()
            observe = state_norm(state)
            s_ = observe
            latency_ave, energy_ave, energy_que, latency_ave_t, energy_ave_t, energy_que_t = env.get_reward(args)
            latency_average_lst.append(latency_ave_t)  # 本来就是列表，应该不需要tolist
            energy_evergy_lst.append(energy_ave_t)
            energy_que_con_lst.append(energy_que_t)

            reward_set, reward_ave = utils_516_SAC.calc__reward(latency_ave, energy_ave, energy_que)
            reward_lst.append(reward_ave)


            # print(args.epsilon)
            print('\nStep-%d' % i_iter)
            print('reward_set ', reward_set)
            print('reward: ', reward_ave)
            print('latency: ', latency_ave_t)
            # print('reward: ', np.round(reward, 2))
            print('energy: ', energy_ave_t)
            print('energy_que: ', energy_que_t)

            print('action:', a_n)

            terminal = False
            done_n = False
            replay_buffer.store(s, a_n, reward_ave, s_, done_n)  # Store the transition  # Store the transition
            self.total_steps += 1
            if self.args.use_noise_decay:
                self.noise_std = self.noise_std - self.args.noise_std_decay if self.noise_std - self.args.noise_std_decay > self.args.noise_std_min else self.args.noise_std_min

            if self.total_steps > self.args.batch_size:
                agent.learn(replay_buffer)

            writer.add_scalar('Reward_ave', reward_ave, global_step=i_iter + 1)

            if (i_iter + 1) % 10 == 0:
                # 12 项
                np.save('{}/{}.npy'.format(writepath, "TD3_Reward_ave_1_10w"), np.array(reward_lst))
                np.save('{}/{}.npy'.format(writepath, "TD3_latency_average_10w"), np.array(latency_average_lst))
                np.save('{}/{}.npy'.format(writepath, "TD3_energy_evergy_10w"), np.array(energy_evergy_lst))
                np.save('{}/{}.npy'.format(writepath, "TD3_energy_que_con_10w"), np.array(energy_que_con_lst))



if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MADDPG and MATD3 in MPE environment")
    parser.add_argument("--max_train_steps", type=int, default=int(8000), help=" Maximum number of training steps")
    # parser.add_argument("--episode_limit", type=int, default=25, help="Maximum number of steps per episode")
    # parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    # parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")
    parser.add_argument("--max_action", type=float, default=0.8, help="Max action")
    parser.add_argument("--algorithm", type=str, default="MATD3", help="MADDPG or MATD3")
    parser.add_argument("--buffer_size", type=int, default=int(8000), help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--hidden_dim1", type=int, default=64,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--noise_std_init", type=float, default=0.3,
                        help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_std_min", type=float, default=0.2,
                        help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_decay_steps", type=float, default=3000,
                        help="How many steps before the noise_std decays to the minimum")
    parser.add_argument("--use_noise_decay", type=bool, default=True, help="Whether to decay the noise_std")
    parser.add_argument("--lr_a", type=float, default=0.0001, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=0.00015, help="Learning rate of critic")
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

    parser.add_argument("--game_epsilon", type=float, default=0.9, help="total band of SBS")
    parser.add_argument("--adopt_mtr", type=float, default=0.05, help="total band of SBS")

    LOG_TRAIN = 'TD3 219.txt'

    args = parser.parse_args()
    args.noise_std_decay = (args.noise_std_init - args.noise_std_min) / args.noise_decay_steps

    runner = Runner(args, number=1, seed=1)
    runner.run()
