
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from replay_buffer import ReplayBuffer
from maddpg import MADDPG
from matd3_514 import MATD3
from env_519 import EnvMove
import utils_07
from normalization import Normalization
import os, shutil


class Runner:
    def __init__(self, args, number, seed):
        self.args = args
        self.seed = seed
        self.number = number
        self.args.N = 5  # 智能体数量
        self.args.obs_dim_n = [3, 3, 3, 3, 3]  # 状态维度
        self.args.action_dim_n = [3, 3, 3, 3, 3]

        #Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Create N agents
        if self.args.algorithm == "MADDPG":
            print("Algorithm: MADDPG")
            self.agent_n = [MADDPG(args, agent_id) for agent_id in range(args.N)]
        elif self.args.algorithm == "MATD3":
            print("Algorithm: MATD3")
            self.agent_n = [MATD3(args, agent_id) for agent_id in range(args.N)]
        else:
            print("Wrong!!!")

        self.replay_buffer = ReplayBuffer(self.args)

        # Create a tensorboard
        #self.writer = SummaryWriter(log_dir='runs/{}/{}_env_{}_number_{}_seed_{}'.format(self.args.algorithm, self.args.algorithm, self.env_name, self.number, self.seed))

        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0
        self.noise_std = self.args.noise_std_init  # Initialize noise_std

    def run(self, ):
        # writepath = 'MATD3_quanzhong_1003_1'
        writepath = 'MATD3_Reward_ave_10w_nse2'
        if os.path.exists(writepath): shutil.rmtree(writepath)
        if not os.path.exists(writepath): os.mkdir(writepath)
        writer = SummaryWriter(log_dir=writepath)

        state_norm = Normalization(shape=3)
        latency_average_lst, energy_evergy_lst = [], []  #
        energy_que_con_lst = []
        reward_lst = []
        env = EnvMove()
        # env.countReset()  # 回头仔细想想别漏了
        a_n = [np.random.uniform(0.01, 0.99, 3) for i in self.agent_n]
        env.activity()
        env.scheduling(a_n)  #
        env.provisioning(a_n)  #
        state_0, state_1, state_2, state_3, state_4 = env.get_state()
        state_total = [state_0, state_1, state_2, state_3, state_4]
        # observe = [utilsthree.gen_states(pkt) for agent, pkt in zip(self.agent_n, pkt_total)] #后面再调调宏基站和小基站分开，利用宏基站数据包总数量的区别
        observe = [state_norm(sta) for agent, sta in zip(self.agent_n, state_total)]
        for i_iter in range(args.max_train_steps):
            # ue_list = []
            env.countReset()
            env.activity()
            s = observe
            a_n = [agent.choose_action(obs, noise_std=self.noise_std) for agent, obs in zip(self.agent_n, s)]
            # for _ in range(5):
            #     a_n[_] = (a_n[_] + 1) / 2

            env.scheduling(a_n)  # 分配资源块
            env.provisioning(a_n)  # 计算数据速率以及清理buffer

            # for i in range(self.args.N):
            #     UE_num = env.find_ue_num(i)
            #     ue_list.append(UE_num)
            state_0, state_1, state_2, state_3, state_4 = env.get_state()
            state_total = [state_0, state_1, state_2, state_3, state_4]

            # observe = [utilsthree.gen_states(pkt) for agent, pkt in zip(self.agent_n, pkt_total)]
            observe = [state_norm(sta) for agent, sta in zip(self.agent_n, state_total)]
            s_ = observe
            # latency_average, tp, qoe, qoe_ave_t, latency_average_t = env.get_reward(args)
            latency_ave, energy_ave, energy_que, latency_ave_t, energy_ave_t, energy_que_t = env.get_reward(args)
            latency_average_lst.append(latency_ave_t)  # 本来就是列表，应该不需要tolist
            energy_evergy_lst.append(energy_ave_t)
            energy_que_con_lst.append(energy_que_t)

            reward_set, reward_ave = utils_07.calc__reward(latency_ave, energy_ave, energy_que)
            reward_lst.append(reward_ave)
            # reward_ave_lst.append(reward_ave)
            # utility_lst.append(utility.tolist())
            # utility_set_lst.append(utility_set)

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
            done_n = [False, False, False, False, False]
            self.replay_buffer.store_transition(s, a_n, reward_set, s_, done_n)  # Store the transition
            self.total_steps += 1

            # Decay noise_std
            if self.args.use_noise_decay:
                self.noise_std = self.noise_std - self.args.noise_std_decay if self.noise_std - self.args.noise_std_decay > self.args.noise_std_min else self.args.noise_std_min
            if self.replay_buffer.current_size > self.args.batch_size:
                # Train each agent individually
                for agent_id in range(self.args.N):
                    self.agent_n[agent_id].train(self.replay_buffer, self.agent_n)

            # writer.add_scalar('Reward_ave', reward_ave, global_step=i_iter + 1)
            # writer.add_scalar('Reward_ave', reward_ave, global_step=i_iter + 1)
            # writer.add_scalar('Reward_ave', reward_ave, global_step=i_iter + 1)
            # writer.add_scalar('Reward_ave', reward_ave, global_step=i_iter + 1)
            if (i_iter + 1) % 10 == 0:
                # 12 项

                # np.save('{}/{}.npy'.format(writepath, "MATD3_Reward_ave_1"), np.array(reward_lst))
                # np.save('{}/{}.npy'.format(writepath, "MATD3_latency_average"), np.array(latency_average_lst))
                # np.save('{}/{}.npy'.format(writepath, "MATD3_energy_evergy"), np.array(energy_evergy_lst))
                # np.save('{}/{}.npy'.format(writepath, "MATD3_energy_que_con"), np.array(energy_que_con_lst))
                # np.save('{}/{}.npy'.format(writepath, "MATD3_Reward_ave_1_0.0001"), np.array(reward_lst))
                # np.save('{}/{}.npy'.format(writepath, "MATD3_latency_average_0.0001"), np.array(latency_average_lst))
                # np.save('{}/{}.npy'.format(writepath, "MATD3_energy_evergy_0.0001"), np.array(energy_evergy_lst))
                # np.save('{}/{}.npy'.format(writepath, "MATD3_energy_que_con_0.0001"), np.array(energy_que_con_lst))
                np.save('{}/{}.npy'.format(writepath, "MATD3_Reward_ave_1_1003_1"), np.array(reward_lst))
                np.save('{}/{}.npy'.format(writepath, "MATD3_latency_average_1003_1"), np.array(latency_average_lst))
                np.save('{}/{}.npy'.format(writepath, "MATD3_energy_evergy_1003_1"), np.array(energy_evergy_lst))
                np.save('{}/{}.npy'.format(writepath, "MATD3_energy_que_con_1003_1"), np.array(energy_que_con_lst))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MADDPG and MATD3 in MPE environment")
    parser.add_argument("--max_train_steps", type=int, default=int(8000), help=" Maximum number of training steps")
    # parser.add_argument("--episode_limit", type=int, default=25, help="Maximum number of steps per episode")
    # parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    # parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")
    parser.add_argument("--max_action", type=float, default=0.90, help="Max action")

    parser.add_argument("--algorithm", type=str, default="MATD3", help="MADDPG or MATD3")
    parser.add_argument("--buffer_size", type=int, default=int(8000), help="The capacity of the replay buffer")
    # parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")#no2
    parser.add_argument("--hidden_dim", type=int, default=128, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--hidden_dim1", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--noise_std_init", type=float, default=0.25, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_std_min", type=float, default=0.05, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_decay_steps", type=float, default=4000, help="How many steps before the noise_std decays to the minimum")
    parser.add_argument("--use_noise_decay", type=bool, default=True, help="Whether to decay the noise_std")
    parser.add_argument("--lr_a", type=float, default=0.001, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=0.0015, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Softly update the target network")
    parser.add_argument("--use_orthogonal_init", type=bool, default=False, help="Orthogonal initialization")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
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

    # runner = Runner(args, number=1, seed=3)
    # runner = Runner(args, number=1, seed=6)#nose
    runner = Runner(args, number=1, seed=8)  # nose2
    runner.run()
