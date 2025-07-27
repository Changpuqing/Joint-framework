
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from replay_buffer import ReplayBuffer
from MASAC_5072 import MASAC
import copy
from normalization import Normalization
import os, shutil
from env_90 import EnvMove
import utils_07


class Runner:
    def __init__(self, args, number, seed):
        self.args = args
        self.seed = seed
        self.number = number
        self.args.N = 5  # 智能体数量
        self.args.obs_dim_n = [3, 3, 3,3, 3]  # 状态维度
        self.args.action_dim_n = [3, 3, 3, 3, 3]  # 智能体的动作维度

        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Create N agents
        if self.args.algorithm == "MASAC":
            print("Algorithm: MASAC")
            self.agent_n = [MASAC(args, agent_id) for agent_id in range(args.N)]

        self.replay_buffer = ReplayBuffer(self.args)

        #Create a tensorboard
        #self.writer = SummaryWriter(log_dir=self.writepath)

        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0

        #self.noise_std = self.args.noise_std_init  # Initialize noise_std

    def run(self ):
        writepath = 'MASAC_code_10w'
        if os.path.exists(writepath): shutil.rmtree(writepath)
        if not os.path.exists(writepath): os.mkdir(writepath)
        writer = SummaryWriter(log_dir=writepath)

        state_norm = Normalization(shape=3)
        latency_average_lst, energy_evergy_lst = [], []  #
        energy_que_con_lst = []
        reward_lst = []
        env = EnvMove()
        # env.countReset()  # 回头仔细想想别漏了
        env.ue_position()
        a_n = [np.random.uniform(0.01, 0.99, 3) for i in self.agent_n]
        env.evolutionary_init(args)
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
            a_n = [agent.choose_action(obs) for agent, obs in zip(self.agent_n, s)]
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
            latency_ave, energy_ave, energy_que, latency_ave_t, energy_ave_t, energy_que_t= env.get_reward(args)
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

            if self.replay_buffer.current_size > self.args.batch_size:
                # Train each agent individually
                for agent_id in range(self.args.N):
                    self.agent_n[agent_id].learn(self.replay_buffer, self.agent_n)

            writer.add_scalar('Reward_ave', reward_ave, global_step=i_iter + 1)
            if (i_iter + 1) % 10 == 0:
                # 12 项

                np.save('{}/{}.npy'.format(writepath, "MATD3_Reward_ave_1_5w"), np.array(reward_lst))
                np.save('{}/{}.npy'.format(writepath, "MATD3_latency_average_5w"), np.array(latency_average_lst))
                np.save('{}/{}.npy'.format(writepath, "MATD3_energy_evergy_5w"), np.array(energy_evergy_lst))
                np.save('{}/{}.npy'.format(writepath, "MATD3_energy_que_con_5w"), np.array(energy_que_con_lst))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MADDPG and MATD3 in MPE environment")
    parser.add_argument("--max_train_steps", type=int, default=int(8000), help=" Maximum number of training steps")
    parser.add_argument("--max_action", type=float, default=0.9, help="Max action")
    parser.add_argument("--algorithm", type=str, default="MASAC", help="MADDPG or MATD3")
    parser.add_argument("--buffer_size", type=int, default=int(6000), help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--hidden_dim1", type=int, default=128, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--hidden_dim2", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=0.0015, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=0.0015, help="Learning rate of critic")
    parser.add_argument("--alpha", type=float, default=0.2, help="alpha")
    parser.add_argument("--lr_alpha", type=float, default=0.001, help="Learning rate of alpha")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Softly update the target network")
    # parser.add_argument("--log_alpha", type=float, default=-torch.zeros(1, requires_grad=True), help="log_alpha")
    parser.add_argument("--use_orthogonal_init", type=bool, default=False, help="Orthogonal initialization")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
    parser.add_argument("--adaptive_alpha", type=bool, default=True, help="adaptive_alpha")
    parser.add_argument("--COMP_MBS", type=int, default=40, help="total band of SBS")
    parser.add_argument("--COMP_UAV", type=int, default=15, help="total band of SBS")
    parser.add_argument("--adopt_mtr", type=float, default=0.05, help="total band of SBS")#zhuyao0.03 /90人时是0.05
    parser.add_argument("--game_epsilon", type=float, default=0.9, help="total band of SBS")
    # --------------------------------------MATD3--------------------------------------------------------------------
    parser.add_argument("--policy_noise", type=float, default=0.2, help="Target policy smoothing")
    parser.add_argument("--noise_clip", type=float, default=0.5, help="Clip noise")
    parser.add_argument("--policy_update_freq", type=int, default=2, help="The frequency of policy updates")
    # --------------------------------------MASAC--------------------------------------------------------------------
    parser.add_argument("--alpha0", type=float, default=0.2, help="alpha")
    parser.add_argument("--alpha1", type=float, default=0.2, help="alpha")
    parser.add_argument("--alpha2", type=float, default=0.2, help="alpha")
    parser.add_argument("--alpha3", type=float, default=0.2, help="alpha")
    parser.add_argument("--alpha4", type=float, default=0.2, help="alpha")

    parser.add_argument("--LEARNING_WINDOW", type=int, default=20, help="the number of timeslot")
    # parser.add_argument("--BAND_MBS", type=int, default=20, help="total band of MBS")

    parser.add_argument("--UE_NUMS", type=int, default=5, help="total band of SBS")

    LOG_TRAIN = 'MASAC_test.txt'

    args = parser.parse_args()
#    args.noise_std_decay = (args.noise_std_init - args.noise_std_min) / args.noise_decay_steps

    # runner = Runner(args, number=1, seed=3)
    runner = Runner(args, number=1, seed=4)
    runner.run()
