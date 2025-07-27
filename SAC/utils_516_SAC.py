# -*- coding: utf-8 -*-

import itertools
import numpy as np

def gen_state_(pkt_nums, pos):
    mean = np.array([218.8, 5338, 293])
    std = np.array([51, 847, 42.5])
    state = np.hstack(((pkt_nums - mean) / std, pos))
    return state

# 1:2:3
def gen_statem(pkt_nums):
    # mean = np.array([218.8, 5338, 293])
    # std = np.array([51, 847, 42.5])
    mean = np.array([3000, 200])
    std = np.array([500, 30.5])
    state = (pkt_nums - mean) / std
    return state
def gen_states(pkt_nums):
    # mean = np.array([218.8, 5338, 293])
    # std = np.array([51, 847, 42.5])
    # if i == 0:
    #     mean = np.array([3000, 200])
    #     std = np.array([500, 30.5])
    #     state = (pkt_nums - mean) / std
    # else:
    #     mean = np.array([3000, 200])
    #     std = np.array([500, 30.5])
    #     state = (pkt_nums - mean) / std
    #state = np.array(([0, 0],[0, 0],[0, 0],[0, 0],[0, 0]))
    if pkt_nums[0] >= 1000:
        mean = np.array([3500, 800])
        std = np.array([500, 160.5])
        state = (pkt_nums - mean) / std
    else:
        mean = np.array([800, 80])
        std = np.array([114.5, 20.5])
        state = (pkt_nums - mean) / std
    return state



def calc__reward(latency_ave_t, energy_ave_t, energy_que_t):
    latency_par = 1
    beta = 1
    # reward_0 = -(energy_que_t[0]* * energy_ave_t[0] + (latency_ave_t[0]*100 + energy_ave_t[0]))
    # reward_1 = -(energy_que_t[1] * energy_ave_t[1] + (latency_ave_t[1] + energy_ave_t[1]))
    # reward_2 = -(energy_que_t[2] * energy_ave_t[2] + (latency_ave_t[2] + energy_ave_t[2]))
    # reward_3 = -(energy_que_t[3] * energy_ave_t[3] + (latency_ave_t[3] + energy_ave_t[3]))
    # reward_4 = -(energy_que_t[4] * energy_ave_t[4] + (latency_ave_t[4] + energy_ave_t[4]))

    reward_0 = - (latency_ave_t[0] * 100 + energy_ave_t[0]+ energy_que_t[0] * 0.2)
    reward_1 = -(latency_ave_t[1] * 100+ energy_ave_t[1]+ energy_que_t[1] * 0.2)
    reward_2 = -(latency_ave_t[2] * 100+ energy_ave_t[2]+ energy_que_t[2] * 0.2)
    reward_3 = -(latency_ave_t[3] * 100+ energy_ave_t[3]+ energy_que_t[3] * 0.2)
    reward_4 = -(latency_ave_t[4] * 100+ energy_ave_t[4]+ energy_que_t[4] * 0.2)

    reward_total = [reward_0, reward_1, reward_2, reward_3, reward_4]
    reward_ave = sum(reward_total)/5

    return reward_total,reward_ave




