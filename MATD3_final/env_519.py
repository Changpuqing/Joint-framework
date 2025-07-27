#517 日，把13-14改成了12-13
#518日，把小基站mimo调小了一点
import numpy as np
import time
np.random.seed(4)
import matplotlib.pyplot as plt
import argparse as args

class EnvMove(object):
    def __init__(self,
                 ser0_pos=np.array([-50, 50]),  # 无人机器位置
                 ser1_pos=np.array([0, -50]),
                 ser2_pos=np.array([50, 50]),

                 UE_max_no = 5,  #总的用户数量
                 Queue_max=20,  #######
                 noise_PSD=-204,  # -174 dbm/Hz
                 sig2_dB=-114,
                 chan_mod='36814',  ########
                 carrier_freq=10 * 10 ** 9,  # 2 GHz
                 time_subframe=0.5 * 10 ** (-3),  # by LTE, 0.5 ms
                 band_whole=20 * 10 ** 6,  # 10MHz
                 schedu_method='round_robin',
                 ser_prob=np.array([1, 2, 2], dtype=np.float32),  #######
                 # dl_mimo_s=100,  #####
                 # dl_mimo_s=70,  #nose
                 dl_mimo_s=85,  # nose2
                 dl_mimo_m=1,
                 rx_gain=20,  # dB
                 learning_windows=20,
                 cpu_cycle = 100,
                 UE_local_cap = 0.5*10**9,
                 p_max=100,
                 p_min = 250,
                 k_UE = 10e-27,
                 k_ser = 10e-30,
                 ser_cap = 2 * 10 ** 9,
                 # content = ['1','2','3'],  ########
                 position_xy=250,
                 ):
        self.ser0_pos = ser0_pos
        self.ser1_pos = ser1_pos
        self.ser2_pos = ser2_pos
        self.sig2_dB = sig2_dB#这里相当于
        self.sig2 = 10 ** (self.sig2_dB / 10)

        self.BSm_radius = 100   #这里改改
        self.BSs_radius = 100
        self.band_whole = band_whole
        self.chan_mod = chan_mod
        self.carrier_freq = carrier_freq
        self.time_subframe = round(time_subframe, 4)              ##########
        self.noise_PSD = noise_PSD
        self.sys_clock = 0                                          ####
        self.schedu_method = schedu_method
        self.dl_mimo_m = dl_mimo_m
        self.dl_mimo_s = dl_mimo_s
        self.UE_rx_gain = rx_gain
        self.UE_max_no = UE_max_no
        self.cpu_cycle = cpu_cycle
        self.UE_local_cap = UE_local_cap
        self.UE_buffer_total = np.zeros([20, self.UE_max_no])
        self.UE_buffer_local = np.zeros([20, self.UE_max_no])
        self.UE_buffer_trans = np.zeros([20, self.UE_max_no])
        self.UE_latency_tran = np.zeros([20, self.UE_max_no])
        self.UE_latency_local = np.zeros([20, self.UE_max_no])
        self.UE_energy_tran = np.zeros([20, self.UE_max_no])
        self.UE_energy_local = np.zeros([20, self.UE_max_no])
        self.UE_latency_ser = np.zeros([20, self.UE_max_no])
        self.UE_energy_ser = np.zeros([20, self.UE_max_no])
        self.read_time = 0
        self.ser_choose = np.zeros(5)
        self.p_action = np.zeros(5)  #这里是初始化选择功率动作
        self.ser_dis = np.zeros(5) #每个用户距离所选服务器的距离
        self.p_max = p_max
        self.k_UE = k_UE
        self.k_ser = k_ser  ######
        self.UE_band = np.zeros(self.UE_max_no)
        self.learning_windows = learning_windows      ###保留四位小数
        self.p_min = p_min


        self.UE_latency_total = np.zeros([20, self.UE_max_no])
        ##--------------------------------------------------------------------------
        self.UE_pos = np.random.uniform(-self.BSm_radius, self.BSm_radius, [self.UE_max_no, 2])  ##用户位置


    def channel_model(self):
        if self.chan_mod == '36814':
            shadowing_var = 8  # rayleigh fading shadowing variance 8dB
            # if i == 0:
            #     dis = np.sqrt(np.sum((self.BS0_pos - self.UE_pos) ** 2, axis=1)) / 1000  # unit changes to km
            # elif i == 1:
            #     dis = np.sqrt(np.sum((self.BS1_pos - self.UE_pos) ** 2, axis=1)) / 1000  # unit changes to km
            # elif i == 2:
            #     dis = np.sqrt(np.sum((self.BS2_pos - self.UE_pos) ** 2, axis=1)) / 1000  # unit changes to km
            # elif i == 3:
            #     dis = np.sqrt(np.sum((self.BS3_pos - self.UE_pos) ** 2, axis=1)) / 1000  # unit changes to km
            # elif i == 4:
            #     dis = np.sqrt(np.sum((self.BS4_pos - self.UE_pos) ** 2, axis=1)) / 1000  # unit changes to km
            self.path_loss = 145.4 + 37.5 * np.log10(self.ser_dis).reshape(-1, 1)
            self.chan_loss = self.path_loss + np.random.normal(0, shadowing_var, self.UE_max_no).reshape(-1, 1)

    ##----------------------------------------------------------------------------------------------------------##
    def scheduling(self,a_n):
        for i in range(20):  #用于传输的任务
            for j in range(5):
                self.UE_buffer_trans[i][j] = self.UE_buffer_total[i][j] * a_n[j][1]

        self.UE_buffer_local = self.UE_buffer_total - self.UE_buffer_trans #本地的等于总任务减去卸载的，这个应该在计算传输任务的前面
        for i in range(self.UE_max_no): #这里是根据动作选服务器
            if a_n[i][0]>=0 and  a_n[i][0]<0.33:
                self.ser_choose[i] = 0  #0代表本地
            elif a_n[i][0]>=0.33 and  a_n[i][0]<=0.66:
                self.ser_choose[i] = 1   #剩下的分别代表服务器
            elif a_n[i][0]>=0.66 and  a_n[i][0]<=1:
                self.ser_choose[i] = 2
            # elif a_n[i][0] >= 0.75 and a_n[i][0] <= 1:
            #     self.ser_choose[i] = 3
        #计算每个服务器用户的数量
        self.ser_UE_number = np.zeros(3)
        for i in range(self.UE_max_no):
            if self.ser_choose[i] == 0:
                self.ser_UE_number[0] += 1
            elif self.ser_choose[i] == 1:
                self.ser_UE_number[1] += 1
            elif self.ser_choose[i] == 2:
                self.ser_UE_number[2] += 1


        for i in range(self.UE_max_no):  #计算距离
            if self.ser_choose[i] == 0:
                self.ser_dis[i] = np.sqrt(np.sum((self.ser0_pos - self.UE_pos[i]) ** 2, axis=0))
            elif self.ser_choose[i] == 1:
                self.ser_dis[i] = np.sqrt(np.sum((self.ser1_pos - self.UE_pos[i]) ** 2, axis=0)) ##这里有错误
            elif self.ser_choose[i] == 2:
                self.ser_dis[i] = np.sqrt(np.sum((self.ser2_pos - self.UE_pos[i]) ** 2, axis=0))

        #距离算完了。距离算完之后算发射功率
        for i in range(self.UE_max_no):
            # self.p_action[i] = self.p_max * a_n[i][2]
            self.p_action[i] = self.p_max * a_n[i][2] + 200

    def computing_caching(self, rate): #这部分有个漏洞，如果有两个同时传完怎么办
        #先计算传输时延，如果全部本地计算那么传输时延为0，能耗也就为0，所以不用单独讨论，传输时延为卸载的任务大小比上速率
        #队列任务积压问题描述不清楚，这里先认定如果0.5秒内没有处理完就说明有一个积压试试效果
        for i in range(20):
            for j in range(self.UE_max_no):
                self.UE_latency_tran[i][j] = self.UE_buffer_trans[i][j]/rate[j]
        #计算传输能耗
        for i in range(20):
            for j in range(self.UE_max_no):
                self.UE_energy_tran[i][j] = self.UE_latency_tran[i][j] * self.p_action[j]
        #然后计算本地处理时延
        for i in range(20):
            for j in range(self.UE_max_no):
                self.UE_latency_local[i][j] = self.UE_buffer_local[i][j] * self.cpu_cycle/ self.UE_local_cap

        #计算本地能耗
        for i in range(20):
            for j in range(self.UE_max_no):
                self.UE_energy_local[i][j] = self.UE_latency_local[i][j] * (self.UE_local_cap**3) * self.k_UE
        #计算服务器计算的时延要先计算计算每个用户分得的计算资源
        #计算每个用户分得的计算资源
        self.UE_cap = np.zeros(self.UE_max_no)
        for i in range(self.UE_max_no):
            if self.ser_choose[i] == 0:
              self.UE_cap[i] = self.carrier_freq/self.ser_UE_number[0]
            elif self.ser_choose[i] == 1:
              self.UE_cap[i] = self.carrier_freq/self.ser_UE_number[1]
            elif self.ser_choose[i] == 2:
              self.UE_cap[i] = self.carrier_freq/self.ser_UE_number[2]
        #然后计算每个用户在服务器处理任务用的时延
        for i in range(20):
            for j in range(self.UE_max_no):
                self.UE_latency_ser[i][j] = self.UE_buffer_trans[i][j] * self.cpu_cycle/ self.UE_cap[j]

        #计算服务器计算能耗
        for i in range(20):
            for j in range(self.UE_max_no):
                # self.UE_energy_ser[i][j] = self.UE_latency_ser[i][j] * (self.carrier_freq**3) * self.k_ser #
                self.UE_energy_ser[i][j] = self.UE_latency_ser[i][j] * (self.UE_cap[j] ** 3) * self.k_ser

        #计算总时延
        for i in range(20):
            for j in range(self.UE_max_no):
                self.UE_latency_total[i][j] = max(self.UE_latency_local[i][j],self.UE_latency_tran[i][j]+self.UE_latency_ser[i][j])
        #计算总能耗
        self.UE_energy_total = self.UE_energy_tran + self.UE_energy_local + self.UE_energy_ser

        #计算每个用户的平均时延
        for i in range(self.UE_max_no):
            self.ave_latency = np.sum(self.UE_latency_total,axis=0)/self.learning_windows

        #计算每个用户的平均能耗
        # self.ave_energy = np.zeros(self.UE_max_no)
        for i in range(self.UE_max_no):
            self.ave_energy = np.sum(self.UE_energy_total,axis=0)/self.learning_windows

        #计算能量队列积压
        # for i in range(self.UE_max_no):
        #     for j in range(self.learning_windows):
        #         if self.UE_energy_total[i][j] >= 2:
        self.energy_q_con = np.zeros(self.UE_max_no)
        for i in range(self.UE_max_no):
            self.energy_q_con[i] = (np.sum(self.UE_energy_total, axis=0))[i] - 2 * self.learning_windows  ###
            if self.energy_q_con[i] < 0:
                self.energy_q_con[i] = 0
        print()

    def provisioning(self,a_n):
        # UE_index = np.where(self.UE_band != 0) #如果这里也算上上个智能体的band怎么办
        self.rx_power = np.zeros(self.UE_max_no)
        self.channel_model()
        for i in range(self.UE_max_no):
            self.rx_power[i] = 10 ** ((self.p_action[i] - self.chan_loss[i] + self.UE_rx_gain) / 10) * self.dl_mimo_s

        rx_power = self.rx_power.reshape(1, -1)[0] #[0]
        rate = np.zeros(self.UE_max_no)
        #rate[UE_index] = self.UE_band[UE_index] * np.log10(1 + rx_power[UE_index] / (10 ** (self.noise_PSD / 10) * self.UE_band[UE_index])) * self.dl_mimo
        for i in range(self.UE_max_no):
            self.UE_band = self.band_whole/self.UE_max_no
            rate[i] = self.UE_band * np.log10(1 + rx_power[i] / (self.sig2 * 0.01)) * self.dl_mimo_m

        #计算完速率开始计算时延
        self.computing_caching(rate)
        # self.store_reward(rate, i)

    def activity(self):
        # for ue_id in range(5):
            # self.read_time+=1
        for i in range(self.learning_windows): #生成总任务
            for j in range(self.UE_max_no):
                tmp_buffer_size = np.random.randint(900000, 1100000)
                self.UE_buffer_total[i][j] = tmp_buffer_size
        print('a') #删掉

    def get_state(self): #状态为任务大小，距离和队列积压
        state0 = [np.sum(self.UE_buffer_total,axis=0)[0],self.ser_dis[0],self.energy_q_con[0]]
        state1 = [np.sum(self.UE_buffer_total, axis=0)[1], self.ser_dis[1], self.energy_q_con[1]]
        state2 = [np.sum(self.UE_buffer_total, axis=0)[2], self.ser_dis[2], self.energy_q_con[2]]
        state3 = [np.sum(self.UE_buffer_total, axis=0)[3], self.ser_dis[3], self.energy_q_con[3]]
        state4 = [np.sum(self.UE_buffer_total, axis=0)[4], self.ser_dis[4], self.energy_q_con[4]]

        return state0,state1,state2,state3,state4

    def get_reward(self,args):
        self.energy_ave_t = np.sum(self.ave_energy)/self.UE_max_no
        self.latency_ave_t = np.sum(self.ave_latency)/self.UE_max_no
        self.energy_que_t = np.sum(self.energy_q_con)/self.UE_max_no

        return self.ave_latency, self.ave_energy, self.energy_q_con,self.latency_ave_t,self.energy_ave_t,  self.energy_que_t

    def countReset(self):
        self.UE_buffer_total = np.zeros([20, self.UE_max_no])
        self.UE_buffer_local = np.zeros([20, self.UE_max_no])
        self.UE_buffer_trans = np.zeros([20, self.UE_max_no])
        self.UE_latency_tran = np.zeros([20, self.UE_max_no])
        self.UE_latency_local = np.zeros([20, self.UE_max_no])
        self.UE_energy_tran = np.zeros([20, self.UE_max_no])
        self.UE_energy_local = np.zeros([20, self.UE_max_no])
        self.UE_latency_ser = np.zeros([20, self.UE_max_no])
        self.UE_energy_ser = np.zeros([20, self.UE_max_no])


