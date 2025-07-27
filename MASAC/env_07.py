import numpy as np
import time
np.random.seed(4)
import matplotlib.pyplot as plt
import argparse as args

class EnvMove(object):
    def __init__(self,
                 ser0_pos=np.array([-50, 50]),  # 服务器位置
                 ser1_pos=np.array([0, -50]),
                 ser2_pos=np.array([50, 50]),
                 BS0_pos=np.array([0, 0]),  # 基站位置
                 BS1_pos=np.array([250, 0]),
                 BS2_pos=np.array([0, 250]),
                 BS3_pos=np.array([-250, 0]),
                 BS4_pos=np.array([0, -250]),

                 # UE_max_no = 5,  #总的用户数量
                 UE_max_no=5,
                 Queue_max=20,  #######
                 noise_PSD=-204,  # -174 dbm/Hz
                 sig2_dB=-114,
                 chan_mod='36814',  ########
                 carrier_freq=10 * 10 ** 9,  # 2 GHz
                 time_subframe=0.5 * 10 ** (-3),  # by LTE, 0.5 ms
                 band_whole=20 * 10 ** 6,  # 10MHz
                 schedu_method='round_robin',
                 ser_prob=np.array([1, 2, 2], dtype=np.float32),  #######
                 dl_mimo_s=100,  #####
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
                 price_m=0.3,
                 price_s=0.5,
                 dis_else=250,
                 BSm_radius=500,
                 BSs_radius=100,
                 ):
        self.dis_else = dis_else
        self.BS0_pos = BS0_pos
        self.ser0_pos = ser0_pos
        self.ser1_pos = ser1_pos
        self.ser2_pos = ser2_pos
        self.BS1_pos = BS1_pos
        self.BS2_pos = BS2_pos
        self.BS3_pos = BS3_pos
        self.BS4_pos = BS4_pos
        self.BSm_radius = BSm_radius
        self.BSs_radius = BSs_radius
        self.sig2_dB = sig2_dB#这里相当于
        self.sig2 = 10 ** (self.sig2_dB / 10)
        self.position_xy = position_xy,
        # self.BSm_radius = 100   #这里改改
        # self.BSs_radius = 100
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
        self.rate_game = np.zeros(100)
        self.UE_max_no_game = 100


        self.UE_latency_total = np.zeros([20, self.UE_max_no])
        ##--------------------------------------------------------------------------
        self.UE_pos = np.random.uniform(-self.BSm_radius, self.BSm_radius, [self.UE_max_no, 2])  ##用户位置

    def ue_position(self):
        pi = np.pi
        theta = np.linspace(0, pi * 2, 1000)
        x0 = np.sin(theta) * self.BSm_radius
        y0 = np.cos(theta) * self.BSm_radius
        x1, x2, x3, x4 = 250, 0, -250, 0  # 圆心的x轴坐标
        y1, y2, y3, y4 = 0, 250, 0, -250  # 圆心的y轴坐标
        figure_0 = plt.figure(figsize=(6, 6))

        circle1 = plt.Circle((x1, y1), self.BSs_radius, color='y', fill=False)  # 对小基站画圆
        circle2 = plt.Circle((x2, y2), self.BSs_radius, color='y', fill=False)
        circle3 = plt.Circle((x3, y3), self.BSs_radius, color='y', fill=False)
        circle4 = plt.Circle((x4, y4), self.BSs_radius, color='y', fill=False)

        plt.gcf().gca().add_artist(circle1)
        plt.gcf().gca().add_artist(circle2)
        plt.gcf().gca().add_artist(circle3)
        plt.gcf().gca().add_artist(circle4)

        plt.plot(x0, y0, label="cycle", color="blue", linewidth=2)

        plt.scatter(self.BS0_pos, self.BS0_pos, s=80, c='black', marker='^')  ###加圆心点
        plt.scatter(self.BS1_pos, self.BS2_pos, c='yellow', s=60, marker='s')
        plt.scatter(self.BS3_pos, self.BS4_pos, c='yellow', s=60, marker='s')

        plt.title("Environment")

        self.random_in_circle_me(20, self.BSm_radius)
        # self.random_in_circle_mu(10, self.BSm_radius)

        for i in range(4):
            if i == 0:
                self.random_in_circle_se(25, self.BSs_radius, i) #第一个小基站内的用户
                # self.random_in_circle_se(12, self.BSs_radius, i)
                # self.random_in_circle_su(18, self.BSs_radius, i)
            elif i == 1:
                # self.random_in_circle_se(8, self.BSs_radius, i)
                # self.random_in_circle_su(12, self.BSs_radius, i)
                self.random_in_circle_se(20, self.BSs_radius, i)
            elif i == 2:
                # self.random_in_circle_se(8, self.BSs_radius, i)
                # self.random_in_circle_su(12, self.BSs_radius, i)
                self.random_in_circle_se(20, self.BSs_radius, i)
            elif i == 3:
                # self.random_in_circle_se(4, self.BSs_radius, i)
                # self.random_in_circle_su(6, self.BSs_radius, i)
                self.random_in_circle_se(15, self.BSs_radius, i)
        # self.UE_cat = np.hstack((self.UE_cat00, self.UE_cat01, self.UE_cat10, self.UE_cat11, self.UE_cat20,
        #                          self.UE_cat21, self.UE_cat30, self.UE_cat31, self.UE_cat40, self.UE_cat41))

        self.UE_pos = np.hstack(
            (self.UE_position0, self.UE_position1, self.UE_position2, self.UE_position3, self.UE_position4))
        self.UE_pos = self.UE_pos.T
        # plt.show()

        # 初始化后第一次统计用户的cell信息
        self.UE_cell = np.zeros(100)  # 这里的ue-cell我理解为基站内用户的数量
        self.UE_cell[(np.sum((self.UE_pos - self.BS1_pos) ** 2, axis=1) <= self.BSs_radius ** 2)] = 1
        self.UE_cell[(np.sum((self.UE_pos - self.BS2_pos) ** 2, axis=1) <= self.BSs_radius ** 2)] = 2
        self.UE_cell[(np.sum((self.UE_pos - self.BS3_pos) ** 2, axis=1) <= self.BSs_radius ** 2)] = 3
        self.UE_cell[(np.sum((self.UE_pos - self.BS4_pos) ** 2, axis=1) <= self.BSs_radius ** 2)] = 4

        # 初始用户位置备份，为了统计每个种群中用户的数量
        self.UE_cell_backup = np.zeros(100)
        self.UE_cell_backup[(np.sum((self.UE_pos - self.BS1_pos) ** 2, axis=1) <= self.BSs_radius ** 2)] = 1
        self.UE_cell_backup[(np.sum((self.UE_pos - self.BS2_pos) ** 2, axis=1) <= self.BSs_radius ** 2)] = 2
        self.UE_cell_backup[(np.sum((self.UE_pos - self.BS3_pos) ** 2, axis=1) <= self.BSs_radius ** 2)] = 3
        self.UE_cell_backup[(np.sum((self.UE_pos - self.BS4_pos) ** 2,
                                    axis=1) <= self.BSs_radius ** 2)] = 4  # numpy.where(condition[, x, y])

        # 统计初始每个种群中用户的数量
        self.ue_num_0 = len(np.where(self.UE_cell_backup == 0)[0])
        self.ue_num_1 = len(np.where(self.UE_cell_backup == 1)[0])
        self.ue_num_2 = len(np.where(self.UE_cell_backup == 2)[0])
        self.ue_num_3 = len(np.where(self.UE_cell_backup == 3)[0])
        self.ue_num_4 = len(np.where(self.UE_cell_backup == 4)[0])
        print([self.ue_num_0, self.ue_num_1, self.ue_num_2, self.ue_num_3, self.ue_num_4])

        plt.legend()
        # plt.show()
        # plt.close()

    def random_in_circle_me(self, N, R):
        t = np.random.random(N)  # 生成一组随机数
        t2 = np.random.random(N)  # 生成第二组随机数
        r = np.sqrt(t) * R  # 密度与半径有平方反比关系
        theta = t2 * 2 * np.pi  # 角度是均匀分布
        h = r * np.cos(theta)  # 换算成直角坐标系
        j = r * np.sin(theta)
        self.UE_position0 = np.vstack((h, j))
        self.UE_cat00 = np.zeros(N)
        plt.plot(h, j, '*', color="red")

    # def random_in_circle_mu(self, N, R):
    #     t = np.random.random(N)  # 生成一组随机数
    #     t2 = np.random.random(N)  # 生成第二组随机数
    #     r = np.sqrt(t) * R  # 密度与半径有平方反比关系
    #     theta = t2 * 2 * np.pi  # 角度是均匀分布
    #     h = r * np.cos(theta)  # 换算成直角坐标系
    #     j = r * np.sin(theta)
    #     self.UE_position01 = np.vstack((h, j))
    #     self.UE_cat01 = np.ones(N)
    #     self.UE_position0 = np.hstack((self.UE_position00, self.UE_position01))
    #     plt.plot(h, j, '*', color="blue")

    def random_in_circle_se(self, N, R, i):  # 这个就是OK的  #宏基站内蓝色的用户
        t = np.random.random(N)  # 生成一组随机数
        t2 = np.random.random(N)  # 生成第二组随机数
        r = np.sqrt(t) * R  # 密度与半径有平方反比关系
        theta = t2 * 2 * np.pi  # 角度是均匀分布
        if i == 0:
            h = r * np.cos(theta) + self.position_xy  # 换算成直角坐标系
            j = r * np.sin(theta)
            self.UE_position1 = np.vstack((h, j))
            # self.UE_cat10 = np.zeros(N)
        elif i == 1:
            h = r * np.cos(theta)  # 换算成直角坐标系
            j = r * np.sin(theta) + self.position_xy
            self.UE_position2 = np.vstack((h, j))
            # self.UE_cat20 = np.zeros(N)
        elif i == 2:
            h = r * np.cos(theta) - self.position_xy  # 换算成直角坐标系
            j = r * np.sin(theta)
            self.UE_position3 = np.vstack((h, j))
            # self.UE_cat30 = np.zeros(N)
        elif i == 3:
            h = r * np.cos(theta)  # 换算成直角坐标系
            j = r * np.sin(theta) - self.position_xy
            self.UE_position4 = np.vstack((h, j))
            # self.UE_cat40 = np.zeros(N)
        plt.plot(h, j, '*', color="red")

    def random_in_circle_su(self, N, R, i):  # 这个就是OK的  #宏基站内红色的用户
        t = np.random.random(N)  # 生成一组随机数
        t2 = np.random.random(N)  # 生成第二组随机数
        r = np.sqrt(t) * R  # 密度与半径有平方反比关系
        theta = t2 * 2 * np.pi  # 角度是均匀分布
        # h = r*np.cos(theta)     #换算成直角坐标系
        # j = r * np.sin(theta)
        if i == 0:
            h = r * np.cos(theta) + self.dis_else
            j = r * np.sin(theta)
            self.UE_position11 = np.vstack((h, j))
            self.UE_position1 = np.hstack((self.UE_position10, self.UE_position11))
            self.UE_cat11 = np.ones(N)
        elif i == 1:
            h = r * np.cos(theta)
            j = r * np.sin(theta) + self.dis_else
            self.UE_position21 = np.vstack((h, j))
            self.UE_position2 = np.hstack((self.UE_position20, self.UE_position21))
            self.UE_cat21 = np.ones(N)
        elif i == 2:
            h = r * np.cos(theta) - self.dis_else
            j = r * np.sin(theta)
            self.UE_position31 = np.vstack((h, j))
            self.UE_position3 = np.hstack((self.UE_position30, self.UE_position31))
            self.UE_cat31 = np.ones(N)
        elif i == 3:
            h = r * np.cos(theta)
            j = r * np.sin(theta) - self.dis_else
            self.UE_position41 = np.vstack((h, j))
            self.UE_position4 = np.hstack((self.UE_position40, self.UE_position41))
            self.UE_cat41 = np.ones(N)
        plt.plot(h, j, '*', color="blue")
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

    def evolutionary_init(self, args):
        figure_1 = plt.figure(figsize=(6, 6))  #

        group0 = np.where(self.UE_cell_backup == 0)
        group1 = np.where(self.UE_cell_backup == 1)
        group2 = np.where(self.UE_cell_backup == 2)
        group3 = np.where(self.UE_cell_backup == 3)
        group4 = np.where(self.UE_cell_backup == 4)
        group0 = np.asarray(group0).flatten()
        group0 = group0.tolist()
        group1 = np.asarray(group1).flatten()
        group1 = group1.tolist()
        group2 = np.asarray(group2).flatten()
        group2 = group2.tolist()
        group3 = np.asarray(group3).flatten()
        group3 = group3.tolist()
        group4 = np.asarray(group4).flatten()
        group4 = group4.tolist()
        group_t = [group1, group2, group3, group4]

        num_t = np.sum([len(group1), len(group2), len(group3), len(group4)])
        # for u in range(9): #固定小基站1的状态为0.1，也就是最左侧一条边
        #     for i in range(1,5): #这里的i就是小基站
        #         for j in group_t[i - 1]: #这里开始选择组别
        #             for _ in j:
        #                 if i - 1 != 1:
        #                    self.UE_cell[_] = np.random.choice([0,i], 1, p = [0.15, 1 - 0.15])
        #                 else:
        #                    self.UE_cell[_] = np.random.choice([0, i], 1, p=[(u/10 + 0.12), 1 - (u/10 + 0.12)])
        #
        #     self.evolutionary(args)

        # for u in range(8):  # #固定小基站2的状态为0.1，也就是最下面一条边
        #     for i in range(1, 5):
        #         for j in group_t[i - 1]:
        #             for _ in j:
        #                 if i - 1 != 0:
        #                     self.UE_cell[_] = np.random.choice([0, i], 1, p=[0.15, 1 - 0.15])
        #                 else:
        #                     self.UE_cell[_] = np.random.choice([0, i], 1, p=[(u / 10 + 0.15), 1 - (u / 10 + 0.15)])
        #
        #     self.evolutionary(args)
        #
        # for u in range(8):  # 固定小基站1的状态为0.9，也就是最右手侧一条边
        #     for i in range(1, 5):
        #         for j in group_t[i - 1]:
        #             for _ in j:
        #                 if i - 1 != 1:
        #                     self.UE_cell[_] = np.random.choice([0, i], 1, p=[0.85, 1 - 0.85])
        #                 else:
        #                     self.UE_cell[_] = np.random.choice([0, i], 1, p=[(u / 10 + 0.15), 1 - (u / 10 + 0.15)])
        #
        #     self.evolutionary(args)
        #
        # for u in range(7):  # 固定小基站2的状态为0.9，也就是最上面一条边
        #     for i in range(1, 5):
        #         for j in group_t[i - 1]:
        #             for _ in j:
        #                 if i - 1 != 0:
        #                     self.UE_cell[_] = np.random.choice([0, i], 1, p=[0.85, 1 - 0.85])
        #                 else:
        #                     self.UE_cell[_] = np.random.choice([0, i], 1, p=[(u / 10 + 0.25), 1 - (u / 10 + 0.25)])
        #
        #     self.evolutionary(args)
        a0 = len(group0)  # 17
        a1 = len(group1)  # 20
        a2 = len(group2)  # 20
        a3 = len(group3)  # 22
        a4 = len(group4)  # 21

        print()
        # 不按位置试试
        # for u in range(9):

        # 指定连接
        for u in range(1, 10, 1):  # 第一组线
            for i in range(1, 5):  # 四个小基站
                if i == 1:
                    for j in range(len(group_t[i - 1])):
                        if j == 0:
                            self.UE_cell[group_t[i - 1][j]] = 0
                        else:
                            self.UE_cell[group_t[i - 1][j]] = i
                elif i == 2:
                    for j in range(len(group_t[i - 1])):
                        if j < (len(group_t[i - 1])) * (u / 10):
                            self.UE_cell[group_t[i - 1][j]] = 0
                        else:
                            self.UE_cell[group_t[i - 1][j]] = i
                elif i == 3:  # 小基站3和4现在不重要
                    for j in range(len(group_t[i - 1])):
                        if j == 0:
                            self.UE_cell[group_t[i - 1][j]] = 0
                        else:
                            self.UE_cell[group_t[i - 1][j]] = i
                    #     if j == 0:
                    #         self.UE_cell[group_t[i - 1][j]] = 0
                    #     else:
                    #         self.UE_cell[group_t[i - 1][j]] = 1
                elif i == 4:
                    for j in range(len(group_t[i - 1])):
                        if j < (len(group_t[i - 1])) * (u / 10):
                            self.UE_cell[group_t[i - 1][j]] = 0
                        else:
                            self.UE_cell[group_t[i - 1][j]] = i
            self.evolutionary(args)

        for u in range(1, 9, 1):  # 第二组线
            for i in range(1, 5):  # 四个小基站
                if i == 1:
                    for j in range(len(group_t[i - 1])):
                        if j <= (len(group_t[i - 1])) * (u / 10 + 0.1):
                            self.UE_cell[group_t[i - 1][j]] = 0
                        else:
                            self.UE_cell[group_t[i - 1][j]] = 1
                elif i == 2:
                    for j in range(len(group_t[i - 1])):
                        if j == 0:
                            self.UE_cell[group_t[i - 1][j]] = 0
                        else:
                            self.UE_cell[group_t[i - 1][j]] = i
                elif i == 3:  # 小基站3和4现在不重要
                    for j in range(len(group_t[i - 1])):
                        if j == 0:
                            self.UE_cell[group_t[i - 1][j]] = 0
                        else:
                            self.UE_cell[group_t[i - 1][j]] = i
                elif i == 4:
                    for j in range(len(group_t[i - 1])):
                        if j == 0:
                            self.UE_cell[group_t[i - 1][j]] = 0
                        else:
                            self.UE_cell[group_t[i - 1][j]] = i

            self.evolutionary(args)

        for u in range(1, 9, 1):  # 第三组线
            for i in range(1, 5):  # 四个小基站
                if i == 1:
                    for j in range(len(group_t[i - 1])):
                        if j == 0:
                            self.UE_cell[group_t[i - 1][j]] = 1
                        else:
                            self.UE_cell[group_t[i - 1][j]] = 0
                elif i == 2:
                    for j in range(len(group_t[i - 1])):
                        if j <= (len(group_t[i - 1])) * (u / 10 + 0.1):
                            self.UE_cell[group_t[i - 1][j]] = 0
                        else:
                            self.UE_cell[group_t[i - 1][j]] = i
                elif i == 3:  # 小基站3和4现在不重要
                    for j in range(len(group_t[i - 1])):
                        if j == 0:
                            self.UE_cell[group_t[i - 1][j]] = 0
                        else:
                            self.UE_cell[group_t[i - 1][j]] = i
                elif i == 4:
                    for j in range(len(group_t[i - 1])):
                        if j == 0:
                            self.UE_cell[group_t[i - 1][j]] = 0
                        else:
                            self.UE_cell[group_t[i - 1][j]] = i

            self.evolutionary(args)

        for u in range(1, 8, 1):  # 第四组线
            for i in range(1, 5):  # 四个小基站
                if i == 1:
                    for j in range(len(group_t[i - 1])):
                        if j <= (len(group_t[i - 1])) * (u / 10 + 0.1):
                            self.UE_cell[group_t[i - 1][j]] = 0
                        else:
                            self.UE_cell[group_t[i - 1][j]] = i
                elif i == 2:
                    for j in range(len(group_t[i - 1])):
                        if j == 0:
                            self.UE_cell[group_t[i - 1][j]] = i
                        else:
                            self.UE_cell[group_t[i - 1][j]] = 0
                elif i == 3:  # 小基站3和4现在不重要
                    for j in range(len(group_t[i - 1])):
                        if j == 0:
                            self.UE_cell[group_t[i - 1][j]] = 0
                        else:
                            self.UE_cell[group_t[i - 1][j]] = i
                elif i == 4:
                    for j in range(len(group_t[i - 1])):
                        if j == 0:
                            self.UE_cell[group_t[i - 1][j]] = 0
                        else:
                            self.UE_cell[group_t[i - 1][j]] = i
            self.evolutionary(args)

        # for u in range(9): #固定小基站1的状态为0.1，也就是最左侧一条边
        #     for i in range(1,5): #这里的i就是小基站
        #         for j in range(len(group_t[i - 1])): #这里开始选择组别
        #             if j == 0:
        #                 self.UE_cell[group_t[i - 1][j]] = i
        #             elif j == 1:
        #                 self.UE_cell[group_t[i - 1][j]] = 0
        #             else:
        #                 if i - 1 != 1:
        #                    self.UE_cell[group_t[i - 1][j]] = np.random.choice([0,i], 1, p = [0.01, 1 - 0.01])
        #                 else:
        #                    self.UE_cell[group_t[i - 1][j]] = np.random.choice([0, i], 1, p=[(u/10 + 0.01), 1 - (u/10 + 0.01)])
        #
        #     self.evolutionary(args)
        #
        # for u in range(8): ##固定小基站2的状态为0.1，也就是最下面一条边
        #     for i in range(1,5): #这里的i就是小基站
        #         for j in range(len(group_t[i - 1])): #这里开始选择组别
        #             if j == 0:
        #                 self.UE_cell[group_t[i - 1][j]] = i
        #             elif j == 1:
        #                 self.UE_cell[group_t[i - 1][j]] = 0
        #             else:
        #                 if i - 1 != 0:
        #                    self.UE_cell[group_t[i - 1][j]] = np.random.choice([0,i], 1, p = [0.01, 1 - 0.01])
        #                 else:
        #                    self.UE_cell[group_t[i - 1][j]] = np.random.choice([0, i], 1, p=[(u/10 + 0.01), 1 - (u/10 + 0.01)])
        #
        #     self.evolutionary(args)
        #
        # for u in range(8): # 固定小基站1的状态为0.9，也就是最右手侧一条边
        #     for i in range(1,5): #这里的i就是小基站
        #         for j in range(len(group_t[i - 1])): #这里开始选择组别
        #             if j == 0:
        #                 self.UE_cell[group_t[i - 1][j]] = i
        #             elif j == 1:
        #                 self.UE_cell[group_t[i - 1][j]] = 0
        #             else:
        #                 if i - 1 != 1:
        #                    self.UE_cell[group_t[i - 1][j]] = np.random.choice([0,i], 1, p = [0.99, 1 - 0.99])
        #                 else:
        #                    self.UE_cell[group_t[i - 1][j]] = np.random.choice([0, i], 1, p=[(u/10 + 0.29), 1 - (u/10 + 0.29)])
        #
        #     self.evolutionary(args)
        #
        # for u in range(7): # 固定小基站2的状态为0.9，也就是最上面一条边
        #     for i in range(1,5): #这里的i就是小基站
        #         for j in range(len(group_t[i - 1])): #这里开始选择组别
        #             if j == 0:
        #                 self.UE_cell[group_t[i - 1][j]] = i
        #             elif j == 1:
        #                 self.UE_cell[group_t[i - 1][j]] = 0
        #             else:
        #                 if i - 1 != 0:
        #                    self.UE_cell[group_t[i - 1][j]] = np.random.choice([0,i], 1, p = [0.99, 1 - 0.99])
        #                 else:
        #                    self.UE_cell[group_t[i - 1][j]] = np.random.choice([0, i], 1, p=[(u/10 + 0.01), 1 - (u/10 + 0.01)])
        #
        #     self.evolutionary(args)
        hhh = [a0, a1, a2, a3, a4]
        print(hhh)
        figure_1 = plt.show()
        # figure_2 = plt.figure()
        # x_date = ["BS{}".format(i) for i in range(5)]
        # self.UE_cat = np.random.choice(self.ser_cat, self.UE_max_no_game, p=self.ser_prob)

    def evolutionary(self, args):
        LOG_TRAIN = 'evolutionary11.txt'
        x_list = []
        y_list = []
        self.ue_connected_bs_0()  # 计算每个基站连接用户的数量

        # self.loadm = len(np.where(self.UE_cell == 0)[0]) / self.UE_max_no_game  #计算每个基站的负载 即与基站关联
        # self.loads1 = len(np.where(self.UE_cell == 1)[0]) / self.UE_max_no_game
        # self.loads2 = len(np.where(self.UE_cell == 2)[0]) / self.UE_max_no_game
        # self.loads3 = len(np.where(self.UE_cell == 3)[0]) / self.UE_max_no_game
        # self.loads4 = len(np.where(self.UE_cell == 4)[0]) / self.UE_max_no_game
        x = []
        y = []
        z = []
        w = []

        p0 = []
        p1 = []
        p2 = []
        p3 = []
        p4 = []

        p_a_0 = []
        p_a_1 = []
        p_a_2 = []
        p_a_3 = []
        p_a_4 = []
        self.state_computation(x, y, z, w)  # 计算初始状态
        # x.append(self.x_1_m)
        # y.append(self.x_2_m)

        # 迭代40次计算效用和x，y
        ite = 40
        for i in range(ite):
            self.computingpayoff(args, p0, p1, p2, p3, p4, p_a_0, p_a_1, p_a_2, p_a_3, p_a_4)  # 这里求payoff主要是因为下面选择策略得用
            # 下面开始策略选择
            for i in range(100):
                if self.UE_cell_backup[i] == 0:
                    continue  # 只在宏基站下的用户不选择
                elif self.UE_cell_backup[i] == 1:
                    if self.UE_cell[i] == 0:
                        if (self.pi_1_0 < self.pi_ave_1) & (abs(self.pi_1_0 - self.pi_ave_1) > args.adopt_mtr):
                            # if self.pi_1_0 < self.pi_ave_1:
                            # if (self.pi_ave_1 - self.pi_1_0) / self.pi_ave_1 > np.random.random():
                            if np.random.random() >= args.game_epsilon:
                                self.UE_cell[i] = 1
                    else:
                        if (self.pi_1_1 < self.pi_ave_1) & (abs(self.pi_1_1 - self.pi_ave_1) > args.adopt_mtr):
                            # if self.pi_1_1 < self.pi_ave_1:
                            # if (self.pi_ave_1 - self.pi_1_1) / self.pi_ave_1 > np.random.random():
                            if np.random.random() >= args.game_epsilon:
                                self.UE_cell[i] = 0
                elif self.UE_cell_backup[i] == 2:
                    if self.UE_cell[i] == 0:
                        if (self.pi_2_0 < self.pi_ave_2) & (abs(self.pi_2_0 - self.pi_ave_2) > args.adopt_mtr):
                            # if self.pi_2_0 < self.pi_ave_2:
                            # if (self.pi_ave_2 - self.pi_2_0) / self.pi_ave_2 > np.random.random():
                            if np.random.random() >= args.game_epsilon:
                                self.UE_cell[i] = 2
                    else:
                        if (self.pi_2_2 < self.pi_ave_2) & (abs(self.pi_2_2 - self.pi_ave_2) > args.adopt_mtr):
                            # if self.pi_2_2 < self.pi_ave_2:
                            # if (self.pi_ave_2 - self.pi_2_2) / self.pi_ave_2 > np.random.random():
                            if np.random.random() >= args.game_epsilon:
                                self.UE_cell[i] = 0
                elif self.UE_cell_backup[i] == 3:
                    if self.UE_cell[i] == 0:
                        if (self.pi_3_0 < self.pi_ave_3) & (abs(self.pi_3_0 - self.pi_ave_3) > args.adopt_mtr):
                            # if self.pi_3_0 < self.pi_ave_3:
                            # if (self.pi_ave_3 - self.pi_3_0) / self.pi_ave_3 > np.random.random():
                            if np.random.random() >= args.game_epsilon:
                                self.UE_cell[i] = 3
                    else:
                        if (self.pi_3_3 < self.pi_ave_3) & (abs(self.pi_3_3 - self.pi_ave_3) > args.adopt_mtr):
                            # if self.pi_3_3 < self.pi_ave_3:
                            # if (self.pi_ave_3 - self.pi_3_3) / self.pi_ave_3 > np.random.random():
                            if np.random.random() >= args.game_epsilon:
                                self.UE_cell[i] = 0
                elif self.UE_cell_backup[i] == 4:
                    if self.UE_cell[i] == 0:
                        if (self.pi_4_0 < self.pi_ave_4) & (abs(self.pi_4_0 - self.pi_ave_4) > args.adopt_mtr):
                            # if self.pi_4_0 < self.pi_ave_4:
                            # if (self.pi_ave_4 - self.pi_4_0) / self.pi_ave_4 > np.random.random():
                            if np.random.random() >= args.game_epsilon:
                                self.UE_cell[i] = 4
                    else:
                        if (self.pi_4_4 < self.pi_ave_4) & (abs(self.pi_4_4 - self.pi_ave_4) > args.adopt_mtr):
                            # if self.pi_4_4 < self.pi_ave_4:
                            # if (self.pi_ave_4 - self.pi_4_4) / self.pi_ave_4 > np.random.random():
                            if np.random.random() >= args.game_epsilon:
                                self.UE_cell[i] = 0

            self.state_computation(x, y, z, w)
        self.computingpayoff(args, p0, p1, p2, p3, p4, p_a_0, p_a_1, p_a_2, p_a_3, p_a_4)
        self.ue_connected_bs()
        plt.plot(x, y)
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.p_a_0 = p_a_0
        self.p_a_1 = p_a_1
        self.p_a_2 = p_a_2
        self.p_a_3 = p_a_3
        self.p_a_4 = p_a_4
        value = [x, y]
        print('x', x)
        print('y', y)
        # x_list.append(self.x)
        # y_list.append(self.y)
        with open(LOG_TRAIN, 'a+') as f:
            for i in range(len(self.x)):
                print(
                    '%.3f, %.3f, %.3f, %.3f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (
                    self.x[i], self.y[i], self.z[i], self.w[i],
                    self.p0[i], self.p_a_0[i], self.p1[i], self.p_a_1[i], self.p2[i], self.p_a_2[i], self.p3[i],
                    self.p_a_3[i], self.p4[i], self.p_a_4[i]), file=f)
        x_list = []
        y_list = []
        reward_lst = []
        utility_lst = []

        # y_date =

    def computingpayoff(self, args, p0, p1, p2, p3, p4, p_a_0, p_a_1, p_a_2, p_a_3, p_a_4):
        self.sum_rate_game_0 = np.zeros(1)
        self.sum_rate_game_1 = np.zeros(1)
        self.sum_rate_game_2 = np.zeros(1)
        self.sum_rate_game_3 = np.zeros(1)
        self.sum_rate_game_4 = np.zeros(1)

        # self.chan_mod_game()
        self.band_computation(args)

        rx_power_m = 2 * 10 ** (-7)
        rx_power_s = 2 * 10 ** (-9)
        for i in range(100):
            if self.UE_cell[i] == 0:
                self.rate_game[i] = self.band_mbs_alo * np.log10(1 + rx_power_m / (self.sig2 * 0.01)) * 0.75
            elif self.UE_cell[i] == 1:
                self.rate_game[i] = self.band_sbs1_alo * np.log10(1 + rx_power_s / (self.sig2 * 0.01))
            elif self.UE_cell[i] == 2:
                self.rate_game[i] = self.band_sbs2_alo * np.log10(1 + rx_power_s / (self.sig2 * 0.01))
            elif self.UE_cell[i] == 3:
                self.rate_game[i] = self.band_sbs3_alo * np.log10(1 + rx_power_s / (self.sig2 * 0.01))
            elif self.UE_cell[i] == 4:
                self.rate_game[i] = self.band_sbs4_alo * np.log10(1 + rx_power_s / (self.sig2 * 0.01))
        self.coefficient_R = 10
        # 下面这个先待定
        # for i in range (self.UE_max_no):
        #     self.payoff_game[i] = np.log(1 + (self.coefficient_R * self.rate_game[i])/self.num_t[self.UE_cell[i]]) - self.price * self.num_t[self.UE_cell[i]]

        ###计算每个种群的平均效用,先计算每个种群的效用，再计算平均效用
        for i in range(100):
            if self.UE_cell[i] == 0:
                self.sum_rate_game_0 += self.rate_game[i]
            elif self.UE_cell[i] == 1:
                self.sum_rate_game_1 += self.rate_game[i]
            elif self.UE_cell[i] == 2:
                self.sum_rate_game_2 += self.rate_game[i]
            elif self.UE_cell[i] == 3:
                self.sum_rate_game_3 += self.rate_game[i]
            elif self.UE_cell[i] == 4:
                self.sum_rate_game_4 += self.rate_game[i]

        self.num0_e = len(np.where(self.UE_cell == 0)[0])
        self.num1_e = len(np.where(self.UE_cell == 1)[0])
        self.num2_e = len(np.where(self.UE_cell == 2)[0])
        self.num3_e = len(np.where(self.UE_cell == 3)[0])
        self.num4_e = len(np.where(self.UE_cell == 4)[0])

        # self.pi_0_0 = self.sum_rate_game_0/len(np.where(self.UE_cell == 0)[0]) - self.price_m * len(np.where(self.UE_cell == 0)[0])
        self.pi_0_0 = np.log(1 + self.sum_rate_game_0 / len(np.where(self.UE_cell == 0)[0])) \
                      * (1 / (1 - (0.7 - len(np.where(self.UE_cell == 0)[0]) / self.UE_max_no_game)))

        self.pi_1_0 = self.pi_0_0  # 小基站1内的用户选择宏基站的效用
        self.pi_2_0 = self.pi_0_0
        self.pi_3_0 = self.pi_0_0
        self.pi_4_0 = self.pi_0_0

        # self.pi_1_1 = self.sum_rate_game_1 / len(np.where(self.UE_cell == 1)[0]) - self.price_s * len(np.where(self.UE_cell == 1)[0])
        # self.pi_2_2 = self.sum_rate_game_2 / len(np.where(self.UE_cell == 2)[0]) - self.price_s * len(np.where(self.UE_cell == 2)[0])
        # self.pi_3_3 = self.sum_rate_game_3 / len(np.where(self.UE_cell == 3)[0]) - self.price_s * len(np.where(self.UE_cell == 3)[0])
        # self.pi_4_4 = self.sum_rate_game_4 / len(np.where(self.UE_cell == 4)[0]) - self.price_s * len(np.where(self.UE_cell == 4)[0])

        self.pi_1_1 = np.log(1 + self.sum_rate_game_1 / len(np.where(self.UE_cell == 1)[0])) * (
                    1 / (1 - (0.1 - len(np.where(self.UE_cell == 1)[0]) / self.UE_max_no_game)))
        self.pi_2_2 = np.log(1 + self.sum_rate_game_2 / len(np.where(self.UE_cell == 2)[0])) * (
                    1 / (1 - (0.1 - len(np.where(self.UE_cell == 2)[0]) / self.UE_max_no_game)))
        self.pi_3_3 = np.log(1 + self.sum_rate_game_3 / len(np.where(self.UE_cell == 3)[0])) * (
                    1 / (1 - (0.1 - len(np.where(self.UE_cell == 3)[0]) / self.UE_max_no_game)))
        self.pi_4_4 = np.log(1 + self.sum_rate_game_4 / len(np.where(self.UE_cell == 4)[0])) * (
                    1 / (1 - (0.1 - len(np.where(self.UE_cell == 4)[0]) / self.UE_max_no_game)))

        self.pi_ave_0 = self.pi_1_0
        self.pi_ave_1 = (self.pi_1_0 * self.x_1_m + self.pi_1_1 * self.x_1_1)
        self.pi_ave_2 = (self.pi_2_0 * self.x_2_m + self.pi_2_2 * self.x_2_2)
        self.pi_ave_3 = (self.pi_3_0 * self.x_3_m + self.pi_3_3 * self.x_3_3)
        self.pi_ave_4 = (self.pi_4_0 * self.x_4_m + self.pi_4_4 * self.x_4_4)

        p0.append(self.pi_1_0)
        p1.append(self.pi_1_1)
        p2.append(self.pi_2_2)
        p3.append(self.pi_3_3)
        p4.append(self.pi_4_4)

        p_a_0.append(self.pi_ave_0)
        p_a_1.append(self.pi_ave_1)
        p_a_2.append(self.pi_ave_2)
        p_a_3.append(self.pi_ave_3)
        p_a_4.append(self.pi_ave_4)

        return p0, p1, p2, p3, p4, p_a_0, p_a_1, p_a_2, p_a_3, p_a_4

    def band_computation(self, args):
        self.band_mbs_alo = args.COMP_MBS / len(np.where(self.UE_cell == 0)[0]) * 10
        self.band_sbs1_alo = args.COMP_UAV / len(np.where(self.UE_cell == 1)[0]) * 10
        self.band_sbs2_alo = args.COMP_UAV / len(np.where(self.UE_cell == 2)[0]) * 10
        self.band_sbs3_alo = args.COMP_UAV / len(np.where(self.UE_cell == 3)[0]) * 10
        self.band_sbs4_alo = args.COMP_UAV / len(np.where(self.UE_cell == 4)[0]) * 10

    def ue_connected_bs(self):
        self.num0 = len(np.where(self.UE_cell == 0)[0])
        self.num1 = len(np.where(self.UE_cell == 1)[0])
        self.num2 = len(np.where(self.UE_cell == 2)[0])
        self.num3 = len(np.where(self.UE_cell == 3)[0])
        self.num4 = len(np.where(self.UE_cell == 4)[0])

        self.num_t = [self.num0, self.num1, self.num2, self.num3, self.num4]
        print(self.num_t)

    def ue_connected_bs_0(self):
        self.num0 = len(np.where(self.UE_cell == 0)[0])
        self.num1 = len(np.where(self.UE_cell == 1)[0])
        self.num2 = len(np.where(self.UE_cell == 2)[0])
        self.num3 = len(np.where(self.UE_cell == 3)[0])
        self.num4 = len(np.where(self.UE_cell == 4)[0])

        self.num_t = [self.num0, self.num1, self.num2, self.num3, self.num4]
        print('began', self.num_t)

    def state_computation(self, x, y, z, w):
        # 统计在种群i中选择基站k的比例，用来计算状态
        self.num_1_m = len(np.where((self.UE_cell_backup == 1) & (self.UE_cell == 0))[0])  # 6
        self.num_2_m = len(np.where((self.UE_cell_backup == 2) & (self.UE_cell == 0))[0])  #
        self.num_3_m = len(np.where((self.UE_cell_backup == 3) & (self.UE_cell == 0))[0])  #
        self.num_4_m = len(np.where((self.UE_cell_backup == 4) & (self.UE_cell == 0))[0])  #
        # 求初始状态
        self.x_1_m = self.num_1_m / self.ue_num_1  # 用到了前面的初始每个基站内的用户数量
        self.x_1_1 = 1 - self.x_1_m
        self.x_2_m = self.num_2_m / self.ue_num_2
        self.x_2_2 = 1 - self.x_2_m
        self.x_3_m = self.num_3_m / self.ue_num_3
        self.x_3_3 = 1 - self.x_3_m
        self.x_4_m = self.num_4_m / self.ue_num_4
        self.x_4_4 = 1 - self.x_4_m

        x.append(self.x_1_m)
        y.append(self.x_2_m)
        z.append(self.x_3_m)
        w.append(self.x_4_m)

        return x, y, z, w

    def chan_mod_game(self):
        shadowing_var = 8  # rayleigh fading shadowing variance 8dB
        for i in range(5):
            if i == 0:
                self.dis0 = np.sqrt(np.sum((self.BS0_pos - self.UE_pos) ** 2, axis=1)) / 1000  # unit changes to km
            elif i == 1:
                self.dis1 = np.sqrt(np.sum((self.BS1_pos - self.UE_pos) ** 2, axis=1)) / 1000  # unit changes to km
            elif i == 2:
                self.dis2 = np.sqrt(np.sum((self.BS2_pos - self.UE_pos) ** 2, axis=1)) / 1000  # unit changes to km
            elif i == 3:
                self.dis3 = np.sqrt(np.sum((self.BS3_pos - self.UE_pos) ** 2, axis=1)) / 1000  # unit changes to km
            elif i == 4:
                self.dis4 = np.sqrt(np.sum((self.BS4_pos - self.UE_pos) ** 2, axis=1)) / 1000  # unit changes to km
        self.path_loss0 = 145.4 + 37.5 * np.log10(self.dis0).reshape(-1, 1)
        self.path_loss1 = 145.4 + 37.5 * np.log10(self.dis1).reshape(-1, 1)
        self.path_loss2 = 145.4 + 37.5 * np.log10(self.dis2).reshape(-1, 1)
        self.path_loss3 = 145.4 + 37.5 * np.log10(self.dis3).reshape(-1, 1)
        self.path_loss4 = 145.4 + 37.5 * np.log10(self.dis4).reshape(-1, 1)
        self.chan_loss0 = self.path_loss0 + np.random.normal(0, shadowing_var, self.UE_max_no_game).reshape(-1, 1)
        self.chan_loss1 = self.path_loss1 + np.random.normal(0, shadowing_var, self.UE_max_no_game).reshape(-1, 1)
        self.chan_loss2 = self.path_loss2 + np.random.normal(0, shadowing_var, self.UE_max_no_game).reshape(-1, 1)
        self.chan_loss3 = self.path_loss3 + np.random.normal(0, shadowing_var, self.UE_max_no_game).reshape(-1, 1)
        self.chan_loss4 = self.path_loss4 + np.random.normal(0, shadowing_var, self.UE_max_no_game).reshape(-1, 1)

# def bufferUpdate(buffer, rate, time_subframe):
#     bSize = buffer.size
#     for i in range(bSize):
#         if buffer[i] >= rate * time_subframe:
#             buffer[i] -= rate * time_subframe
#             rate = 0
#             break
#         else:
#             rate -= buffer[i] / time_subframe  #
#             buffer[i] = 0
#     return buffer

# def latencyUpdate(latency, buffer, time_subframe):
#     lSize = latency.size
#     for i in range(lSize):
#         if buffer[i] != 0:
#             latency[i] += time_subframe
#     return latency