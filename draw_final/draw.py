import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import  ConnectionPatch
import pandas as pd
from matplotlib.pyplot import MultipleLocator  # 从pyplot导入MultipleLocator类，这个类用于设置刻度间隔
font1 = {'family': 'Times New Roman',
'weight': 'normal',
'size': 12,}

y2 = []
y3 = []
y4 = []
y5 = []

def zone_and_linked(ax, axins, zone_left, zone_right, x, y, linked='bottom',
                    x_ratio=0.05, y_ratio=0.05):
    """缩放内嵌图形，并且进行连线
    ax:         调用plt.subplots返回的画布。例如： fig,ax = plt.subplots(1,1)
    axins:      内嵌图的画布。 例如 axins = ax.inset_axes((0.4,0.1,0.4,0.3))
    zone_left:  要放大区域的横坐标左端点
    zone_right: 要放大区域的横坐标右端点
    x:          X轴标签
    y:          列表，所有y值
    linked:     进行连线的位置，{'bottom','top','left','right'}
    x_ratio:    X轴缩放比例
    y_ratio:    Y轴缩放比例
    """
    xlim_left = x[zone_left] - (x[zone_right] - x[zone_left]) * x_ratio
    xlim_right = x[zone_right] + (x[zone_right] - x[zone_left]) * x_ratio

    y_data = np.hstack([yi[zone_left:zone_right] for yi in y])
    ylim_bottom = np.min(y_data) - (np.max(y_data) - np.min(y_data)) * y_ratio
    ylim_top = np.max(y_data) + (np.max(y_data) - np.min(y_data)) * y_ratio

    axins.set_xlim(xlim_left, xlim_right)
    axins.set_ylim(ylim_bottom, ylim_top)

    ax.plot([xlim_left, xlim_right, xlim_right, xlim_left, xlim_left],
            [ylim_bottom, ylim_bottom, ylim_top, ylim_top, ylim_bottom], "black")

    if linked == 'bottom':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_left, ylim_bottom)
        xyA_2, xyB_2 = (xlim_right, ylim_top), (xlim_right, ylim_bottom)
    elif linked == 'top':
        xyA_1, xyB_1 = (xlim_left, ylim_bottom), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_right, ylim_top)
    elif linked == 'left':
        xyA_1, xyB_1 = (xlim_right, ylim_top), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_left, ylim_bottom)
    elif linked == 'right':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_right, ylim_top)
        xyA_2, xyB_2 = (xlim_left, ylim_bottom), (xlim_right, ylim_bottom)

    con = ConnectionPatch(xyA=xyA_1, xyB=xyB_1, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax)
    axins.add_artist(con)
    con = ConnectionPatch(xyA=xyA_2, xyB=xyB_2, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax)
    axins.add_artist(con)


class gogogo:
    def __init__(self, weight,inter, inter2, weight1):
        self.weight = weight
        self.weight1 = weight1
        self.inter = inter
    def smooth(self,reward,inter,weight):# inter--步长
        reward = reward.flatten()  # 平铺
        reward = reward.astype(np.float32)
        len = reward.shape[0]  # 获取数据长度
        need_smooth_reward = reward[0:len:inter]

        smooth_reward = []
        for i in range(need_smooth_reward.shape[0]):
            if i == 0:
                smooth_reward.append(need_smooth_reward[i])
            else:
                smooth_reward.append(smooth_reward[-1] * weight + need_smooth_reward[i] * (1-weight))

        return np.array(smooth_reward)
    def smooth1(self,reward,inter,weight1):# inter--步长
        reward = reward.flatten()  # 平铺
        reward = reward.astype(np.float32)
        len = reward.shape[0]  # 获取数据长度
        need_smooth_reward = reward[0:len:inter]

        smooth_reward = []
        for i in range(need_smooth_reward.shape[0]):
            if i == 0:
                smooth_reward.append(need_smooth_reward[i])
            else:
                smooth_reward.append(smooth_reward[-1] * weight1 + need_smooth_reward[i] * (1-weight1))

        return np.array(smooth_reward)
    def run_utility(self):

        self.file1 = self.smooth(np.load(r'E:\A5000\Code\draw_final\MATD3_830\MATD3_energy_evergy.npy', allow_pickle=True, encoding="latin1"), self.inter, self.weight)

        self.file2 = self.smooth(
            np.load(r'E:\A5000\Code\draw_final\MATD3_830\MATD3_energy_ave_10w_nose.npy', allow_pickle=True,
                    encoding="latin1"),
            self.inter, self.weight)
        self.file3 = self.smooth(
            np.load(r'E:\A5000\Code\draw_final\MATD3_830\MATD3_energy_ave_10w_nose2.npy', allow_pickle=True,
                    encoding="latin1"),
            self.inter, self.weight)

        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.style.use('bmh')
        plt.rcParams['axes.facecolor'] = 'white'
        plt.xlabel('Episode', fontsize=15, fontproperties="Times New Roman")
        plt.ylabel('Reward', fontsize=15, fontproperties="Times New Roman")
        plt.ylabel('Latency', fontsize=15, fontproperties="Times New Roman")
        # plt.ylabel('Energy Queue', fontsize=15, fontproperties="Times New Roman")
        plt.ylabel('Energy', fontsize=15, fontproperties="Times New Roman")

        # 设置坐标轴刻度
        x_ticks = np.arange(0, 8001, 1000)
        y_ticks = np.arange(-15, 0, 1)#reward
        y_ticks = np.arange(0, 0.15, 0.01)#latency
        y_ticks = np.arange(0, 3, 0.3)


        plt.xticks(x_ticks, fontproperties="Times New Roman")
        plt.yticks(y_ticks, fontproperties="Times New Roman")
        ax = plt.gca()
        # y 轴用科学记数法
        # ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
        ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')

        # plt.xticks(x_ticks)
        # plt.yticks(y_ticks)

        # 设置刻度的字号
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.grid(linewidth=0.5,linestyle=':')
        bwith = 0.8  # 边框宽度设置为2
        ax = plt.gca()  # 获取边框
        ax.spines['top'].set_color('black')  # 设置上‘脊梁’为红色
        ax.spines['right'].set_color('black')  # 设置上‘脊梁’为无色
        ax.spines['left'].set_color('black')  # 设置上‘脊梁’为红色
        ax.spines['bottom'].set_color('black')  # 设置上‘脊梁’为无色
        ax.spines['bottom'].set_linewidth(bwith)
        ax.spines['left'].set_linewidth(bwith)
        ax.spines['top'].set_linewidth(bwith)
        ax.spines['right'].set_linewidth(bwith)
        # plt.grid(color='black', linestyle='-.', linewidth=1)

        # plt.plot(x4, self.file4+2, color='darkorange', linewidth=1.0, linestyle=':', label='SAC')
        # plt.plot(x3, self.file3+2, color='dimgray', linewidth=1.0, linestyle=':', label='TD3')
        # plt.plot(x2, self.file2+2, color='blue', linewidth=1.0, linestyle=':', label='MASAC')
        # plt.plot(x1, self.file1+2, color='red', linewidth=1.0, linestyle=':', label='MATD3')
        # plt.plot(x4, self.file4, color='darkorange', linewidth=1.0, linestyle=':', label='SAC')
        plt.plot(x3, self.file3, color='navy', linewidth=1.0, linestyle=':', label='Select the nearest')
        plt.plot(x2, self.file2, color='seagreen', linewidth=1.0, linestyle=':', label='Select by Random')
        plt.plot(x1, self.file1, color='red', linewidth=1.0, linestyle=':', label='Select by EG')
        # 设置图例大小
        # plt.legend(fontsize=10, loc='best')
        plt.legend(loc='upper right', fontsize=10)  # 标签位置
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles[::-1], labels[::-1], loc='lower right', prop=font1)
        # plt.grid(axis='both', which='major', color='black', linewidth=0.5, linestyle='-')
        # plt.savefig('....', dpi=500, bbox_inches='tight')
        plt.savefig('....', dpi=500, bbox_inches='tight')

if __name__ == '__main__':
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 12, }
    '''x1 = list(range(0, 6000))
    # m = len(x1)
    # inter = 1
    # x1 = x1[0:m:inter]
    # x1 = np.array(x1)
    # 
    # x2 = list(range(0, 6001))
    # n = len(x2)
    # inter2 = 100
    # x2 = x2[0:n:inter2]
    # x2 = np.array(x2)'''
    x1 = list(range(0, 8000))
    m = len(x1)
    inter = 1
    x1 = x1[0:m:inter]
    x1 = np.array(x1)

    x2 = list(range(0, 8000))
    n = len(x2)
    inter1 = 1
    x2 = x2[0:n:inter]
    x2 = np.array(x2)

    x3 = list(range(0, 8000))
    n = len(x3)
    inter2 = 1
    x3 = x3[0:n:inter]
    x3 = np.array(x3)

    x4 = list(range(0, 8000))
    n = len(x4)
    inter = 1
    x4 = x2[0:n:inter]
    x4 = np.array(x4)

    weight = 0.96
    weight1 = 0.3
    runner = gogogo(weight, inter, inter2,weight1)
    runner.run_utility()
    plt.show()



