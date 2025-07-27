import numpy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator  # 从pyplot导入MultipleLocator类，这个类用于设置刻度间隔
font1 = {'family': 'Times New Roman',
'weight': 'normal',
'size': 12,
         }
class gogogo:
    def __init__(self, weight,inter, inter2):
        self.weight = weight
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
# def smooth(reward,inter,weight):# inter--步长
#     # reward = reward.flatten()  # 平铺
#     # reward = reward.astype(np.float32)
#     len_r = len(reward)  # 获取数据长度
#     need_smooth_reward = reward[0:len_r:inter]
#
#     smooth_reward = []
#     for i in range(len_r):
#         if i == 0:
#             smooth_reward.append(need_smooth_reward[i])
#         else:
#             smooth_reward.append(smooth_reward[-1] * weight + need_smooth_reward[i] * (1-weight))
#
#     return np.array(smooth_reward)


def smooth(y1):
    smooth_reward = []
    for i in range(y1.shape[0]):
        if i == 0:
            smooth_reward.append(y1[i])  # 直接添加到列表中
        else:
            smooth_reward.append(smooth_reward[-1] * 0.6 + y1[i] * 0.4)
    return np.array(smooth_reward)

def smooth2(y2):
    smooth_reward = []
    for i in range(y2.shape[0]):
        if i == 0:
            smooth_reward.append(y2[i])  # 直接添加到列表中
        else:
            smooth_reward.append(smooth_reward[-1] * 0.8 + y2[i] * 0.2)
    return np.array(smooth_reward)

def smooth3(y3):
    smooth_reward = []
    for i in range(y3.shape[0]):
        if i == 0:
            smooth_reward.append(y3[i])  # 直接添加到列表中
        else:
            smooth_reward.append(smooth_reward[-1] * 0.8 + y3[i] * 0.2)
    return np.array(smooth_reward)
#
def smooth4(y4):
    smooth_reward = []
    for i in range(y4.shape[0]):
        if i == 0:
            smooth_reward.append(y4[i])  # 直接添加到列表中
        else:
            smooth_reward.append(smooth_reward[-1] * 0.6 + y4[i] * 0.4)
    return np.array(smooth_reward)

def smooth5(y5):
    smooth_reward = []
    for i in range(y5.shape[0]):
        if i == 0:
            smooth_reward.append(y5[i])  # 直接添加到列表中
        else:
            smooth_reward.append(smooth_reward[-1] * 0.8+ y5[i] * 0.2)
    return np.array(smooth_reward)


# y1 = [0.015,0.016,0.020,0.022,0.026,0.028] #时延
# y2 = [0.017,0.021,0.026,0.028,0.036,0.040]
# y3 = [0.025,0.04,0.042,0.049,0.059,0.064]
# y4 = [0.023,0.030,0.034,0.039,0.044,0.052]

y1 = [1.38,1.59,1.80,2.09,2.34,2.54] #能量
y2 = [1.27,1.56,1.78,1.97,2.23,2.41]
y3 = [1.19,1.35,1.7,1.92,2.26,2.46]
y4 = [1.14,1.34,1.67,1.84,2.14,2.23]

# y1 = [0,0,0.9,1.59,6.0,10.5] #队列
# y2 = [0,0,0.75,0.96,3.34,7.9]
# y3 = [0,0,3.2,4.5,5.1,7.4]
# y4 = [0,0,0.1,0.6,1.6,4.1]


x1 = [500000,600000,700000,800000,900000,1000000]
# y2 = [54,75,90,110,130]
# x2 = [60,80,100,120,140]
# y3 = [60,99,108,130,140]
# x3 = [60,80,100,120,140]
# y4 = [40,70,80,108,126]
# x4 = [60,80,100,120,140]
# y5 = [38,49,60,70,80]
# x5 = [60,80,100,120,140]

# y1 = [108,205,78,82,71,75]
# x1 = [0.2,0.3,0.4,0.5,0.6,0.7]
# y2 = [96,78,75,63,61,60]
# x2 = [0.2,0.3,0.4,0.5,0.6,0.7]

# 去掉上、右坐标线
'''fig, ax = plt.subplots() 
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)'''
# 设置默认字体，解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('bmh')
plt.rcParams['axes.facecolor']='white'

# 设置坐标轴标签与字号
plt.xlabel('Task Size', fontsize=15,fontproperties="Times New Roman")
# plt.xlabel('Task Size(Mbits)', fontsize=15,fontproperties="Times New Roman")
# plt.xlabel('CPU Frequency of UE(GHz)', fontsize=15,fontproperties="Times New Roman")
# plt.xlabel('Number of UEs', fontsize=15,fontproperties="Times New Roman")
plt.ylabel('Latency', fontsize=15,fontproperties="Times New Roman")
plt.ylabel('Energy', fontsize=15,fontproperties="Times New Roman")
# plt.ylabel('Energy Queue', fontsize=15,fontproperties="Times New Roman")

# 设置坐标范围
# plt.xlim(xmax=1000, xmin=0)
# plt.ylim(ymax=160, ymin=40)

# plt.xlim(xmax=140, xmin=60)
# plt.ylim(ymax=260, ymin=40)

# 设置坐标轴刻度
# x_ticks = np.arange(0, 1200,200)
# y_ticks = np.arange(40,200,20)

# x_ticks = np.arange(60, 160,20)
# y_ticks = np.arange(20,260,20)

# x_ticks = np.arange(0.2, 0.8,0.1)
# y_ticks = np.arange(60,220,20)

x_ticks = np.arange(500000, 1100000,100000)
y_ticks = np.arange(0,0.1,0.01)
y_ticks = np.arange(0,3,0.5)
# y_ticks = np.arange(0,11,2)
ax = plt.gca()
# y 轴用科学记数法
# ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
ax.ticklabel_format(style='sci', scilimits=(0, 0), axis='x')

# plt.xticks(x_ticks)
# plt.yticks(y_ticks)
plt.xticks(x_ticks, fontproperties="Times New Roman")
plt.yticks(y_ticks, fontproperties="Times New Roman")
'''# 把x轴的刻度间隔设置为1，把y轴的刻度间隔设置为10，并存在变量里
x_major_locator = MultipleLocator(2000)
y_major_locator = MultipleLocator(100)'''
# 设置刻度的字号
plt.tick_params(axis='both', which='major', labelsize=13)
'''# 设置图片长宽
plt.figure(figsize=(12, 8))'''
plt.grid(linewidth=0.5, linestyle=':')
bwith = 1.2  # 边框宽度设置为2
ax = plt.gca()  # 获取边框
ax.spines['top'].set_color('black')  # 设置上‘脊梁’为红色
ax.spines['right'].set_color('black')  # 设置上‘脊梁’为无色
ax.spines['left'].set_color('black')  # 设置上‘脊梁’为红色
ax.spines['bottom'].set_color('black')  # 设置上‘脊梁’为无色
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)

# 绘制折线图

# plt.plot(x1, y1, color='darkgreen', linewidth=1.2, linestyle='-', mark
#-------------------
# plt.plot(x1, y1, color='darkgreen', linewidth=1.2, linestyle='--', marker='*',


plt.plot(x1, y4,color='orange', linewidth=1.2, linestyle='-', marker='D', markersize=5, label='SAC')
plt.plot(x1, y3, color='green', linewidth=1.2, linestyle='-', marker='.', markersize=5, label='TD3')
plt.plot(x1, y2, color='mediumblue', linewidth=1.2, linestyle=':', marker='v', markersize=5, label='MASAC')
plt.plot(x1, y1, color='brown', linewidth=1.2, linestyle='--', marker='o', markersize=5, label='MATD3')
'''font1 = {"family": "Times New Roman",
            "weight": "normal",
            "size": 23,
        }
plt.legend(handles=[A, B], labels=['hao', 'huai'], loc='upper right', prop=font1, labelspacing=1, frameon=True)
'''
# 设置图例大小
# plt.legend(fontsize=10, loc='best')
# plt.legend(loc='lower right', fontsize=10) # 标签位置
ax = plt.gca()
handles,labels = ax.get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1],loc='upper left', prop=font1)
#plt.legend(handles[::-1], labels[::-1],loc='lower right', prop=font1)

# plt.legend(loc='lower right', prop=font1)
# 设置标题
#plt.title('(a) Utility', fontsize=20,fontproperties="Times New Roman")
plt.grid(axis='both', which='major', color='silver', linewidth=0.5, linestyle=':')
# plt.grid(axis='both', which='major', color='black', linewidth=0.5, linestyle=':')
# plt.savefig('E:/A5000/picture/energy_value_dif_buffer.png', dpi=500, bbox_inches='tight')
# 显示图片
plt.show()
