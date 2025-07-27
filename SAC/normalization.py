import numpy as np

# 动态计算mean和std的class
# 核心思想是已知 n 个数据的 mean 和 std，如何计算 n+1 个数据的 mean 和 std
class RunningMeanStd:
    # 动态计算平均值和标准值
    def __init__(self, shape):  # shape: the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1  # 记录更新的次数
        if self.n == 1:   # 只有一个状态
            self.mean = x  # 第一次把 x 本身给了 self.mean
            self.std = x
        else:             # 存在多个历史状态
            old_mean = self.mean.copy()   # 复制老的平均值
            self.mean = old_mean + (x - old_mean) / self.n    # 计算新的平均值
            self.S = self.S + (x - old_mean) * (x - self.mean)   # 计算新的方差
            self.std = np.sqrt(self.S / self.n)   # 计算新的标准差

# 实例化上面的RunningMeanStd，需要传入的参数shape代表当前环境的状态空间的维度
class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):  # x-就是传入的状态
        # Whether to update the mean and std,during the evaluating,update=False
        # 在评估过程中，是否更新平均值和标准值，update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):  # x-刚才传入的奖励
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)
