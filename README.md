# 多智能体强化学习系统

## 目录

- [项目概述](#项目概述)
  - [背景](#背景)
  - [应用场景](#应用场景)
  - [主要特点](#主要特点)
- [项目结构](#项目结构)
- [环境要求](#环境要求)
- [安装指南](#安装指南)
- [使用方法](#使用方法)
  - [训练新模型](#训练新模型)
  - [修改训练参数](#修改训练参数)
  - [环境参数调整](#环境参数调整)
  - [可视化结果](#可视化结果)
- [代码关联关系](#代码关联关系)
  - [代码流程图](#代码流程图)
  - [文件依赖关系](#文件依赖关系)
  - [算法执行流程](#算法执行流程)
- [算法说明](#算法说明)
  - [MASAC](#masac-multi-agent-soft-actor-critic)
  - [MATD3](#matd3-multi-agent-twin-delayed-deep-deterministic-policy-gradient)
- [环境设置](#环境设置)
- [实验结果](#实验结果)
  - [结果可视化](#结果可视化)
  - [主要性能指标](#主要性能指标)
  - [算法比较](#算法比较)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

## 项目概述

这个项目实现了多种多智能体强化学习算法，用于解决复杂的决策问题。主要包括MASAC（Multi-Agent Soft Actor-Critic）和MATD3（Multi-Agent Twin Delayed Deep Deterministic Policy Gradient）算法的实现和应用。

### 背景

多智能体强化学习是强化学习的一个重要分支，它研究多个智能体如何在共享环境中学习最优策略。与单智能体强化学习相比，多智能体场景面临更多挑战，如环境的非平稳性、智能体间的协作与竞争、部分可观察性等。

本项目聚焦于解决多智能体协作场景下的资源分配和任务调度问题，特别是在无线通信和边缘计算环境中。

### 应用场景

本项目的多智能体强化学习算法可应用于以下场景：

1. **无线通信资源分配**：多个基站协作为用户分配无线资源，优化系统整体性能
2. **边缘计算任务调度**：多个边缘服务器协作处理计算任务，减少延迟和能耗
3. **多机器人协作**：多个机器人协作完成任务，如搬运、探索等
4. **智能交通系统**：多个交通信号灯协作控制交通流，减少拥堵

### 主要特点

- **多种算法实现**：包含MASAC、MATD3、SAC、TD3等多种算法
- **可定制环境**：提供可定制的仿真环境，支持不同场景的测试
- **完整训练流程**：从环境交互、经验回放到网络更新的完整实现
- **结果可视化**：提供丰富的可视化工具，展示训练过程和结果
- **模块化设计**：各组件高度模块化，便于扩展和修改

## 项目结构

```
├── MASAC/                  # MASAC算法实现
│   ├── MASAC_main_07.py    # MASAC主程序
│   ├── MASAC_5072.py       # MASAC算法核心实现
│   ├── MASAC_network_5072.py # MASAC网络结构
│   ├── env_90.py           # 环境模拟
│   ├── replay_buffer.py    # 经验回放缓冲区
│   ├── normalization.py    # 状态标准化
│   ├── utils_07.py         # 工具函数
│   └── draw_agri_compa.py  # 结果可视化
├── MATD3_final/            # MATD3算法实现
│   ├── MATD3_main_519_1.py # MATD3主程序
│   ├── matd3_514.py        # MATD3算法核心实现
│   ├── networks.py         # 网络结构
│   ├── env_519.py          # 环境模拟
│   ├── replay_buffer.py    # 经验回放缓冲区
│   ├── normalization.py    # 状态标准化
│   ├── utils_07.py         # 工具函数
│   └── draw.py             # 结果可视化
├── SAC/                    # SAC算法实现
│   ├── SAC_main.py         # SAC主程序
│   ├── env_516.py          # 环境设置
│   ├── normalization.py    # 数据标准化工具
│   ├── utils_516_SAC.py    # 工具函数
│   └── draw.py             # 结果可视化工具
├── TD3/                    # TD3算法实现
│   ├── TD3_main.py         # TD3主程序
│   ├── env_516.py          # 环境设置
│   ├── normalization.py    # 数据标准化工具
│   ├── utils_516_SAC.py    # 工具函数
│   └── draw.py             # 结果可视化工具
└── draw_final/             # 结果可视化工具
    ├── draw.py             # 绘图主程序
    ├── draw_diff_buffer.py # 不同缓冲区结果对比
    └── MATD3_830/          # 实验数据存储目录
```

项目包含以下主要目录：

- **MASAC/**: 多智能体软演员-评论家算法实现
  - [`MASAC_5072.py`](./MASAC/MASAC_5072.py): MASAC算法核心实现，依赖于`MASAC_network_5072.py`中定义的网络结构
  - [`MASAC_network_5072.py`](./MASAC/MASAC_network_5072.py): MASAC网络结构定义，包含Actor和Critic网络
  - [`MASAC_main_07.py`](./MASAC/MASAC_main_07.py): MASAC算法主程序，调用`MASAC_5072.py`、`env_90.py`、`replay_buffer.py`和`normalization.py`
  - [`env_07.py`](./MASAC/env_07.py), [`env_90.py`](./MASAC/env_90.py), [`env_110.py`](./MASAC/env_110.py): 不同环境设置，提供智能体交互的仿真环境
  - [`normalization.py`](./MASAC/normalization.py): 数据标准化工具，用于状态和奖励的标准化处理
  - [`replay_buffer.py`](./MASAC/replay_buffer.py): 经验回放缓冲区，存储和采样智能体的交互经验
  - [`utils_07.py`](./MASAC/utils_07.py): 工具函数，提供状态生成和奖励计算等功能
  - [`draw_agri_compa.py`](./MASAC/draw_agri_compa.py): 结果可视化工具，用于绘制训练过程中的各项指标
  - 各种实验结果目录: `MASAC_code_5w/`, `MASAC_code_6w/`, 等，存储不同参数设置下的实验结果

- **MATD3_final/**: 多智能体双延迟深度确定性策略梯度算法实现
  - [`matd3_514.py`](./MATD3_final/matd3_514.py): MATD3算法核心实现，依赖于`networks.py`中定义的网络结构
  - [`networks.py`](./MATD3_final/networks.py): 网络结构定义，包含Actor和Critic网络
  - [`MATD3_main_519_1.py`](./MATD3_final/MATD3_main_519_1.py): MATD3算法主程序，调用`matd3_514.py`、`env_519.py`、`replay_buffer.py`和`normalization.py`
  - [`env_519.py`](./MATD3_final/env_519.py): 环境设置，提供智能体交互的仿真环境
  - [`maddpg.py`](./MATD3_final/maddpg.py): MADDPG算法实现，作为比较算法
  - [`normalization.py`](./MATD3_final/normalization.py): 数据标准化工具，与MASAC中的功能类似
  - [`replay_buffer.py`](./MATD3_final/replay_buffer.py): 经验回放缓冲区，与MASAC中的功能类似
  - [`utils_07.py`](./MATD3_final/utils_07.py): 工具函数，提供状态生成和奖励计算等功能
  - [`draw.py`](./MATD3_final/draw.py): 结果可视化工具，用于绘制训练过程中的各项指标
  - 各种实验结果目录: `MATD3_quanzhong_5w/`, `MATD3_quanzhong_6w/`, 等，存储不同参数设置下的实验结果

- **SAC/**: 单智能体软演员-评论家算法实现
  - [`SAC_main.py`](./SAC/SAC_main.py): SAC算法主程序和实现
  - [`env_516.py`](./SAC/env_516.py): 环境设置
  - [`normalization.py`](./SAC/normalization.py): 数据标准化工具
  - [`utils_516_SAC.py`](./SAC/utils_516_SAC.py): 工具函数
  - [`draw.py`](./SAC/draw.py): 结果可视化工具

- **TD3/**: 单智能体双延迟深度确定性策略梯度算法实现
  - [`TD3_main.py`](./TD3/TD3_main.py): TD3算法主程序和实现
  - [`env_516.py`](./TD3/env_516.py): 环境设置
  - [`normalization.py`](./TD3/normalization.py): 数据标准化工具
  - [`utils_516_SAC.py`](./TD3/utils_516_SAC.py): 工具函数
  - [`draw.py`](./TD3/draw.py): 结果可视化工具

- **draw_final/**: 最终结果可视化工具
  - [`draw.py`](./draw_final/draw.py): 综合结果可视化工具
  - [`draw_diff_buffer.py`](./draw_final/draw_diff_buffer.py): 不同缓冲区大小的结果比较工具
  - `MATD3_830/`: 存储用于最终可视化的数据

## 环境要求

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- TensorBoard

## 安装指南

1. 克隆仓库：
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. 安装依赖：
   ```bash
   pip install torch numpy matplotlib tensorboard
   ```

## 使用方法

### 训练新模型

```bash
# 训练MASAC模型
python MASAC/MASAC_main_07.py

# 训练MATD3模型
python MATD3_final/MATD3_main_519_1.py

# 训练单智能体SAC模型
python SAC/SAC_main.py

# 训练单智能体TD3模型
python TD3/TD3_main.py
```

### 修改训练参数

您可以通过修改主程序文件中的参数来调整训练过程：

#### MASAC参数调整 ([`MASAC_main_07.py`](./MASAC/MASAC_main_07.py))

```python
# 修改智能体数量
N = 5  # 智能体数量

# 修改训练轮数
max_train_steps = 5e6  # 最大训练步数

# 修改经验回放缓冲区大小
buffer_size = 5e5  # 缓冲区大小

# 修改批量大小
batch_size = 256  # 批量大小
```

#### MATD3参数调整 ([`MATD3_main_519_1.py`](./MATD3_final/MATD3_main_519_1.py))

```python
# 修改智能体数量
N = 5  # 智能体数量

# 修改训练轮数
max_train_steps = 5e6  # 最大训练步数

# 修改经验回放缓冲区大小
buffer_size = 5e5  # 缓冲区大小

# 修改批量大小
batch_size = 256  # 批量大小

# 修改策略噪声
policy_noise = 0.2  # 策略噪声
```

### 环境参数调整

您可以通过修改环境文件来调整仿真环境参数：

#### MASAC环境参数 ([`env_90.py`](./MASAC/env_90.py))

```python
# 修改基站位置
self.BS_positions = np.array([[0, 0], [200, 0], [0, 200], [200, 200], [100, 100]])

# 修改用户数量
self.UE_num = 10

# 修改带宽
self.bandwidth = 1e6  # 1MHz
```

#### MATD3环境参数 ([`env_519.py`](./MATD3_final/env_519.py))

```python
# 修改基站位置
self.BS_positions = np.array([[0, 0], [200, 0], [0, 200], [200, 200], [100, 100]])

# 修改用户数量
self.UE_num = 10

# 修改带宽
self.bandwidth = 1e6  # 1MHz
```

### 可视化结果

```bash
# 可视化MASAC结果
python MASAC/draw_agri_compa.py

# 可视化MATD3结果
python MATD3_final/draw.py

# 比较不同算法结果
python draw_final/draw.py

# 比较不同缓冲区大小的结果
python draw_final/draw_diff_buffer.py
```

### 自定义可视化

您可以通过修改绘图文件来自定义可视化效果：

```python
# 在draw.py中修改图表样式
plt.figure(figsize=(10, 6))  # 修改图表大小
plt.grid(True)  # 添加网格线
plt.title('自定义标题')  # 修改标题

# 修改要加载的数据文件
data = np.load('path/to/your/data.npy')  # 加载自定义数据
```

## 代码关联关系

### 代码流程图

#### MASAC算法流程

```
+-------------------+     +-------------------+     +-------------------+
|                   |     |                   |     |                   |
| MASAC_main_07.py  |---->| MASAC_5072.py     |---->| MASAC_network_5072.py |
| (主程序)          |     | (算法实现)        |     | (网络结构定义)    |
|                   |     |                   |     |                   |
+-------------------+     +-------------------+     +-------------------+
        |                          ^
        |                          |
        v                          |
+-------------------+     +-------------------+     +-------------------+
|                   |     |                   |     |                   |
| env_90.py         |---->| replay_buffer.py  |---->| normalization.py  |
| (环境模拟)        |     | (经验回放)        |     | (数据标准化)      |
|                   |     |                   |     |                   |
+-------------------+     +-------------------+     +-------------------+
        |                          ^
        |                          |
        v                          |
+-------------------+     +-------------------+
|                   |     |                   |
| utils_07.py       |---->| draw_agri_compa.py|
| (工具函数)        |     | (结果可视化)      |
|                   |     |                   |
+-------------------+     +-------------------+
```

#### MATD3算法流程

```
+-------------------+     +-------------------+     +-------------------+
|                   |     |                   |     |                   |
| MATD3_main_519_1.py|---->| matd3_514.py      |---->| networks.py       |
| (主程序)          |     | (算法实现)        |     | (网络结构定义)    |
|                   |     |                   |     |                   |
+-------------------+     +-------------------+     +-------------------+
        |                          ^
        |                          |
        v                          |
+-------------------+     +-------------------+     +-------------------+
|                   |     |                   |     |                   |
| env_519.py        |---->| replay_buffer.py  |---->| normalization.py  |
| (环境模拟)        |     | (经验回放)        |     | (数据标准化)      |
|                   |     |                   |     |                   |
+-------------------+     +-------------------+     +-------------------+
        |                          ^
        |                          |
        v                          |
+-------------------+     +-------------------+
|                   |     |                   |
| utils_07.py       |---->| draw.py           |
| (工具函数)        |     | (结果可视化)      |
|                   |     |                   |
+-------------------+     +-------------------+
```

### 文件依赖关系

1. **MASAC算法依赖关系**：
   - [`MASAC_main_07.py`](./MASAC/MASAC_main_07.py) 是主程序，调用 [`MASAC_5072.py`](./MASAC/MASAC_5072.py) 创建智能体
   - [`MASAC_5072.py`](./MASAC/MASAC_5072.py) 依赖 [`MASAC_network_5072.py`](./MASAC/MASAC_network_5072.py) 中定义的网络结构
   - [`MASAC_main_07.py`](./MASAC/MASAC_main_07.py) 使用 [`env_90.py`](./MASAC/env_90.py) 创建环境
   - [`replay_buffer.py`](./MASAC/replay_buffer.py) 提供经验回放功能，被 [`MASAC_main_07.py`](./MASAC/MASAC_main_07.py) 调用
   - [`normalization.py`](./MASAC/normalization.py) 提供状态标准化功能，被 [`MASAC_main_07.py`](./MASAC/MASAC_main_07.py) 调用
   - [`utils_07.py`](./MASAC/utils_07.py) 提供工具函数，被 [`MASAC_main_07.py`](./MASAC/MASAC_main_07.py) 和环境调用
   - [`draw_agri_compa.py`](./MASAC/draw_agri_compa.py) 用于可视化结果

2. **MATD3算法依赖关系**：
   - [`MATD3_main_519_1.py`](./MATD3_final/MATD3_main_519_1.py) 是主程序，调用 [`matd3_514.py`](./MATD3_final/matd3_514.py) 创建智能体
   - [`matd3_514.py`](./MATD3_final/matd3_514.py) 依赖 [`networks.py`](./MATD3_final/networks.py) 中定义的网络结构
   - [`MATD3_main_519_1.py`](./MATD3_final/MATD3_main_519_1.py) 使用 [`env_519.py`](./MATD3_final/env_519.py) 创建环境
   - [`replay_buffer.py`](./MATD3_final/replay_buffer.py) 提供经验回放功能，被 [`MATD3_main_519_1.py`](./MATD3_final/MATD3_main_519_1.py) 调用
   - [`normalization.py`](./MATD3_final/normalization.py) 提供状态标准化功能，被 [`MATD3_main_519_1.py`](./MATD3_final/MATD3_main_519_1.py) 调用
   - [`utils_07.py`](./MATD3_final/utils_07.py) 提供工具函数，被 [`MATD3_main_519_1.py`](./MATD3_final/MATD3_main_519_1.py) 和环境调用
   - [`draw.py`](./MATD3_final/draw.py) 用于可视化结果

3. **算法之间的共享组件**：
   - `normalization.py`：不同算法目录下的标准化工具功能相似
   - `replay_buffer.py`：不同算法目录下的经验回放缓冲区功能相似
   - `utils_07.py`：工具函数在不同算法中被共享或有相似实现

### 算法执行流程

#### MASAC算法执行流程

1. **初始化阶段**：
   - [`MASAC_main_07.py`](./MASAC/MASAC_main_07.py) 创建环境 [`env_90.py`](./MASAC/env_90.py)
   - 初始化经验回放缓冲区 [`replay_buffer.py`](./MASAC/replay_buffer.py)
   - 创建MASAC智能体 [`MASAC_5072.py`](./MASAC/MASAC_5072.py)，该智能体使用 [`MASAC_network_5072.py`](./MASAC/MASAC_network_5072.py) 中定义的网络结构
   - 初始化状态标准化器 [`normalization.py`](./MASAC/normalization.py)

2. **训练阶段**：
   - 环境生成初始状态
   - 智能体根据状态选择动作
   - 环境执行动作并返回奖励、下一状态
   - 经验存储到回放缓冲区
   - 从回放缓冲区采样批量数据进行训练
   - 更新Actor和Critic网络
   - 自适应调整温度参数alpha

3. **评估与可视化**：
   - 定期评估智能体性能
   - 使用 [`draw_agri_compa.py`](./MASAC/draw_agri_compa.py) 可视化训练结果

#### MATD3算法执行流程

1. **初始化阶段**：
   - [`MATD3_main_519_1.py`](./MATD3_final/MATD3_main_519_1.py) 创建环境 [`env_519.py`](./MATD3_final/env_519.py)
   - 初始化经验回放缓冲区 [`replay_buffer.py`](./MATD3_final/replay_buffer.py)
   - 创建MATD3智能体 [`matd3_514.py`](./MATD3_final/matd3_514.py)，该智能体使用 [`networks.py`](./MATD3_final/networks.py) 中定义的网络结构
   - 初始化状态标准化器 [`normalization.py`](./MATD3_final/normalization.py)

2. **训练阶段**：
   - 环境生成初始状态
   - 智能体根据状态选择动作
   - 环境执行动作并返回奖励、下一状态
   - 经验存储到回放缓冲区
   - 从回放缓冲区采样批量数据进行训练
   - 更新Critic网络
   - 延迟更新Actor网络（每两次Critic更新后更新一次Actor）
   - 软更新目标网络

3. **评估与可视化**：
   - 定期评估智能体性能
   - 使用 [`draw.py`](./MATD3_final/draw.py) 可视化训练结果

## 算法说明

### MASAC (Multi-Agent Soft Actor-Critic)

MASAC是一种基于最大熵强化学习的多智能体算法，它结合了策略梯度和Q-learning的优点。该算法通过最大化累积奖励和策略熵的加权和来优化策略，有助于探索和避免过早收敛到次优策略。

**核心特点**：
- 使用随机策略而非确定性策略
- 包含熵正则化项，鼓励探索
- 使用软状态值函数和软Q函数
- 支持自动调整温度参数alpha

### MATD3 (Multi-Agent Twin Delayed Deep Deterministic Policy Gradient)

MATD3是TD3算法的多智能体扩展版本，它通过以下机制解决了DDPG算法中的过估计问题：
1. 使用双Q网络：维护两个Q网络，取较小值减轻过估计
2. 延迟策略更新：减少策略更新频率，提高稳定性
3. 目标策略平滑：在目标动作中添加噪声，减少策略过拟合

**核心特点**：
- 使用确定性策略
- 采用Actor-Critic架构
- 通过双Q网络减轻过估计问题
- 延迟更新策略网络，提高训练稳定性

**应用案例**：

为了解决无人机辅助空地一体化网络中的局部拥塞和资源分配不平衡问题，本研究构建了一个由地面边缘服务器、无人机和用户组成的三层网络架构。提出了一个两阶段优化框架，包含节点选择和任务卸载策略。首先，为了缓解多用户动态分布带来的计算资源限制，开发了基于进化博弈的节点选择算法，实现卸载节点间的有效动态负载均衡。随后，将联合任务卸载和功率分配问题建模为马尔可夫决策过程，设计了基于MATD3的强化学习算法，实现用户卸载比例和传输功率水平的联合最优控制。仿真结果表明，所提出的框架分别减少了约33.4%的总延迟和29.4%的能耗，优于现有策略。该框架在任务密集型场景中展现出优越的可扩展性和能源效率。

**Application Case (English)**:

To address the challenges of local congestion and imbalanced resource allocation in UAV-assisted air-ground integrated networks, this study constructs a three-layer network architecture comprising ground edge servers, UAVs, and users. A two-stage optimization framework is proposed, incorporating the node selection and task offloading strategies. First, to mitigate the computational resource limitations arising from the dynamic distribution of multiple users, an evolutionary game-based node selection algorithm is developed to achieve effective dynamic load balancing across offloading nodes. Subsequently, the joint task offloading and power allocation problem is formulated as a Markov decision process, and a reinforcement learning algorithm based on MATD3 is designed to attain the joint optimal control of user offloading ratios and transmission power levels. Simulation results demonstrate that the proposed framework reduces total delay and energy consumption by approximately 33.4% and 29.4%, respectively, outperforming the existing strategies. The framework demonstrates superior scalability and energy efficiency in task-intensive scenarios.

## 环境设置

项目中的环境模拟了多智能体在特定场景下的交互，包括：
- 多个智能体（默认为5个）
- 每个智能体有自己的观察空间和动作空间
- 智能体需要协作或竞争以优化整体性能

**环境参数**：
- 观察维度：每个智能体3维
- 动作维度：每个智能体3维
- 奖励计算：考虑延迟、能量消耗等因素

## 实验结果

项目包含多种算法在不同环境参数下的实验结果，主要存储在各算法目录下的子文件夹中。

### 结果可视化

- MASAC算法结果可通过 [`MASAC/draw_agri_compa.py`](./MASAC/draw_agri_compa.py) 可视化
- MATD3算法结果可通过 [`MATD3_final/draw.py`](./MATD3_final/draw.py) 可视化
- 不同算法的比较结果可通过 [`draw_final/draw.py`](./draw_final/draw.py) 和 [`draw_final/draw_diff_buffer.py`](./draw_final/draw_diff_buffer.py) 可视化

### 主要性能指标

1. **奖励曲线**：展示训练过程中智能体获得的平均奖励变化
2. **延迟指标**：展示系统中的任务处理延迟
3. **能耗指标**：展示系统的能量消耗情况
4. **吞吐量**：展示系统处理任务的效率

### 算法比较

项目中实现的多种算法（MASAC、MATD3、SAC、TD3）在相同环境下的性能比较：

1. **收敛速度**：MASAC和MATD3通常比单智能体算法收敛更快
2. **稳定性**：MATD3由于使用双Q网络和延迟更新，通常比MASAC更稳定
3. **最终性能**：多智能体算法在协作任务中通常优于单智能体算法

实验结果保存在各个算法目录下的子文件夹中，包括：
- 奖励曲线
- 能量消耗
- 延迟数据
- TensorBoard事件文件

可以使用提供的绘图工具可视化这些结果。

## 贡献指南

欢迎对项目进行贡献！请遵循以下步骤：

1. Fork本仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启一个Pull Request

### 代码风格

- 请遵循PEP 8 Python代码风格指南
- 为所有函数和类添加适当的文档字符串
- 使用有意义的变量名和函数名
- 添加必要的注释，特别是对于复杂的算法部分

### 添加新算法

如果您想添加新的强化学习算法，请遵循以下结构：

1. 在项目根目录下创建新的算法目录
2. 实现算法的核心类和网络结构
3. 创建主程序文件，用于训练和测试
4. 添加环境文件，确保与现有环境兼容
5. 添加可视化工具，用于展示结果
6. 更新README.md，添加新算法的说明

## 许可证

本项目采用MIT许可证 - 详情请参见LICENSE文件

```
MIT License

Copyright (c) 2023 Multi-Agent Reinforcement Learning Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 总结

本项目提供了一个完整的多智能体强化学习框架，实现了MASAC和MATD3等先进算法，并提供了可定制的环境和丰富的可视化工具。通过本项目，您可以：

1. 学习和理解多智能体强化学习算法的原理和实现
2. 在自定义环境中测试和比较不同算法的性能
3. 将这些算法应用到实际问题中，如无线通信资源分配、边缘计算任务调度等
4. 基于现有框架开发新的多智能体强化学习算法

希望本项目能够帮助您更好地理解和应用多智能体强化学习技术。如有任何问题或建议，欢迎通过Issue或Pull Request与我们交流。