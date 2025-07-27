# Task Offloading and Power Allocation in UAV-Assisted Air-Ground Network MEC Systems

## 1. Project Overview

To address the challenges of local congestion and imbalanced resource allocation in UAV-assisted air-ground integrated networks, this project codes a three-layer network architecture, and a two-stage optimization framework, incorporating the node selection and task offloading strategies. The framework demonstrates superior scalability and energy efficiency in task-intensive scenarios.

## 2. Environment Setup

### 2.1 Dependencies

This project is built upon the Python 3.8 environment and utilizes PyTorch 1.19.0 for building and training deep learning models. The main dependencies and their versions required to run the code are as follows:

-   Python 3.8
-   PyTorch 1.19.0
-   NumPy
-   TensorBoard

### 2.2 Installation

You can install all necessary dependencies using the following command:

```bash
pip install torch==1.19.0 numpy tensorboard
```

## 3. Code Structure

The project code is well-organized, with core functional modules located in the `代码/MATD3_final/` directory. Below is a description of the key files and their functionalities:

```
代码/
├── MATD3_final/                          # MATD3 algorithm implementation and related files
│   ├── MATD3_main_519_1.py             # Main entry point for the project, responsible for algorithm initialization, training, and evaluation
│   ├── matd3_514.py                    # Core implementation of the MATD3 algorithm
│   ├── env_519.py                      # Simulation environment definition, including system, communication, and computing models
│   ├── networks.py                     # Defines the neural network structures for Actor and Critic networks
│   ├── replay_buffer.py                # Experience replay buffer for storing and sampling training experiences
│   ├── normalization.py                # Data normalization processing module
│   └── utils_07.py                     # Auxiliary functions, such as reward calculation
├── MASAC/                                # MASAC algorithm implementation
├── SAC/                                  # SAC algorithm implementation
├── TD3/                                  # TD3 algorithm implementation
└── draw_final/                           # Plotting tools for results
```

## 4. Evolutionary Game-Based Node Selection

Before task offloading, users need to select appropriate offloading nodes. An efficient node selection scheme is crucial for balancing server load and optimizing network system metrics. This project models the node selection problem within an evolutionary game framework, specifically detailing the following elements:

-   **Players:** All users within the coverage of base stations and UAVs.
-   **Strategies:** The base stations and UAVs that users can choose to connect to.
-   **Population:** All users within the coverage of the same UAV form a population.
-   **Population State:** Represents the proportion of users selecting a specific node within a population.
-   **Utility Function:** Defined based on the computing resources and load of the nodes, aiming to characterize the degree of congestion at the nodes.

Through the evolutionary game, the system can dynamically adjust node selection strategies, allowing the players (users) to reach an equilibrium state, thereby achieving load balancing and resource optimization. For a detailed evolutionary game algorithm flow, please refer to the code implementation.

## 5. Running Guide

### 4.1 Data Preparation

1.  **Extract Code:** First, please extract the provided `代码.rar` file to your project's root directory. After extraction, you will find a folder named `代码` containing all the source code.

### 4.2 Training and Simulation

1.  **Start Training:** After configuring the environment, you can start the algorithm training and simulation process by running the `MATD3_main_519_1.py` file:

    ```bash
    python 代码/MATD3_final/MATD3_main_519_1.py
    ```

2.  **Parameter Adjustment:** You can find and adjust various hyperparameters in the `MATD3_main_519_1.py` file to suit different simulation requirements or for performance tuning. For example, you can modify `--max_train_steps` (maximum training steps), `--batch_size` (batch size), `--lr_a` (Actor learning rate), `--lr_c` (Critic learning rate), etc.

### 4.3 Result Visualization

During training, the project will generate log files and result data in `.npy` format, which record key metrics during the training process. You can visualize the results using the following methods:

1.  **TensorBoard:** Use TensorBoard to monitor training curves and changes in various metrics in real-time:

    ```bash
    tensorboard --logdir 代码/MATD3_final/MATD3_Reward_ave_10w_nse2
    ```

    **Note:** The `--logdir` path may need to be adjusted based on the actual log directory generated during runtime. Typically, log files are saved in a subdirectory starting with `runs/` within `代码/MATD3_final/`.

2.  **Custom Plotting:** The `draw_final/` directory provides Python scripts for plotting results. You can modify these scripts as needed to generate custom performance charts.

## 5. Simulation Parameters

In the `MATD3_main_519_1.py` file, various simulation and algorithm hyperparameters are defined using the `argparse` module. Here are some key parameters and their descriptions:

| Parameter            | Description                                  | Default Value |
| :------------------- | :------------------------------------------- | :------------ |
| `--max_train_steps`  | Maximum training steps for the algorithm     | 8000          |
| `--max_action`       | Maximum value for the agent's action space   | 0.90          |
| `--algorithm`        | Reinforcement learning algorithm to use (MATD3 or MADDPG) | MATD3         |
| `--buffer_size`      | Capacity of the experience replay buffer     | 8000          |
| `--batch_size`       | Batch size for sampling experiences in each training iteration | 128           |
| `--lr_a`             | Learning rate for the Actor network          | 0.001         |
| `--lr_c`             | Learning rate for the Critic network         | 0.0015        |
| `--gamma`            | Discount factor, used to calculate the present value of future rewards | 0.95          |
| `--tau`              | Parameter for soft updating the target network | 0.005         |
| `--policy_noise`     | Standard deviation of noise added in target policy smoothing | 0.1           |
| `--noise_clip`       | Range for noise clipping                     | 0.5           |
| `--policy_update_freq` | Frequency of policy network updates          | 2             |

## 6. Result Analysis

The `.npy` files generated by the project (e.g., `MATD3_Reward_ave_1_1003_1.npy`, `MATD3_latency_average_1003_1.npy`, `MATD3_energy_evergy_1003_1.npy`, etc.) contain key performance indicator data such as rewards, task latency, and energy consumption during training. This data can be used to reproduce the simulation results from the paper and conduct in-depth analysis of the performance advantages of the proposed evolutionary game-based node selection algorithm and MATD3 task offloading algorithm in balancing server load, reducing user energy consumption, and optimizing system latency.

## 7. Citation

If you use the code or methods from this project in your research, please consider citing the following paper:

[Paper citation information to be added]

## 8. Acknowledgments

Thanks to all contributors who provided help and support for this project.

