# 使用表格型 Q-learning (Tabular Q-learning) 解决 CartPole 问题
# 这是一个 Value-Based 的方法 (Model-Free)，与 CEM (Policy-Based/Evolutionary) 形成对比
# CartPole 的状态空间是连续的，需要进行离散化 (Discretization) 才能使用 Q-table

import gymnasium as gym
import numpy as np
import math
from tensorboardX import SummaryWriter

# ==============================================================================
# 1. 超参数设置 (Hyperparameters)
# ==============================================================================
LEARNING_RATE = 0.1     # 学习率 (Alpha): 决定了新信息覆盖旧信息的程度
GAMMA = 0.99            # 折扣因子 (Discount Factor): 衡量未来奖励的重要性 (0~1)
EPSILON_START = 1.0     # 初始探索率 (Initial Epsilon): 开始时完全随机探索
EPSILON_END = 0.01      # 最终探索率 (Final Epsilon): 训练后期保留少量探索
EPSILON_DECAY = 0.995   # 探索率衰减系数: 每经过一个 Episode，Epsilon * 0.995

# ==============================================================================
# 2. 离散化设置 (Discretization)
# ==============================================================================
# CartPole 的状态包含 4 个连续变量。为了构建 Q-table，必须将其“分桶” (Binning)。
# 状态定义: [位置, 速度, 角度, 角速度]
BINS = [
    np.linspace(-2.4, 2.4, 20),      # 位置 (Cart Position): 划分为 20 个区间
    np.linspace(-4.0, 4.0, 20),      # 速度 (Cart Velocity): 范围扩大到 ±4.0
    np.linspace(-0.209, 0.209, 20),  # 角度 (Pole Angle): 划分为 20 个区间 (~ ±12度)
    np.linspace(-4.0, 4.0, 20)       # 角速度 (Pole Velocity): 范围扩大到 ±4.0
]

# ==============================================================================
# 3. 对比实验设置 (Comparison Setup)
# ==============================================================================
# CEM 是基于"代" (Generation) 的，每一代跑 Batch Size 个 Episode。
# Q-learning 是基于"回合" (Episode) 的，逐回合更新。
# 为了在 TensorBoard 上能够横向对比 (X轴对齐)，我们每隔 BATCH_SIZE 个 Episode 记录一次数据点，
# 这样 Q-learning 的一个"数据点"就对应 CEM 的"一代"。
BATCH_SIZE = 16         # 对应 CEM 的 Batch Size
MAX_EPISODES = 3000     # 总共训练多少个 Episode (增加一些以保证收敛)

class QLearner:
    """
    Q-learning 智能体
    维护一个 Q-table: Q(s, a)，用于评估在状态 s 下采取动作 a 的价值。
    """
    def __init__(self, n_actions):
        self.n_actions = n_actions
        # 初始化 Q-table
        # 维度: [位置bins, 速度bins, 角度bins, 角速度bins, 动作数]
        # 也就是 [21, 21, 21, 21, 2] 的一个 5维数组
        # 使用 +1 是因为 np.digitize 返回的索引范围是 0 到 len(bins) (包含越界情况)
        self.q_table = np.random.uniform(low=-1, high=1, size=([len(b) + 1 for b in BINS] + [n_actions]))

    def get_discrete_state(self, state):
        """
        状态离散化 (Discretization)
        将连续的 Observation 向量映射到 Q-table 的索引坐标。
        """
        discrete_state = []
        for i in range(len(state)):
            # np.digitize 返回值:
            # 0: x < bins[0]
            # 1: bins[0] <= x < bins[1]
            # ...
            # len(bins): x >= bins[-1]
            discrete_state.append(np.digitize(state[i], BINS[i]))
        return tuple(discrete_state)

    def get_action(self, state, epsilon):
        """
        Epsilon-Greedy 策略 (探索 vs 利用)
        以 epsilon 的概率随机探索，以 1-epsilon 的概率选择当前最优动作。
        """
        if np.random.random() < epsilon:
            return np.random.randint(0, self.n_actions) # 探索 (Exploration)
        else:
            discrete_state = self.get_discrete_state(state)
            return np.argmax(self.q_table[discrete_state]) # 利用 (Exploitation)

    def update(self, state, action, reward, next_state, done):
        """
        Q-learning 核心更新公式 (Off-Policy TD Control):
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
        """
        discrete_state = self.get_discrete_state(state)
        discrete_next_state = self.get_discrete_state(next_state)
        
        # 获取当前 Q 值
        current_q = self.q_table[discrete_state + (action,)]
        
        # 计算目标 Q 值 (TD Target)
        if done:
            max_future_q = 0.0 # 终止状态没有未来奖励
        else:
            max_future_q = np.max(self.q_table[discrete_next_state])
            
        # 贝尔曼方程更新
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + GAMMA * max_future_q)
        
        # 更新 Q-table
        self.q_table[discrete_state + (action,)] = new_q

if __name__ == "__main__":
    # 显式指定 max_episode_steps=500，防止某些环境下默认为 200
    env = gym.make("CartPole-v1", max_episode_steps=500)
    print(f"Environment initialized with max_episode_steps={env.spec.max_episode_steps}")
    
    writer = SummaryWriter(comment="-q-learning") # 日志后缀，用于 TensorBoard区分
    
    agent = QLearner(env.action_space.n)
    
    epsilon = EPSILON_START
    episode_rewards = []
    running_reward = 0.0
    
    # iter_no 用于模拟 CEM 的 "Iteration" 计数，方便 TensorBoard 对齐
    iter_no = 0 
    
    print("开始训练 Q-learning Agent...")
    print(f"参数设置: Alpha={LEARNING_RATE}, Gamma={GAMMA}, Bins={len(BINS[0])}")
    
    for episode in range(MAX_EPISODES):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # 1. 选择动作
            action = agent.get_action(obs, epsilon)
            
            # 2. 执行动作
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 3. 更新 Q-table
            agent.update(obs, action, reward, next_obs, done)
            
            # 状态流转
            obs = next_obs
            total_reward += reward
            
        # 衰减 Epsilon (线性衰减或指数衰减均可，这里用指数衰减)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        
        episode_rewards.append(total_reward)
        
        # ======================================================================
        # 日志记录与监控
        # ======================================================================
        # 每隔 BATCH_SIZE 个 Episode 汇总一次数据，作为一个 "Data Point" 记录到 TensorBoard
        if (episode + 1) % BATCH_SIZE == 0:
            batch_rewards = episode_rewards[-BATCH_SIZE:]
            reward_mean = np.mean(batch_rewards)
            reward_max = np.max(batch_rewards)
            reward_min = np.min(batch_rewards)
            
            # 计算 Reward Bound (为了与 CEM 对比，虽然 Q-learning 不用这个做筛选)
            reward_bound = np.percentile(batch_rewards, 70) 
            
            if iter_no == 0:
                running_reward = reward_mean
            else:
                running_reward = 0.99 * running_reward + 0.01 * reward_mean
            
            print("%d: reward_mean=%.1f, reward_max=%.1f, running_reward=%.1f, epsilon=%.2f" % (
                iter_no, reward_mean, reward_max, running_reward, epsilon), flush=True)
            
            # 使用统一的 Key 以便横向对比
            writer.add_scalar("Reward/Mean", reward_mean, iter_no)
            writer.add_scalar("Reward/Max", reward_max, iter_no)
            writer.add_scalar("Reward/Min", reward_min, iter_no)
            writer.add_scalar("Reward/Bound", reward_bound, iter_no) 
            writer.add_scalar("Reward/Running", running_reward, iter_no)
            writer.add_scalar("Training/Epsilon", epsilon, iter_no)
            
            iter_no += 1
            
        # 简单的提前停止条件
        if running_reward > 475:
             print(f"Solved in {episode} episodes! Final Running Reward: {running_reward}")
             break

    writer.close()
    print("训练结束。")
