# 使用瓦片编码 (Tile Coding) 的 Q-learning
# 这是强化学习中最经典、最高效的线性函数近似方法之一 (Local Approximation)
# 尤其适合在嵌入式设备或计算资源受限的场景下使用
# 核心思想：使用多层网格 (Tilings) 覆盖状态空间，每层网格之间有微小的偏移。
# 任何一个状态都会在每一层网格中激活一个 Tile，最终特征就是这些激活 Tile 的集合。

import gymnasium as gym
import numpy as np
from tensorboardX import SummaryWriter

# ==============================================================================
# 1. 超参数设置
# ==============================================================================
LEARNING_RATE = 0.1     # 学习率
GAMMA = 0.99            # 折扣因子
EPSILON_START = 1.0     # 初始探索率
EPSILON_END = 0.01      # 最终探索率
EPSILON_DECAY = 0.995   # 衰减系数

# Tile Coding 设置
NUM_TILINGS = 8         # 网格层数 (层数越多，分辨度越高，计算量越大)
NUM_BINS = 4            # 每个维度的划分格数 (例如 4^4 = 256 个格子)
# 总特征数 = NUM_TILINGS * (NUM_BINS ** 4)
# 8 * 256 = 2048 个特征 (非常稀疏，每次只有 8 个是 1)

BATCH_SIZE = 16
MAX_EPISODES = 1024

class TileCoder:
    """
    瓦片编码特征提取器
    """
    def __init__(self, env, n_tilings=8, n_bins=4):
        self.n_tilings = n_tilings
        self.n_bins = n_bins
        self.state_dim = env.observation_space.shape[0]
        
        # 定义状态空间范围 (需要根据经验设定，因为某些状态分量是 inf)
        self.low = np.array([-2.4, -4.0, -0.209, -4.0])
        self.high = np.array([2.4, 4.0, 0.209, 4.0])
        self.range = self.high - self.low
        
        # 计算每个 Tiling 的偏移量
        # 我们对每个维度施加不同的偏移，以打破对称性
        # 简单的策略：第 i 层网格在所有维度上偏移 i/n_tilings * bin_width
        self.offsets = []
        bin_width = 1.0 / n_bins
        for i in range(n_tilings):
            self.offsets.append(i * bin_width / n_tilings)
            
        # 计算每一层网格的特征总数 (n_bins ^ state_dim)
        self.features_per_tiling = n_bins ** self.state_dim
        self.n_features = n_tilings * self.features_per_tiling

    def get_active_indices(self, state):
        """
        输入状态，返回激活的特征索引列表 (Sparse Representation)
        """
        # 1. 归一化状态到 [0, 1]
        clipped_state = np.clip(state, self.low, self.high)
        normalized = (clipped_state - self.low) / self.range
        
        active_indices = []
        
        for i in range(self.n_tilings):
            # 2. 施加偏移
            # 注意：Tile Coding 的偏移通常是让网格动，也就是坐标减去偏移
            # 或者等价地：坐标加上偏移
            # 这里我们让坐标加上偏移
            offset = self.offsets[i]
            scaled = normalized + offset
            
            # 3. 离散化坐标
            # 确保索引在 [0, n_bins-1] 范围内
            indices = np.floor(scaled * self.n_bins).astype(int)
            indices = np.clip(indices, 0, self.n_bins - 1)
            
            # 4. 计算扁平化索引 (Flat Index)
            # 类似于多维数组转一维数组
            # index = d0 + d1*N + d2*N^2 + ...
            flat_index_in_tiling = 0
            for d in range(self.state_dim):
                flat_index_in_tiling += indices[d] * (self.n_bins ** d)
                
            # 加上这一层的基准偏移
            global_index = i * self.features_per_tiling + flat_index_in_tiling
            active_indices.append(global_index)
            
        return active_indices

class LinearTileAgent:
    """
    线性 Tile Coding 智能体
    Q(s, a) = sum(weights[a][active_indices])
    """
    def __init__(self, env):
        self.env = env
        self.n_actions = env.action_space.n
        
        self.coder = TileCoder(env, NUM_TILINGS, NUM_BINS)
        self.feature_dim = self.coder.n_features
        
        # 权重矩阵
        self.weights = np.zeros((self.n_actions, self.feature_dim))
        
    def get_q_value(self, active_indices, action):
        # 稀疏点积：只需要把激活索引对应的权重加起来即可
        # 速度极快
        return np.sum(self.weights[action][active_indices])

    def get_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            active_indices = self.coder.get_active_indices(state)
            q_values = [self.get_q_value(active_indices, a) for a in range(self.n_actions)]
            return np.argmax(q_values)

    def update(self, state, action, reward, next_state, done):
        """
        稀疏更新
        """
        active_indices = self.coder.get_active_indices(state)
        prediction = self.get_q_value(active_indices, action)
        
        if done:
            target = reward
        else:
            next_indices = self.coder.get_active_indices(next_state)
            next_q_values = [self.get_q_value(next_indices, a) for a in range(self.n_actions)]
            target = reward + GAMMA * np.max(next_q_values)
            
        error = target - prediction
        
        # 只更新激活的权重 (Sparse Update)
        # alpha / num_tilings 是为了平均化
        # 通常 tile coding 的学习率需要除以 tiling 数量
        step_size = LEARNING_RATE / NUM_TILINGS
        
        for idx in active_indices:
            self.weights[action][idx] += step_size * error

if __name__ == "__main__":
    # 显式指定 max_episode_steps=500
    env = gym.make("CartPole-v1", max_episode_steps=500)
    writer = SummaryWriter(comment="-tile-coding")
    
    agent = LinearTileAgent(env)
    
    epsilon = EPSILON_START
    episode_rewards = []
    running_reward = 0.0
    iter_no = 0 
    
    print(f"开始训练 Tile Coding Q-learning...")
    print(f"总特征数: {agent.feature_dim}, 激活特征数: {NUM_TILINGS}")
    
    for episode in range(MAX_EPISODES):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(obs, epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.update(obs, action, reward, next_obs, done)
            
            obs = next_obs
            total_reward += reward
            
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        episode_rewards.append(total_reward)
        
        if (episode + 1) % BATCH_SIZE == 0:
            batch_rewards = episode_rewards[-BATCH_SIZE:]
            reward_mean = np.mean(batch_rewards)
            reward_max = np.max(batch_rewards)
            
            if iter_no == 0:
                running_reward = reward_mean
            else:
                running_reward = 0.99 * running_reward + 0.01 * reward_mean
            
            print(f"{iter_no}: reward_mean={reward_mean:.1f}, reward_max={reward_max:.1f}, running_reward={running_reward:.1f}, epsilon={epsilon:.2f}", flush=True)
            
            writer.add_scalar("Reward/Mean", reward_mean, iter_no)
            writer.add_scalar("Reward/Running", running_reward, iter_no)
            
            iter_no += 1
            
        if running_reward > 475:
             print(f"Solved in {episode} episodes!")
             break

    writer.close()
