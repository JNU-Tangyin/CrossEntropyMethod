# 使用傅里叶基 (Fourier Basis) 的 Q-learning
# 这是另一种强大的线性函数近似方法 (Global Approximation)
# 参考文献: Konidaris, G., Osentoski, S., & Thomas, P. (2011). Value Function Approximation in Reinforcement Learning using the Fourier Basis.

import gymnasium as gym
import numpy as np
import itertools
from tensorboardX import SummaryWriter
import sklearn.preprocessing

# ==============================================================================
# 1. 超参数设置
# ==============================================================================
LEARNING_RATE = 0.01    # 傅里叶基通常需要较小的学习率
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# 傅里叶基阶数 (Order)
# 阶数越高，能拟合的频率越高（细节越多），但特征维度也会指数增长
# 特征数 = (order + 1) ^ state_dim
# CartPole 状态维度为 4。
# Order=3 -> 4^4 = 256 特征
# Order=5 -> 6^4 = 1296 特征
FOURIER_ORDER = 3

BATCH_SIZE = 16
MAX_EPISODES = 1024

class FourierFeaturizer:
    """
    傅里叶基特征提取器
    原理: 将状态映射到 [0,1] 区间，然后计算 cos(pi * c * s)
    """
    def __init__(self, env, order):
        self.order = order
        self.state_dim = env.observation_space.shape[0]
        
        # 1. 生成系数向量 c (Coefficients)
        # 这是一个生成所有可能的 [0, ..., order] 组合的过程
        # 例如 2维状态, order=1: [[0,0], [0,1], [1,0], [1,1]]
        iter_range = range(order + 1)
        self.coefficients = np.array(list(itertools.product(iter_range, repeat=self.state_dim)))
        
        self.n_features = len(self.coefficients)
        
        # 2. 状态归一化范围 (Min-Max Scaling)
        # 傅里叶基要求输入在 [0, 1] 之间
        # CartPole 的边界其实是定义的，但速度项可能越界，我们需要估计一个合理范围
        # Position: [-2.4, 2.4] (终止条件) -> 放宽到 [-4.8, 4.8]
        # Velocity: [-inf, inf] -> 估计 [-4, 4]
        # Angle: [-.209, .209] (终止条件) -> 放宽到 [-.418, .418]
        # Angular Velocity: [-inf, inf] -> 估计 [-4, 4]
        self.state_low = np.array([-2.4, -4.0, -0.209, -4.0])
        self.state_high = np.array([2.4, 4.0, 0.209, 4.0])
        self.state_range = self.state_high - self.state_low

    def transform(self, state):
        # 1. Min-Max 归一化到 [0, 1]
        # np.clip 保证状态不会超出我们预设的范围
        clipped_state = np.clip(state, self.state_low, self.state_high)
        normalized_state = (clipped_state - self.state_low) / self.state_range
        
        # 2. 计算傅里叶特征
        # phi_i(s) = cos(pi * c_i . s)
        # dot product: (N_features, State_dim) dot (State_dim, ) -> (N_features, )
        dot_products = np.dot(self.coefficients, normalized_state)
        features = np.cos(np.pi * dot_products)
        
        return features

class LinearQLearner:
    """
    线性 Q-learning (与 RBF 版本完全通用)
    """
    def __init__(self, env):
        self.env = env
        self.n_actions = env.action_space.n
        
        # 使用傅里叶特征提取器
        self.featurizer = FourierFeaturizer(env, FOURIER_ORDER)
        self.feature_dim = self.featurizer.n_features
        
        # 初始化权重
        self.weights = np.zeros((self.n_actions, self.feature_dim))
        
    def get_q_value(self, state_features, action):
        return np.dot(self.weights[action], state_features)

    def get_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            features = self.featurizer.transform(state)
            q_values = [self.get_q_value(features, a) for a in range(self.n_actions)]
            return np.argmax(q_values)

    def update(self, state, action, reward, next_state, done):
        features = self.featurizer.transform(state)
        prediction = self.get_q_value(features, action)
        
        if done:
            target = reward
        else:
            next_features = self.featurizer.transform(next_state)
            next_q_values = [self.get_q_value(next_features, a) for a in range(self.n_actions)]
            target = reward + GAMMA * np.max(next_q_values)
            
        error = target - prediction
        
        # 简单梯度下降
        # 进阶技巧：Alpha 应该根据 ||c|| 进行缩放 (Alpha / ||c||_2)，此处简化处理
        self.weights[action] += LEARNING_RATE * error * features

if __name__ == "__main__":
    # 使用 500 步限制
    env = gym.make("CartPole-v1", max_episode_steps=500)
    writer = SummaryWriter(comment="-linear-fourier")
    
    agent = LinearQLearner(env)
    
    epsilon = EPSILON_START
    episode_rewards = []
    running_reward = 0.0
    iter_no = 0 
    
    print(f"开始训练 Fourier Basis (Order={FOURIER_ORDER}) Q-learning...")
    print(f"特征维度: {agent.feature_dim}")
    
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
