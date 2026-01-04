# 使用线性函数近似 (Linear Function Approximation) 的 Q-learning
# 这是一个处理连续状态空间的经典方法，不需要使用深度神经网络
# 核心思想：Q(s, a) 不再查表，而是近似为一个线性函数：Q(s, a) = theta_a * features(s)
# 特征提取 (Feature Extraction) 使用径向基函数 (RBF, Radial Basis Function)

import gymnasium as gym
import numpy as np
from tensorboardX import SummaryWriter
from sklearn.kernel_approximation import RBFSampler
import sklearn.pipeline
import sklearn.preprocessing

# ==============================================================================
# 1. 超参数设置
# ==============================================================================
LEARNING_RATE = 0.1     # 学习率
GAMMA = 0.99            # 折扣因子
EPSILON_START = 1.0     # 初始探索率
EPSILON_END = 0.01      # 最终探索率
EPSILON_DECAY = 0.995   # 衰减系数

# RBF 特征设置
# 为什么需要 RBF？
# 原始状态是 [x, v, theta, omega]，简单的线性组合 w*x + b 无法拟合复杂的价值函数（例如“倒V字型”的价值分布）。
# RBF 将低维状态映射到高维空间，相当于在状态空间中撒了一把“锚点” (Basis Functions)。
# 离某个锚点越近，该特征激活值越高。这样线性模型就可以通过组合这些锚点来拟合非线性曲线。
# 我们使用 4 个不同 gamma (高斯核宽度) 的 RBF 采样器堆叠，以捕捉不同尺度的特征 (有的关注局部细节，有的关注全局趋势)。
N_COMPONENTS = 100      # 每个 RBF 采样器的特征数量

# 对比实验设置
BATCH_SIZE = 16
MAX_EPISODES = 1024

class RBFFeaturizer:
    """
    RBF 特征提取器
    将低维的连续状态 (4维) 映射到高维特征空间 (400维)
    """
    def __init__(self, env):
        # 1. 采样一些状态样本来初始化标准化器 (Scaler)
        # 这一步是为了让输入数据的均值为0，方差为1，有利于 RBF 提取特征
        # 如果数据分布差异很大，RBF 的效果会大打折扣
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

        # 2. 初始化 RBF 采样器 (使用 sklearn)
        # FeatureUnion 将多个转换器的输出拼接在一起
        # Gamma 越大，高斯核越窄，关注越局部的特征
        self.featurizer = sklearn.pipeline.FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=N_COMPONENTS)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=N_COMPONENTS)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=N_COMPONENTS)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=N_COMPONENTS))
        ])
        self.featurizer.fit(self.scaler.transform(observation_examples))

    def transform(self, state):
        # 输入: (4,) -> 输出: (400,)
        # 先标准化，再进行 RBF 映射
        scaled = self.scaler.transform([state])
        return self.featurizer.transform(scaled)[0]

class LinearQLearner:
    """
    线性 Q-learning 智能体
    Q(s, a) = weights[a] dot features(s)
    """
    def __init__(self, env):
        self.env = env
        self.n_actions = env.action_space.n
        
        # 初始化特征提取器
        self.featurizer = RBFFeaturizer(env)
        
        # 初始化权重: [动作数, 特征数]
        # 特征数 = 4 * N_COMPONENTS = 400
        # 权重矩阵 W 的每一行对应一个动作的权重向量 w_a
        self.feature_dim = 4 * N_COMPONENTS
        self.weights = np.zeros((self.n_actions, self.feature_dim))

    def get_q_value(self, state_features, action):
        # 线性模型预测 Q 值：特征向量与权重向量的点积
        # Q(s, a) = w_a * phi(s)
        return np.dot(self.weights[action], state_features)

    def get_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            features = self.featurizer.transform(state)
            q_values = [self.get_q_value(features, a) for a in range(self.n_actions)]
            return np.argmax(q_values)

    def update(self, state, action, reward, next_state, done):
        """
        线性近似的权重更新规则 (Gradient Descent):
        损失函数 Loss = 1/2 * (Target - Prediction)^2
        梯度 dLoss/dw = -(Target - Prediction) * features(s)
        更新规则 w <- w - alpha * dLoss/dw
                  <- w + alpha * (Target - Prediction) * features(s)
        """
        features = self.featurizer.transform(state)
        
        # 计算 Prediction (当前估计值)
        prediction = self.get_q_value(features, action)
        
        # 计算 Target (目标值)
        if done:
            target = reward
        else:
            next_features = self.featurizer.transform(next_state)
            next_q_values = [self.get_q_value(next_features, a) for a in range(self.n_actions)]
            target = reward + GAMMA * np.max(next_q_values)
        
        # 计算误差 (TD Error)
        error = target - prediction
        
        # 更新权重
        # w_a = w_a + alpha * error * features
        self.weights[action] += LEARNING_RATE * error * features

if __name__ == "__main__":
    # 显式指定 max_episode_steps=500
    env = gym.make("CartPole-v1", max_episode_steps=500)
    writer = SummaryWriter(comment="-linear-rbf")
    
    agent = LinearQLearner(env)
    
    epsilon = EPSILON_START
    episode_rewards = []
    running_reward = 0.0
    iter_no = 0 
    
    print("开始训练 Linear RBF Q-learning Agent...")
    print("特征维度:", agent.feature_dim)
    
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
            reward_min = np.min(batch_rewards)
            
            running_reward = reward_mean if iter_no == 0 else 0.99 * running_reward + 0.01 * reward_mean
            
            print(f"{iter_no}: reward_mean={reward_mean:.1f}, reward_max={reward_max:.1f}, running_reward={running_reward:.1f}, epsilon={epsilon:.2f}", flush=True)
            
            writer.add_scalar("Reward/Mean", reward_mean, iter_no)
            writer.add_scalar("Reward/Max", reward_max, iter_no)
            writer.add_scalar("Reward/Min", reward_min, iter_no)
            writer.add_scalar("Reward/Running", running_reward, iter_no)
            writer.add_scalar("Training/Epsilon", epsilon, iter_no)
            
            iter_no += 1
            
        if running_reward > 475:
            print(f"Solved in {episode} episodes! Final Running Reward: {running_reward}")
            break

    writer.close()
