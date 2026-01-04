# 使用原始状态特征 (Raw Features) 的线性 Q-learning
# 用于证明：为什么我们需要 RBF 这样的特征映射？
# 这里的 Q(s, a) = w_a * s + b
# 这种简单的线性模型很难拟合 CartPole 复杂的价值函数 (Value Function)
# 因为 Q 值通常不是状态的线性函数 (例如：角度 0 是最好的，正负角度都变差，这是一个非线性的山峰形状，线性函数只能画斜面)

import gymnasium as gym
import numpy as np
from tensorboardX import SummaryWriter
import sklearn.preprocessing

# 超参数
LEARNING_RATE = 0.1
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

BATCH_SIZE = 16
MAX_EPISODES = 3000

class LinearRawAgent:
    def __init__(self, env):
        self.env = env
        self.n_actions = env.action_space.n
        
        # 4个状态维度 + 1个Bias截距项
        self.feature_dim = 5 
        self.weights = np.zeros((self.n_actions, self.feature_dim))
        
        # 仍然需要标准化，否则梯度下降很难收敛
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

    def get_features(self, state):
        # 归一化
        scaled = self.scaler.transform([state])[0]
        # 添加 Bias 项 (x -> [x, 1])
        return np.append(scaled, 1.0)

    def get_q_value(self, features, action):
        return np.dot(self.weights[action], features)

    def get_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            features = self.get_features(state)
            q_values = [self.get_q_value(features, a) for a in range(self.n_actions)]
            return np.argmax(q_values)

    def update(self, state, action, reward, next_state, done):
        features = self.get_features(state)
        prediction = self.get_q_value(features, action)
        
        if done:
            target = reward
        else:
            next_features = self.get_features(next_state)
            next_q_values = [self.get_q_value(next_features, a) for a in range(self.n_actions)]
            target = reward + GAMMA * np.max(next_q_values)
            
        error = target - prediction
        self.weights[action] += LEARNING_RATE * error * features

if __name__ == "__main__":
    env = gym.make("CartPole-v1", max_episode_steps=500)
    writer = SummaryWriter(comment="-linear-raw")
    
    agent = LinearRawAgent(env)
    
    epsilon = EPSILON_START
    episode_rewards = []
    running_reward = 0.0
    iter_no = 0 
    
    print("开始训练 Linear Raw Q-learning Agent...")
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
            
            if iter_no == 0:
                running_reward = reward_mean
            else:
                running_reward = 0.99 * running_reward + 0.01 * reward_mean
            
            print("%d: reward_mean=%.1f, running_reward=%.1f, epsilon=%.2f" % (
                iter_no, reward_mean, running_reward, epsilon), flush=True)
            
            writer.add_scalar("Reward/Mean", reward_mean, iter_no)
            writer.add_scalar("Reward/Running", running_reward, iter_no)
            
            iter_no += 1
            
        if running_reward > 475:
             print(f"Solved in {episode} episodes!")
             break

    writer.close()
