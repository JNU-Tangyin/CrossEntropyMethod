# 使用表格型 SARSA (Tabular SARSA) 解决 CartPole 问题
# SARSA (State-Action-Reward-State-Action) 是 On-Policy 的强化学习算法
# 与 Q-learning (Off-Policy) 的最大区别在于：SARSA 更新 Q 值时，使用的是实际上采取的下一个动作 A'，而不是最大的那个

import gymnasium as gym
import numpy as np
import math
from tensorboardX import SummaryWriter

# ==============================================================================
# 1. 超参数设置 (Hyperparameters)
# ==============================================================================
LEARNING_RATE = 0.1     # 学习率
GAMMA = 0.99            # 折扣因子
EPSILON_START = 1.0     # 初始探索率
EPSILON_END = 0.01      # 最终探索率
EPSILON_DECAY = 0.995   # 衰减系数

# ==============================================================================
# 2. 离散化设置 (Discretization)
# ==============================================================================
# 同样的离散化策略，保证与 Q-learning 可比性
BINS = [
    np.linspace(-2.4, 2.4, 20),
    np.linspace(-4.0, 4.0, 20),
    np.linspace(-0.209, 0.209, 20),
    np.linspace(-4.0, 4.0, 20)
]

# ==============================================================================
# 3. 对比实验设置
# ==============================================================================
BATCH_SIZE = 16
MAX_EPISODES = 3000

class SARSALearner:
    """
    SARSA 智能体
    """
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.q_table = np.random.uniform(low=-1, high=1, size=([len(b) + 1 for b in BINS] + [n_actions]))

    def get_discrete_state(self, state):
        discrete_state = []
        for i in range(len(state)):
            discrete_state.append(np.digitize(state[i], BINS[i]))
        return tuple(discrete_state)

    def get_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            discrete_state = self.get_discrete_state(state)
            return np.argmax(self.q_table[discrete_state])

    def update(self, state, action, reward, next_state, next_action, done):
        """
        SARSA 更新公式:
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
        注意这里直接使用 Q(s', a')，而不是 max Q(s', all_a')
        这意味着我们在优化"当前策略"的表现，而不是"最优策略" (On-Policy)
        """
        discrete_state = self.get_discrete_state(state)
        discrete_next_state = self.get_discrete_state(next_state)
        
        current_q = self.q_table[discrete_state + (action,)]
        
        if done:
            target_q = reward # 终止状态
        else:
            # 关键区别: 使用 next_action 的 Q 值
            target_q = reward + GAMMA * self.q_table[discrete_next_state + (next_action,)]
            
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * target_q
        self.q_table[discrete_state + (action,)] = new_q

if __name__ == "__main__":
    # 显式指定 max_episode_steps=500
    env = gym.make("CartPole-v1", max_episode_steps=500)
    writer = SummaryWriter(comment="-sarsa")
    
    agent = SARSALearner(env.action_space.n)
    
    epsilon = EPSILON_START
    episode_rewards = []
    running_reward = 0.0
    iter_no = 0 
    
    print("开始训练 SARSA Agent...")
    
    for episode in range(MAX_EPISODES):
        obs, _ = env.reset()
        
        # SARSA 需要先根据当前策略选择动作 A
        action = agent.get_action(obs, epsilon)
        
        total_reward = 0
        done = False
        
        while not done:
            # 1. 执行动作 A，观察 R, S'
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 2. 根据当前策略选择下一个动作 A' (On-Policy)
            # 注意: 如果 next_obs 是终止状态，next_action 其实不重要，但代码逻辑需要一个值
            next_action = agent.get_action(next_obs, epsilon)
            
            # 3. SARSA 更新: 使用 (S, A, R, S', A')
            agent.update(obs, action, reward, next_obs, next_action, done)
            
            # 4. 状态流转
            obs = next_obs
            action = next_action # 下一步的动作就是刚才选好的 A'
            
            total_reward += reward
            
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        episode_rewards.append(total_reward)
        
        if (episode + 1) % BATCH_SIZE == 0:
            batch_rewards = episode_rewards[-BATCH_SIZE:]
            reward_mean = np.mean(batch_rewards)
            reward_max = np.max(batch_rewards)
            reward_min = np.min(batch_rewards)
            
            if iter_no == 0:
                running_reward = reward_mean
            else:
                running_reward = 0.99 * running_reward + 0.01 * reward_mean
            
            print("%d: reward_mean=%.1f, reward_max=%.1f, running_reward=%.1f, epsilon=%.2f" % (
                iter_no, reward_mean, reward_max, running_reward, epsilon), flush=True)
            
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
    print("训练结束。")
