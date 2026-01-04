# 使用神经网络 (NN) 来近似策略函数 pi(a|s)
import gymnasium as gym
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# 超参数设置
HIDDEN_SIZE = 128   # 隐藏层神经元数量
BATCH_SIZE = 16     # 每次迭代采集的 Episode 数量 (N)
PERCENTILE = 70     # 精英筛选的百分位阈值 (前 30% 为精英)

class Net(nn.Module):
    """
    策略网络 (Policy Network)
    输入: 状态向量 (Observation)
    输出: 每个动作的未归一化概率得分 (Logits)
    """
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size), # 输入层 -> 隐藏层
            nn.ReLU(),                        # 激活函数
            nn.Linear(hidden_size, n_actions) # 隐藏层 -> 输出层 (动作维度)
        )

    def forward(self, x):
        return self.net(x)

# 定义数据结构
# Episode: 存储单局游戏的完整信息 (总奖励, 每一步的详情)
Episode = namedtuple('Episode', field_names=['reward', 'steps'])

# EpisodeStep: 存储单步交互的信息 (观测状态, 采取的动作)
# 这些"精英步骤"将作为后续训练神经网络的"标签数据"
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

#Accepts environment from gym library
#Neural netowrk
#count of episodes it should generate each iteration
#Step 1, N number of episodes
def iterate_batches(env, net, batch_size):
    """
    无限生成器，用于不断产生训练所需的 Batch 数据。
    逻辑结构：无限循环 -> 生成一个 Batch -> 生成 Batch 中的每一个 Episode -> 执行 Episode 中的每一步
    """
    sm = nn.Softmax(dim=1) # Softmax 层，用于将神经网络输出的 Logits 转化为概率分布
    
    while True:
        batch = [] # 初始化空列表，用于存放当前 Batch 的所有 Episode 数据
        
        # 循环生成 batch_size 个完整的 Episode
        for _ in range(batch_size):
            episode_reward = 0.0 # 记录当前 Episode 的总奖励
            episode_steps = []   # 记录当前 Episode 的每一步 (Observation, Action)
            obs, _ = env.reset() # 重置环境，获取初始状态
            
            # 执行单个 Episode 的循环，直到游戏结束
            while True:
                # 1. 数据准备：将 numpy 数组的 Observation 转换为 PyTorch Tensor
                # unsqueeze(0) 是为了增加一个 Batch 维度，变成 (1, obs_size)，因为神经网络输入要求有 Batch 维度
                obs_v = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                
                # 2. 网络推理：输入状态，通过网络得到动作的 Logits，再经过 Softmax 得到概率分布
                act_probs_v = sm(net(obs_v))
                
                # 3. 获取概率：将 Tensor 转回 Numpy 数组，取 [0] 是因为我们只有一个样本
                act_probs = act_probs_v.data.numpy()[0]
                
                # 4. 动作采样：根据概率分布随机选择动作 (Exploration)
                # 这是 CEM 的核心：即使网络认为某个动作概率大，我们也有机会尝试其他动作
                action = np.random.choice(len(act_probs), p=act_probs)
                
                # 5. 环境交互：执行动作，获取下一个状态、奖励、是否结束等信息
                next_obs, reward, terminated, truncated, _ = env.step(action)
                
                # 6. 数据记录：累加奖励，保存当前步的 (Obs, Action) 用于后续筛选精英样本训练
                episode_reward += reward
                episode_steps.append(EpisodeStep(observation=obs, action=action))
                
                # 7. 结束判断：如果游戏结束 (Terminated) 或超时 (Truncated)
                if terminated or truncated:
                    # 将完整的 Episode (总奖励 + 所有步骤) 加入 Batch
                    batch.append(Episode(reward=episode_reward, steps=episode_steps))
                    break # 跳出当前 Episode 的循环
                
                # 状态更新：准备进行下一步
                obs = next_obs
        
        # 当攒够了 batch_size 个 Episode 后，通过 yield 返回给训练主循环
        # yield 会暂停函数执行，下次调用时从这里继续
        yield batch

def filter_batch(batch, percentile):
    """
    核心步骤：精英筛选 (Selection)
    根据奖励对 Batch 中的 Episode 进行排序，保留表现最好的部分 (Elite Episodes)。
    """
    # 提取 Batch 中所有 Episode 的总奖励
    rewards = list(map(lambda s: s.reward, batch))
    
    # 1. 计算奖励阈值 (Reward Bound)
    # 例如 percentile=70，意味着我们只保留分数高于 70% 样本的那 30% 精英
    reward_bound = np.percentile(rewards, percentile) 
    
    # 统计指标 (用于监控)
    reward_mean = float(np.mean(rewards))
    reward_max = float(np.max(rewards))
    reward_min = float(np.min(rewards))

    # 2. 筛选精英样本
    # 只有总奖励 >= 阈值的 Episode 才会被保留用来训练
    elite_batch = [e for e in batch if e.reward >= reward_bound]
    
    # 3. 数据展平 (Flatten)
    # 将所有精英 Episode 的每一步 (State, Action) 提取出来，合并成一个大的训练集
    # State -> 输入 (X), Action -> 标签 (Y)
    train_obs = [step.observation for e in elite_batch for step in e.steps]
    train_act = [step.action for e in elite_batch for step in e.steps]
    
    # 4. 转换为 Tensor
    # 这里的 train_obs_v 就是输入 X，train_act_v 就是目标 Y
    # 使用 CrossEntropyLoss 训练网络去模仿这些精英动作
    train_obs_v = torch.as_tensor(np.array(train_obs), dtype=torch.float32)
    train_act_v = torch.as_tensor(np.array(train_act), dtype=torch.long)
    
    return train_obs_v, train_act_v, reward_bound, reward_mean, reward_max, reward_min

if __name__ == "__main__":
    env = gym.make("CartPole-v1", max_episode_steps=500)
    
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # 初始化网络、损失函数和优化器
    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss() # 交叉熵损失：用于监督学习，让网络模仿精英动作
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    writer = SummaryWriter(comment="-cem-nn")
    
    running_reward = 0.0

    # 训练主循环
    # iterate_batches 是一个生成器，每次 yield 一个完整的 Batch
    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        # 1. 筛选精英样本
        # obs_v: 精英样本的状态 (输入 X)
        # acts_v: 精英样本采取的动作 (标签 Y)
        obs_v, acts_v, reward_b, reward_m, reward_max, reward_min = filter_batch(batch, PERCENTILE)
        
        # 2. 监督学习 (Supervised Learning)
        # 将强化学习问题转化为监督学习问题：最大化精英动作的似然概率
        optimizer.zero_grad()
        action_scores_v = net(obs_v)             # 网络预测
        loss_v = objective(action_scores_v, acts_v) # 计算预测与精英动作的偏差
        loss_v.backward()                        # 反向传播
        optimizer.step()                         # 更新参数

        # 3. 监控指标：策略熵 (Policy Entropy)
        # 熵越大 -> 探索性越强 (分布越平坦)
        # 熵越小 -> 确定性越强 (分布越尖锐)
        probs_v = torch.softmax(action_scores_v, dim=1)
        entropy_v = -(probs_v * probs_v.log()).sum(dim=1).mean()

        # 计算平滑后的平均奖励 (便于观察趋势)
        running_reward = reward_m if iter_no == 0 \
            else 0.99 * running_reward + 0.01 * reward_m 

        # 4. 停止条件
        if iter_no >= 64:
            print(f"Reached 64 iterations, stopping. Final mean reward: {reward_m}")
            break

        # 5. 日志记录
        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f, running_reward=%.1f" % (
            iter_no, loss_v.item(), reward_m, reward_b, running_reward))
        writer.add_scalar("Training/Loss", loss_v.item(), iter_no)
        writer.add_scalar("Training/PolicyEntropy", entropy_v.item(), iter_no)
        writer.add_scalar("Reward/Bound", reward_b, iter_no)
        writer.add_scalar("Reward/Mean", reward_m, iter_no)
        writer.add_scalar("Reward/Running", running_reward, iter_no)
        writer.add_scalar("Reward/Max", reward_max, iter_no)
        writer.add_scalar("Reward/Min", reward_min, iter_no) 
        # Store for plotting
        # iter_list.append(iter_no)
        # reward_mean_list.append(reward_m)
        # reward_bound_list.append(reward_b)
    
    writer.close()
    #env.close()


