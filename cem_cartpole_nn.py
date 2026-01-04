# use nn to represent policy
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

HIDDEN_SIZE = 128 #Count of neurons in hidden network
BATCH_SIZE = 16 #Count of episodes we play on every iteration
PERCENTILE = 70 #percntile of episodes' total awards

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        #Takes a single observation from the env as an input vector and outputs a number for every action we can perform
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)

#Single episode stored as TOTAL undiscoutned reward and collection of episode steps
Episode = namedtuple('Episode', field_names=['reward', 'steps'])

#Represents one single stpe agent made in the episode, stores observation from env and action completed
#by agent
#Use episode steps from elite episodes as training data
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

#Step number 3! Throw away all episodes below reward boundary
def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch)) #returns episode reward from batch list
    #uses NumPy percentile function
    #From list of rewards and desired percentile, calculate the percentile's value
    reward_bound = np.percentile(rewards, percentile) 
    reward_mean = float(np.mean(rewards)) #used for monitoring
    reward_max = float(np.max(rewards))
    reward_min = float(np.min(rewards))

    #Only train elite episodes!
    elite_batch = [e for e in batch if e.reward >= reward_bound]
    
    # Flatten steps from all elite episodes
    train_obs = [step.observation for e in elite_batch for step in e.steps]
    train_act = [step.action for e in elite_batch for step in e.steps]
    
    #Convert to tensors efficiently (via numpy array to avoid warning/copy)
    train_obs_v = torch.as_tensor(np.array(train_obs), dtype=torch.float32)
    train_act_v = torch.as_tensor(np.array(train_act), dtype=torch.long)
    #Last two values only uesd to check on tensorboard how well our agent is doing
    return train_obs_v, train_act_v, reward_bound, reward_mean, reward_max, reward_min

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss() #loss function
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    writer = SummaryWriter(comment="-cem-nn")
    
    # Lists to store metrics for plotting
    # iter_list = []
    # reward_mean_list = []
    # reward_bound_list = []
    
    running_reward = 0.0

    #training loop
    #enumerate to get counter and value from iterable at the same time
    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m, reward_max, reward_min = filter_batch(batch, PERCENTILE)
        #Classic training algorithm
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()

        # Calculate Entropy for logging (Proxy for exploration/Sigma)
        probs_v = torch.softmax(action_scores_v, dim=1)
        entropy_v = -(probs_v * probs_v.log()).sum(dim=1).mean()

        running_reward = reward_m if iter_no == 0 \
            else 0.99 * running_reward + 0.01 * reward_m 

        if iter_no > 100:
            print(f"Reached 100 iterations, stopping. Final mean reward: {reward_m}")
            break

        #Keeping track of progress
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

    # # Plotting results
    # plt.figure(figsize=(10, 5))
    # plt.plot(iter_list, reward_mean_list, label='Mean Reward')
    # plt.plot(iter_list, reward_bound_list, label='Reward Bound (Elite Threshold)')
    # plt.xlabel('Iteration')
    # plt.ylabel('Reward')
    # plt.title('CEM with NN Training Progress on CartPole-v1')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig('cartpole_rewards.png')
    # print("Plot saved to cartpole_rewards.png")
