# https://blog.csdn.net/hhy_csdn/article/details/107984458
# https://github.com/andrewliao11/cem
# use simply vector to represent policy
import gymnasium as gym
import numpy as np
from tensorboardX import SummaryWriter

env = gym.make('CartPole-v1')
# env = env.unwrapped
# env.render()

#vector of means(mu) and standard dev(sigma) for each paramater
mu = np.random.uniform(size = env.observation_space.shape)
sigma = np.random.uniform(low = 0.001,size = env.observation_space.shape)
print(mu.shape)
print(sigma.shape)

writer = SummaryWriter(comment="-cem-vector")

def noisy_evaluation(env,W,render = False,):
    """
    uses parameter vector W to choose policy for 1 episode,
    returns reward from that episode
    """
    reward_sum = 0
    state, _ = env.reset()
    t = 0
    while True:
        t += 1
        action = int(np.dot(W,state)>0) # 核心决策逻辑：线性策略 (Linear Policy)
        # 相当于在 4维状态空间中切了一刀（超平面），左边推一把，右边推一把。
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        reward_sum += reward
        if render and t%5 == 0: 
            env.render()
        if done or t > 2000: # 
            break
    return reward_sum
    
def init_params(mu,sigma,n):
    """
    种群生成 (Population Generation):
    利用 NumPy 广播机制，上帝掷骰子，一次性从 N(mu, sigma) 中采样 n 个不同的“平行宇宙”权重。
    """
    return np.random.normal(loc=mu, scale=sigma + 1e-7, size=(n, mu.shape[0]))

def get_constant_noise(step):
    return np.clip(5-step/10., a_max=1,a_min=0.5)

running_reward = 0
n = 40
p = 8
n_iter = 100
render = False 

state, info = env.reset()
i = 0
while i < n_iter:
    #initialize an array of parameter vectors
    # 第一步：采样 (Sample) - 从分布里猜一群参数
    wvector_array = init_params(mu,sigma,n)
    
    # 第二步：评估 (Evaluate) - 去环境里跑分
    reward_sums = np.array([noisy_evaluation(env, w, render) for w in wvector_array])

    #sort params/vectors based on total reward of an episode using that policy
    # 第三步：筛选精英 (Select Elites) - 优胜劣汰，只关心那 20% 表现最好的策略
    rankings = np.argsort(reward_sums)
    #pick p vectors with highest reward
    top_vectors = wvector_array[rankings,:]
    top_vectors = top_vectors[-p:,:]
    # print(f"top vectors shape: {top_vectors.shape}")
    #fit new gaussian from which to sample policy
    # 第四步：更新分布 (Update Distribution) - 进化
    for q in range(top_vectors.shape[1]):
        mu[q] = top_vectors[:,q].mean() # 均值移动：下一次采样的中心，移动到精英的中心（向高分区域靠近）
        sigma[q] = top_vectors[:,q].std() + get_constant_noise(i) # 方差收缩：精英们聚在一起，std变小；注入噪声防止过早收敛

    running_reward = 0.99*running_reward + 0.01*reward_sums.mean()
    
    # Calculate Reward Bound (min reward of elite samples)
    # rankings are indices sorted by reward, so rankings[-p] is the index of the p-th best (threshold)
    reward_bound = reward_sums[rankings[-p]]

    # TensorBoard logging
    writer.add_scalar("Reward/Mean", reward_sums.mean(), i)
    writer.add_scalar("Reward/Running", running_reward, i)
    writer.add_scalar("Reward/Bound", reward_bound, i)
    writer.add_scalar("Reward/Max", reward_sums.max(), i)
    writer.add_scalar("Reward/Min", reward_sums.min(), i)
    writer.add_scalar("Training/SigmaMean", sigma.mean(), i)
    
    print(f"# {i}, mean: {reward_sums.mean():.3f}, running mean: {running_reward:.3f}, range: {reward_sums.min():.3f} to {reward_sums.max():.3f}")
    i += 1
writer.close()
