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
      action = int(np.dot(W,state)>0) # use parameters/state to choose action
      state, reward, terminated, truncated, info = env.step(action)
      done = terminated or truncated
      reward_sum += reward
      if render and t%3 == 0: env.render()
      if done or t > 2000: # 
            #print("finished episode, got reward:{}".format(reward_sum)) 
            break

    return reward_sum
    
def init_params(mu,sigma,n):
    """
    以mu和sigma的维度(=4)分量为均值和方差，采样n=40个点，组成n个4维向量
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
    wvector_array = init_params(mu,sigma,n)
    reward_sums = np.zeros((n))
    for k in range(n):
        #sample rewards based on policy parameters in row k of wvector_array
        reward_sums[k] = noisy_evaluation(env,wvector_array[k,:],render)

    #sort params/vectors based on total reward of an episode using that policy
    rankings = np.argsort(reward_sums)
    #pick p vectors with highest reward
    top_vectors = wvector_array[rankings,:]
    top_vectors = top_vectors[-p:,:]
    print("top vectors shpae:{}".format(top_vectors.shape))
    #fit new gaussian from which to sample policy
    for q in range(top_vectors.shape[1]):
        mu[q] = top_vectors[:,q].mean()
        sigma[q] = top_vectors[:,q].std()+get_constant_noise(i) # 在方差更新项加入扰动

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
    
    print("#{},mean:{}, running mean:{} range:{} to {},".format(i, reward_sums.mean(),running_reward,reward_sums.min(),reward_sums.max(),))
    i += 1
writer.close()
