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
    batch = [] #used to accumulate batch (list of episode instances)
    episode_reward = 0.0 #total reward coutner
    episode_steps = [] #list of steps
    obs = env.reset() #reset to obtain first observation 
    if isinstance(obs, tuple):
        obs = obs[0]
    sm = nn.Softmax(dim = 1) #create a softmax layer, used to convert netwrok's output to probablity distribution

    while True:
        # Convert observation to tensor efficiently: (1, obs_size)
        obs_v = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        act_probs_v = sm(net(obs_v)) #feed to softmax function for probability distribution
        act_probs = act_probs_v.data.numpy()[0] #returns tensors which track gradients, unpack them into a NumPy array

        #we have probability of actions
        #use this distribution to obtain actual action at random
        #obtain next observation, reward, indcitaion episode is ending, and extra info
        action = np.random.choice(len(act_probs), p=act_probs)
        
        step_result = env.step(action)
        if len(step_result) == 5:
            next_obs, reward, terminated, truncated, extra_info = step_result
            is_done = terminated or truncated
        else:
            next_obs, reward, is_done, extra_info = step_result
            
        #env.render() for video


        #Step number 2, calculate total award for every espidoe
        #Accumulate total award
        #List of episode steps extended with an observation, action pair
        #Careful! we save observation that was used to choose action, not next_obs
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action))

        #Reset everything after appending finalized episdoe to batch with total award and steps
        if is_done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if isinstance(next_obs, tuple):
                next_obs = next_obs[0]
            #Reached desired count of episodes, return to caller using yield
            if(len(batch) == batch_size):
                yield batch
                batch = [] #reset for next batch
        #assign an observation obtained from environment to curr observation
        obs = next_obs

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
    train_obs = []
    train_act = []
    #For every episode in the batch, if lower than our reward bound we throw it out
    for example in batch:
        if example.reward < reward_bound:
            continue
        #adds ALL elements of an iterable to end of list
        train_obs.extend(map(lambda step: step.observation, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))
    #Convert to tensors for training
    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
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
    iter_list = []
    reward_mean_list = []
    reward_bound_list = []
    
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

        if iter_no == 0:
            running_reward = reward_m
        else:
            running_reward = 0.99 * running_reward + 0.01 * reward_m

        #Keeping track of progress
        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f, running_reward=%.1f" % (
            iter_no, loss_v.item(), reward_m, reward_b, running_reward))
        writer.add_scalar("Training/Loss", loss_v.item(), iter_no)
        writer.add_scalar("Reward/Bound", reward_b, iter_no)
        writer.add_scalar("Reward/Mean", reward_m, iter_no)
        writer.add_scalar("Reward/Running", running_reward, iter_no)
        writer.add_scalar("Reward/Max", reward_max, iter_no)
        writer.add_scalar("Reward/Min", reward_min, iter_no)
        
        # Store for plotting
        iter_list.append(iter_no)
        reward_mean_list.append(reward_m)
        reward_bound_list.append(reward_b)

        if reward_m > 199: #Greater than 199 because in gym it's solved when mean reward greater than 199
            print("Solved!")
        
        if iter_no > 100:
            print("Reached 100 iterations, stopping.")
            break
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
