if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import matplotlib.pyplot as plt
import numpy as np
from torchviz import make_dot
import math
import torch.distributions as dist

# Define the Policy network using PyTorch
class Policy(nn.Module):
    def __init__(self, input_size, hidden_size, action_size):
        super(Policy, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, action_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = self.softmax(self.l2(x))
        return x

# def normal_pdf(x, mu, sigma): # get_action2で用いる
#     """Calculate the probability density function of the normal distribution."""
#     return (1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Define the Agent class
class Agent:
    def __init__(self, input_size, hidden_size, action_size):
        self.gamma = 0.98
        self.lr = 0.001
        self.memory = []
        self.policy = Policy(input_size, hidden_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

    # def get_action1(self, state):
    #     if isinstance(state, tuple):
    #         state = state[0]
    #     state = torch.tensor(state[np.newaxis, :], dtype=torch.float32)
    #     probs = self.policy(state)
    #     action = torch.clamp(torch.normal(mean=0, std=1.0, size=(1,)), min=-2.0, max=2.0).item()
    #     return action, probs
    

    # def get_action2(self, state):
    #     if isinstance(state, tuple):
    #         state = state[0]
    #     state = torch.tensor(state[np.newaxis, :], dtype=torch.float32)
    #     action = self.policy(state)
    #     sampling_action = torch.clamp(torch.normal(mean=action.item(), std=1.0, size=(1,)), min=-2.0, max=2.0).item()
        
    #     # Calculate the probability density
    #     if -2.0 <= sampling_action <= 2.0:
    #         probability_density = normal_pdf(sampling_action, action, 1.0)
    #     else:
    #         probability_density = 0.0
    #     # probability_density = torch.tensor(probability_density[np.newaxis, :], dtype=torch.float32)
    #     probability_density = torch.tensor([[probability_density]], dtype=torch.float32)
        
    #     return sampling_action, probability_density
    
    def get_action(self, state):
        if isinstance(state, tuple):
            state = state[0]
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float32)
        action_mean = self.policy(state).squeeze()
        
        # # Using PyTorch's distributions to sample from the normal distribution and compute log probability
        # normal_distribution = dist.Normal(action_mean, 1.0)
        # sampling_action = torch.clamp(normal_distribution.sample(), min=-2.0, max=2.0).item()
        # log_probability = normal_distribution.log_prob(torch.tensor(sampling_action)).unsqueeze(0).unsqueeze(0)
        # # print(log_probability.requires_grad)

        # Using PyTorch's distributions to sample from the normal distribution and compute log probability
        normal_distribution = dist.Normal(action_mean.squeeze(), 1.0)  # squeeze to match the shape
        sampling_action_tensor = torch.clamp(normal_distribution.sample(), min=-2.0, max=2.0)
        sampling_action = sampling_action_tensor.squeeze()
        print(action_mean, sampling_action)
        log_probability = normal_distribution.log_prob(action_mean).unsqueeze(0).unsqueeze(0)
        return sampling_action, log_probability

        
    def add(self, reward, prob):
        self.memory.append((reward, prob))

    def update(self):
        G = 0
        policy_loss = 0

        # Calculate cumulative rewards
        for r, prob in self.memory[::-1]:
            G = r + self.gamma * G
            # Calculate the loss
            # _debug = torch.log(prob)
            # policy_loss += -torch.log(prob).unsqueeze(0) * G
            policy_loss += - prob * G

        # dot = make_dot(policy_loss)
        # dot.view()
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        self.memory = []


# Initialize the environment and the agent
env = gym.make('Pendulum-v1', render_mode = 'human')
agent = Agent(3, 128, 1)
done = False
reward_history = []

# # while True:

# state = env.reset()
# done = False
# total_reward = 0
# step_counter = 0
# accumulated_reward = 0

# while not done:
#     action, prob = agent.get_action(state)
#     step_results = env.step([action])
#     # print(step_results)
#     next_state, reward, done, _, info = step_results
#     # print(reward, done)
#     agent.add(reward, prob)
#     state = next_state
#     total_reward += reward

#     step_counter += 1
#     accumulated_reward += reward
    
#     if step_counter % 100 == 0:
#         print(f"Steps: {step_counter}, Accumulated Reward: {accumulated_reward}")
#         accumulated_reward = 0  # reset the accumulated reward

#     agent.update()
#     reward_history.append(total_reward)

while not done:

    state = env.reset()
    done = False
    total_reward = 0
    step_counter = 0
    accumulated_reward = 0

    for i in range(100):
        action, prob = agent.get_action(state)
        step_results = env.step([action])
        # print(step_results)
        next_state, reward, done, _, info = step_results
        if done == True:
            break

        # print(reward, done)
        agent.add(reward, prob)
        state = next_state
        total_reward += reward

        # step_counter += 1
        accumulated_reward += reward
        
    print(f"Steps: {step_counter}, Accumulated Reward: {accumulated_reward}")
    accumulated_reward = 0  # reset the accumulated reward

    agent.update()
    reward_history.append(total_reward)

# plot_reward_history(reward_history)
from common.utils import plot_total_reward
plot_total_reward(reward_history)