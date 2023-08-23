if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import matplotlib.pyplot as plt
import numpy as np

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

# Define the Agent class
class Agent:
    def __init__(self, input_size, hidden_size, action_size):
        self.gamma = 0.98
        self.lr = 0.0002
        self.memory = []
        self.policy = Policy(input_size, hidden_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

    def get_action(self, state):
        if isinstance(state, tuple):
            state = state[0]
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float32)
        probs = self.policy(state)
        action = torch.multinomial(probs, 1).item()
        return action, probs[0][action]

    def add(self, reward, prob):
        self.memory.append((reward, prob))

    def update(self):
        R = 0
        policy_loss = []
        returns = []

        for r, _ in self.memory[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)

        for (r, prob), R in zip(self.memory, returns):
            policy_loss.append(-torch.log(prob).unsqueeze(0) * R)

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        self.memory = []

# Initialize the environment and the agent
env = gym.make('CartPole-v0', render_mode = 'human')
agent = Agent(4, 128, 2)
episodes = 3000
reward_history = []

for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, prob = agent.get_action(state)
        next_state, reward, done, info, info1 = env.step(action)

        agent.add(reward, prob)
        state = next_state
        total_reward += reward

    agent.update()
    reward_history.append(total_reward)

    if episode % 50 == 0:
        print(f"Episode: {episode}, Total Reward: {total_reward}")

# plot_reward_history(reward_history)
from common.utils import plot_total_reward
plot_total_reward(reward_history)