import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # for 
import matplotlib.pyplot as plt
from common.gridworld import GridWorld
import torch
import torch.nn as nn
import torch.optim as optim

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import numpy as np

# gpuが使用される場合の設定

# print("CUDA available:" , torch.cuda.is_available())
# print("Number of GPUs:", torch.cuda.device_count())
# if torch.cuda.is_available():
#     print("Current GPU:", torch.cuda.current_device())
#     print("GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))

def one_hot(state):
    HEIGHT, WIDTH = 3, 4
    vec = np.zeros(HEIGHT * WIDTH, dtype=np.float32)
    y, x = state
    idx = WIDTH * y + x
    vec[idx] = 1.0
    return vec[np.newaxis, :]

class QNet(nn.Module):
    def __init__(self, obs_size, n_actions, hidden_size=128):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_actions)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class QLearningAgent:
    def __init__(self, obs_size = 12, action_size = 4, lr=0.01, epsilon = 0.1, gamma=0.99):
        self.obs_size = obs_size
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.action_size = action_size
        
        self.net = QNet(obs_size, action_size)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def get_action(self, state_vec):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = self.net(torch.tensor(state_vec, dtype=torch.float32))
            return torch.argmax(qs).item()

    def update(self, state, action, reward, next_state, done):
        # Convert inputs to tensors
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor([reward], dtype=torch.float32)
        
        if done:
            next_q = torch.zeros(1)
        else:
            next_qs = self.net(next_state)
            next_q = torch.max(next_qs)
            next_q = next_q.detach()

        
        target = self.gamma * next_q + reward
        qs = self.net(state)
        q = qs[:, action]
        
        # Compute loss using MSE
        loss = self.criterion(target, q)
        
        # Backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

env = GridWorld()
agent = QLearningAgent(12, 4)

episodes = 1000
loss_history = []

for episode in range(episodes):
    state = env.reset()
    state = one_hot(state)
    total_loss, cnt = 0, 0
    done = False

    while not done:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        next_state = one_hot(next_state)

        loss = agent.update(state, action, reward, next_state, done)
        total_loss += loss
        cnt += 1
        state = next_state
    

    average_loss = total_loss / cnt
    loss_history.append(average_loss)


plt.xlabel('episode')
plt.ylabel('loss')
plt.plot(range(len(loss_history)), loss_history)
plt.show()

# visualize
Q = {}
for state in env.states():
    for action in env.action_space:
        q = agent.qnet(one_hot(state))[:, action]
        Q[state, action] = float(q.data)
env.render_q(Q)