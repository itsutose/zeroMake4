import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as TFunc

import copy
from collections import deque
import random
import matplotlib.pyplot as plt
import numpy as np
import gym
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = np.stack([x[0][0] if isinstance(x[0], tuple) else x[0] for x in data])
        # if isinstance(state, tuple):
        #     state = state[0]
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data]).astype(np.int32)
        return state, action, reward, next_state, done


class QNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(4,128)
        self.l2 = nn.Linear(128,128)
        self.l3 = nn.Linear(128,action_size)

    def forward(self, x):
        x = TFunc.relu(self.l1(x))
        x = TFunc.relu(self.l2(x))
        x = self.l3(x)
        return x


class DQNAgent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_size = 2

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size)
        self.qnet_target = QNet(self.action_size)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)
        pass

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            # state = state[np.newaxis, :] 
            if isinstance(state, tuple):
                state = state[0]
            state = state[np.newaxis, :]
            state = torch.tensor(state, dtype=torch.float32)
            qs = self.qnet(state)
            return qs.argmax(dim=1).item()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        state = torch.tensor(state, dtype=torch.float32)
        qs = self.qnet(state)
        q = qs[np.arange(self.batch_size), action]

        next_state = torch.tensor(next_state, dtype=torch.float32)
        next_qs = self.qnet_target(next_state)
        # next_q = next_qs.gather(1, next_qs.argmax(dim=1).unsqueeze(-1)).squeeze()
        # values, indices = next_qs.max(dim=1, keepdim=True)
        next_q, indices = next_qs.max(dim=1, keepdim=False) # indices はactionと同じ？
        next_q.detach()
        # 棒の位置に基づく報酬を計算
        # position_reward = 1.0 / (1.0 + abs(state[0][0]))
        # reward += position_reward
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)
        target = reward + (1 - done.float()) * self.gamma * next_q

        loss = TFunc.mse_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sync_qnet(self):
        self.qnet_target = copy.deepcopy(self.qnet)

episodes = 300
sync_interval = 20
env = gym.make('CartPole-v0', render_mode = 'human')
agent = DQNAgent()
reward_history = []


for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(state)
        next_state, reward, done, info, info1 = env.step(action)

        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    if episode % sync_interval == 0:
        agent.sync_qnet()

    reward_history.append(total_reward)
    if episode % 10 == 0:
        print("episode :{}, total reward : {}".format(episode, total_reward))


# === Plot ===
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.plot(range(len(reward_history)), reward_history)
plt.show()


# === Play CartPole ===
agent.epsilon = 0  # greedy policy
state = env.reset()
done = False
total_reward = 0

while not done:
    action = agent.get_action(state)
    next_state, reward, done, info = env.step(action)
    state = next_state
    total_reward += reward
    env.render()
print('Total Reward:', total_reward)
