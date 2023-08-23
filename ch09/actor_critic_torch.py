import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PolicyNet(nn.Module):
    def __init__(self, action_size=2):
        super(PolicyNet, self).__init__()
        self.l1 = nn.Linear(4, 128)  # Assuming input size is 4 as in standard CartPole
        self.l2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        x = F.softmax(x, dim=1)
        return x

class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.l1 = nn.Linear(4, 128)  # Assuming input size is 4 as in standard CartPole
        self.l2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr_pi = 0.0002
        self.lr_v = 0.0005
        self.action_size = 2

        self.pi = PolicyNet()
        self.v = ValueNet()
        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.lr_pi)
        self.optimizer_v = optim.Adam(self.v.parameters(), lr=self.lr_v)

    def get_action(self, state):
        if isinstance(state, tuple):
            state = state[0]
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float32)
        probs = self.pi(state)
        probs = probs[0]
        action = torch.multinomial(probs, 1).item()
        return action, probs[action]

    def update(self, state, action_prob, reward, next_state, done):
        if isinstance(state, tuple):
            state = state[0]
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float32)
        next_state = torch.tensor(next_state[np.newaxis, :], dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)

        # (1) Update V network
        target = reward + self.gamma * self.v(next_state) * (1 - done)
        v = self.v(state)
        loss_v = F.mse_loss(v, target.detach())

        self.optimizer_v.zero_grad()
        loss_v.backward()
        self.optimizer_v.step()

        # (2) Update pi network
        delta = (target - v).detach()
        loss_pi = -torch.log(action_prob) * delta

        self.optimizer_pi.zero_grad()
        loss_pi.backward()
        self.optimizer_pi.step()

if __name__ == "__main__":
    env = gym.make('CartPole-v0', render_mode = 'human')
    agent = Agent()
    reward_history = []
    episodes = 3000

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            get_action = agent.get_action(state)
            a, prob_a = get_action
            env_step = env.step(a)
            next_state, reward, done, info, _ = env_step

            agent.update(state, prob_a, reward, next_state, done)

            state = next_state
            total_reward += reward

        reward_history.append(total_reward)

        if episode % 100 == 0:
            print("episode :{}, total reward : {:.1f}".format(episode, total_reward))

    # plot
    from common.utils import plot_total_reward
    plot_total_reward(reward_history)

    # https://chat.openai.com/c/f641d338-3326-4f9f-b2be-908c6a3fc8e9