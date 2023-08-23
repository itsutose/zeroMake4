import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = self.tanh(self.l2(x))
        return x

class Critic(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = self.l2(x)
        return x

class Agent:
    def __init__(self, input_size, hidden_size):
        self.actor = Actor(input_size, hidden_size, 1)
        self.critic = Critic(input_size, hidden_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = self.actor(state)
        return action.item()

    def train(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)

        # Actor loss
        advantage = reward + (1.0 - done) * self.critic(next_state) - self.critic(state)
        actor_loss = -torch.log(self.actor(state)) * advantage

        # Critic loss
        target_value = reward + (1.0 - done) * self.critic(next_state)
        critic_loss = nn.MSELoss()(self.critic(state), target_value.detach())

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

env = gym.make('Pendulum-v1')
agent = Agent(env.observation_space.shape[0], 128)

for episode in range(1000):
    state = env.reset()
    total_reward = 0

    while True:
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step([action])
        agent.train(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if done:
            break

    print(f"Episode {episode}, Reward: {total_reward}")
