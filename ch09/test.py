import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = self.softmax(self.l2(x))
        return x


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = self.l2(x)
        return x


class Agent:
    def __init__(self, input_size, hidden_size, action_size):
        self.actor = Actor(input_size, hidden_size, action_size)
        self.critic = Critic(input_size, hidden_size)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

    def get_action(self, state):
        if isinstance(state, tuple):
            state = state[0]
        state = torch.tensor(state[np.newaxis, :] , dtype=torch.float32)
        probs = self.actor(state)
        action = torch.multinomial(probs, 1).item()
        return action

    def train(self, state, action, reward, next_state, done):
        if isinstance(state, tuple):
            state = state[0]
        state = torch.tensor(state[np.newaxis, :] , dtype=torch.float32)
        next_state = torch.tensor(next_state[np.newaxis, :] , dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        
        # Actor loss
        probs = self.actor(state)
        prob = probs[action]
        value = self.critic(state)
        next_value = self.critic(next_state)
        
        advantage = (reward + 0.99 * next_value * (1 - done) - value).detach()
        actor_loss = -torch.log(prob) * advantage
        
        # Critic loss
        target = reward + 0.99 * next_value * (1 - done)
        critic_loss = (target - value).pow(2)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


if __name__ == '__main__':
    env = gym.make('Pendulum-v1', render_mode = 'human')
    agent = Agent(env.observation_space.shape[0], 128, 1)
    
    state = env.reset()
    done = False
    total_reward = 0
    step_counter = 0
    accumulated_reward = 0
    episode_reward = 0

    while not done:
        action = agent.get_action(state)
        env_step= env.step([action])
        next_state, reward, done, info, _ = env_step
        agent.train(state, action, reward, next_state, done)

        state = next_state

        step_counter += 1
        accumulated_reward += reward
        
        if step_counter % 100 == 0:
            print(f"Steps: {step_counter}, Accumulated Reward: {accumulated_reward}")
            accumulated_reward = 0  # reset the accumulated reward

# plot_reward_history(reward_history)
# from common.utils import plot_total_reward
# plot_total_reward(reward_history)

# https://chat.openai.com/c/39d9a1ee-3a52-4046-93f0-e47ef8b7b433