if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import matplotlib.pyplot as plt
import numpy as np
from torchviz import make_dot
import torch.distributions as dist
from torch.distributions import Normal
import os

# 現在のスクリプトの絶対パスを取得
absolute_path = os.path.abspath(__file__)

# 現在のスクリプトのディレクトリを取得
directory_path = os.path.dirname(absolute_path)

# 現在のスクリプトのファイル名を取得
filename = os.path.basename(absolute_path)

# print(directory_path, filename)
# print(filename.replace('.py','_py'))

# VSCodeでデバッグモードかどうかを判定
is_debugging = os.environ.get('PYTHONDEBUG') is not None

path = os.path.join(directory_path,filename)

# Define the Policy network using PyTorch
class Policy(nn.Module):

    def __init__(self, input_size = 3, hidden_size = 128, action_size = 1):
        super(Policy, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.mean_layer = nn.Linear(hidden_size, 1)  # Output layer for the mean (mu)
        self.std_layer = nn.Linear(hidden_size, 1)   # Output layer for the std (sigma)
        
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        
        mean = self.mean_layer(x)
        std_dev = F.softplus(self.std_layer(x))  # Ensure that std_dev is positive
        
        return mean, std_dev
    
# from torchsummary import summary
# summary(Policy,(1,3,128,1))

# Define the Agent class
class Agent:
    def __init__(self, input_size, hidden_size, action_size):
        self.gamma = 0.98
        self.lr = 0.001
        self.memory = []
        self.policy = Policy(input_size, hidden_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)


    def get_action(self, state):
        if isinstance(state, tuple):
            state = state[0]
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float32)
        mean, std_dev = self.policy.forward(state)

        # Create a normal distribution parameterized by mean and std_dev
        normal_distribution = Normal(mean, std_dev)

        # Sample an action from the normal distribution
        action = normal_distribution.sample()

        # Clip the action to be within the allowed range [-2, 2]
        action = torch.clamp(action, min=-2, max=2)

        # Calculate the log probability of the action
        log_prob = normal_distribution.log_prob(action)

        return action, log_prob


        
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

        if is_debugging:
            # 計算グラフを表示
            img = make_dot(policy_loss, params=dict(self.policy.named_parameters()))
            img.view(filename=filename, directory=directory_path)

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        self.memory = []

if __name__ == '__main__':

    # Initialize the environment and the agent
    env = gym.make('Pendulum-v1', render_mode = 'human')
    agent = Agent(3, 128, 1)
    done = False
    reward_history = []

    num_state = env.observation_space.shape
    num_action = env.action_space.shape
    max_steps = env.spec.max_episode_steps

    max_episode = 500

    for episode in range(max_episode):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action, prob = agent.get_action(state)
            step_results = env.step([action.item()])
            # print(step_results)
            next_state, reward, done, _, info = step_results
            if done == True:
                break

            # print(action, prob)
            agent.add(reward, prob)
            state = next_state
            total_reward += reward

        episode += 1
        
        print(f"Episode: {episode}, Accumulated Reward: {total_reward}")

        agent.update()
        reward_history.append(total_reward)


    # plot_reward_history(reward_history)
    from common.utils import plot_total_reward
    plot_total_reward(reward_history)


# https://chat.openai.com/c/18fb68f3-e7d4-4870-a22e-756c1861328a