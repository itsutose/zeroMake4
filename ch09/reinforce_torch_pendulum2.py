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

    def __init__(self, input_size, hidden_size, action_size):
        super(Policy, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, action_size)        
        self.l5 = nn.Linear(hidden_size , 1)      
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))
        
        action_probs = torch.sigmoid(self.l4(x))  # 0~1 because of Sigmoidmax        
        scaled_output = torch.tanh(self.l5(x)) * 2  # -2~2 because of Tanh and scaling

        return action_probs, scaled_output

# Define the Agent class
class Agent:
    def __init__(self, input_size, hidden_size, action_size):
        self.gamma = 0.98
        self.lr = 0.001
        self.memory = []
        self.policy = Policy(input_size, hidden_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

    
        # policyの返り値は0~1
    def get_action(self, state):
        if isinstance(state, tuple):
            state = state[0]
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float32)
        action_prob, action = self.policy(state)
        return action, action_prob


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
            policy_loss += -torch.log(prob).unsqueeze(0) * G
            # policy_loss += - prob * G

        if is_debugging:
            # 計算グラフを表示
            img = make_dot(policy_loss, params=dict(self.policy.named_parameters()))
            img.view(filename=filename, directory=directory_path)

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        self.memory = []


# Initialize the environment and the agent
env = gym.make('Pendulum-v1', render_mode = 'human')
agent = Agent(3, 128, 1)
done = False
reward_history = []
# step_counter = 0

num_state = env.observation_space.shape
num_action = env.action_space.shape
max_steps = env.spec.max_episode_steps

max_episode = 100

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

        print(action, prob)
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