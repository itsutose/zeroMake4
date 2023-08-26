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
        self.l2 = nn.Linear(hidden_size, action_size)
        self.l3 = nn.Linear(hidden_size, 1)  # For 0~1 output
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = torch.relu(self.l1(x))
        
        # action_probs = self.softmax(self.l2(x))  # 0~1 because of Softmax
        action_probs = torch.sigmoid(self.l2(x))  # 0~1 because of Sigmoid
        # bounded_output = torch.sigmoid(self.l3(x))  # 0~1 because of Sigmoid
        scaled_output = torch.tanh(self.l3(x)) * 2  # -2~2 because of Tanh and scaling

        return action_probs, scaled_output
    
from torchsummary import summary
summary(Policy,(1,3,128,1))