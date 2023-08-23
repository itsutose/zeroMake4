if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import gym
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L

class Policy(Model):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(128)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.tanh(self.l2(x))
        return x

class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0002
        self.action_size = 1

        self.memory = []
        self.pi = Policy(self.action_size)
        self.optimizer = optimizers.Adam(self.lr)
        self.optimizer.setup(self.pi)
        self.action_log_std = np.zeros(1, dtype=np.float32)

    def get_action(self, state):
        if isinstance(state, tuple):
            state = state[0]
        state = state[np.newaxis, :]
        action_mean = self.pi(state)
        action_std = F.exp(self.action_log_std)
        normal = np.random.normal(0, 1, size=action_mean.shape)
        action = action_mean + action_std * normal
        return action, action_mean

    def add(self, reward, prob):
        data = (reward, prob)
        self.memory.append(data)

    def update(self):
        self.pi.cleargrads()

        G, loss = 0, 0
        for reward, prob in reversed(self.memory):
            G = reward + self.gamma * G
            loss += -F.log(prob) * G

        loss.backward()
        self.optimizer.update()
        self.memory = []

episodes = 3000
env = gym.make('Pendulum-v1', render_mode = 'human')
agent = Agent()
reward_history = []

for episode in range(episodes):
    state = env.reset()
    done = False
    sum_reward = 0

    while not done:
        action, prob = agent.get_action(state)
        next_state, reward, done, info, info1 = env.step(action)

        agent.add(reward, prob)
        state = next_state
        sum_reward += reward

    agent.update()

    reward_history.append(sum_reward)
    if episode % 100 == 0:
        print("episode :{}, total reward : {:.1f}".format(episode, sum_reward))

# plot
from common.utils import plot_total_reward
plot_total_reward(reward_history)