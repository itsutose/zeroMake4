import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # for importing the parent dirs
from collections import defaultdict
import numpy as np
from common.gridworld import GridWorld


class McAgent:
    def __init__(self, epsilon = 0.1, alpha = 0.1):
        self.gamma = 0.9
        self.epsilon = epsilon
        self.alpha = alpha
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        self.cnts = defaultdict(lambda: 0)
        self.memory = []

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
    
    def add(self, state, action, reward):
        data = (state, action, reward)
        self.memory.append(data)

    def reset(self):
        self.memory.clear()

    def eval(self):
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma * G + reward
            self.cnts[state] += 1
            self.Q[state] += (G - self.Q[state]) / self.cnts[state]

    def update(self, improve = False):
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma * G + reward
            key = (state,action)

            self.cnts[key] += 1
            if improve == True:
                self.Q[key] += (G - self.Q[key]) * self.alpha
            else:
                self.Q[key] += (G - self.Q[key]) /self.cnts[key]

            self.pi[state] = greedy_probs(self.Q, state, self.epsilon)

def greedy_probs(Q, state, epsilon = 0, action_size=4):
    # 初期化 Qがupdateのところで代入されていなければここで0が入る
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = np.argmax(qs)

    base_prob = epsilon / action_size
    action_probs = {action: base_prob for action in range(action_size)}
    action_probs[max_action] += (1 - epsilon)
    return action_probs

if __name__ == '__main__':
    env = GridWorld()
    agent = McAgent(epsilon=0.1)

    episodes = 2
    for episode in range(episodes):
        state = env.reset()
        agent.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.add(state, action, reward)
            if done:
                agent.update(improve = False)
                env.render_q(agent.Q) 
                break
            
            state = next_state
        
    env.render_q(agent.Q)