import numpy as np
import gym
import time
# import pygame
# pygame.init()

env = gym.make('CartPole-v1', render_mode="human")
# env = gym.make('CartPole-v0')

state = env.reset()
done = False

while not done:
    env.render()
    time.sleep(0.05)  # 0.05秒待機
    action = np.random.choice([0, 1])
    next_state, reward, done, info, info1 = env.step(action)

# # エピソードが終了した後、ウィンドウを手動で閉じるまで待機
# while True:
#     env.render()

