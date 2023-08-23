# ライブラリを読み込み
import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 環境のインスタンスを作成
env = gym.make('CartPole-v1', render_mode='rgb_array')

# 状態を初期化
state, info = env.reset()
print(state)
print(info)

# 画像データを作成
rgb_data = env.render()
print(rgb_data.shape)
print(rgb_data[250:300, 300:350, 0])

# 状態ラベルを作成
state_text = f'cart position={state[0]:5.2f}, '
state_text += f'cart velocity={state[1]:6.3f}\n'
state_text += f'pole angle   ={state[2]:5.2f}, '
state_text += f'pole velocity={state[3]:6.3f}'

# カートポールを描画
plt.figure(figsize=(9, 7), facecolor='white')
plt.suptitle('Cart Pole', fontsize=20)
plt.imshow(rgb_data)
plt.xticks(ticks=[])
plt.yticks(ticks=[])
plt.title(state_text, loc='left')
plt.show()

