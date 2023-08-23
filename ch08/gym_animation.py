import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import gym


# 状態として利用する値を指定
position_vals = np.arange(-2.4, 2.41, step=0.1).round(1) # カートの位置
angle_vals = np.arange(-3.1, 3.11, step=0.1).round(1) # ポールの角度

# フレーム数を設定
# frame_num = len(position_vals)
frame_num = len(angle_vals)
print(frame_num)

# 図を初期化
fig = plt.figure(figsize=(9, 7), facecolor='white')
fig.suptitle('Cart Pole', fontsize=20)




# 作図処理を関数として定義
def update(i):
    # i番目の値を取得
    # x = position_vals[i]
    theta = angle_vals[i]
    
    # 固定する値を指定
    x = 0.0
    x_dot = 0.0
    # theta = 0.0
    theta_dot = 0.0
    
    # インスタンスを初期化
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    _, _ = env.reset()
    
    # 状態を設定
    state = np.array([x, x_dot, theta, theta_dot])
    env.env.env.env.__dict__['state'] = state
    
    # 画像データを作成
    rgb_data = env.render()
    
    # 状態ラベルを作成
    state_text = f'cart position={x:5.2f}, '
    state_text += 'cart velocity=' + str(x_dot) + '\n'
    state_text += f'pole angle   ={theta:5.2f}, '
    state_text += 'pole velocity=' + str(theta_dot)
    
    # カートポールを描画
    plt.imshow(rgb_data)
    plt.xticks(ticks=[])
    plt.yticks(ticks=[])
    plt.title(state_text, loc='left')

# gif画像を作成
anime = FuncAnimation(fig=fig, func=update, frames=frame_num, interval=100)

# gif画像を保存
anime.save('CartPole_state2.gif')