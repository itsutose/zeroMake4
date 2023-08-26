import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchviz import make_dot
from IPython.display import display

num_classes = 10

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = Net()

# 適当な入力
x = torch.randn(1, 28*28)
# 出力
y = model(x)


import os
directory_path = r"H:\マイドライブ\pytest\virtual_currency\gmo\zeroMake4\ch09"
filename = "torch_viz_py"

path = os.path.join(directory_path,filename)
print(path)

# 計算グラフを表示
img = make_dot(y, params=dict(model.named_parameters()))
# 出力するファイル名（拡張子は不要）

# グラフをレンダリングしてPDFと.gvファイルを生成
# 以下の二つはimg.viewで一括で行うことができる
# img.render(filename=path, format="gv", cleanup=True)
# img.render(filename=path, format="pdf", cleanup=True)
img.view(filename=filename, directory=directory_path)


img.render(filename=path, format="png", cleanup=True)
display(img)
