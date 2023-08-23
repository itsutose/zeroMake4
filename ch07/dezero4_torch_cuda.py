import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
# import tkinter

# gpuが使用される場合の設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )

print("CUDA available:" , torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current GPU:", torch.cuda.current_device())
    print("GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))

# 1. Import necessary modules
torch.manual_seed(0)

# 2. Create dataset
# x = torch.tensor(np.random.rand(100, 1), dtype=torch.float32).to(device)
x = torch.tensor(torch.rand(100, 1), dtype=torch.float32).to(device)
# y = (torch.sin(2 * np.pi * x) + torch.rand(100, 1)).to(device).detach()
y = (torch.sin(2 * np.pi * x) + torch.rand(100, 1, device=device)).detach().to(device)

# 3. Define the model
class TwoLayerNet(nn.Module):
    def __init__(self, hidden_size, out_size):
        super(TwoLayerNet, self).__init__()
        self.l1 = nn.Linear(1, hidden_size)
        self.l2 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        y = torch.sigmoid(self.l1(x))
        y = self.l2(y)
        return y

model = TwoLayerNet(10, 1).to(device)

# 4. Set loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.2)

# 5. Training loop
iters = 10000
for i in range(iters):
    y_pred = model(x)
    loss = criterion(y, y_pred)

    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 1000 == 0:
        print(loss.item())

# 6. Plot results after training
plt.scatter(x.cpu().numpy(), y.cpu().numpy(), s=10)
plt.xlabel('x')
plt.ylabel('y')
t = torch.tensor(np.arange(0, 1, .01).reshape(-1, 1), dtype=torch.float32).to(device)
y_pred = model(t)
plt.plot(t.cpu().numpy(), y_pred.detach().cpu().numpy())
plt.show()
# plt.savefig('filename.png')