import numpy as np
import matplotlib.pyplot as plt
from dezero import Model
from dezero import optimizers
import modified_layers as L
import dezero.layers as L
import dezero.functions as F
from dezero.core import Parameter

# Dataset
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.2
iters = 10000

class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        # self.l1.W = Parameter(None, name='W1')
        # self.l1.b = Parameter(np.zeros(self.l1.out_size, dtype=self.l1.dtype), name='b1')

        self.l2 = L.Linear(out_size)
        # self.l2.W = Parameter(None, name='W2')
        # self.l2.b = Parameter(np.zeros(self.l2.out_size, dtype=self.l2.dtype), name='b2')

    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y

model = TwoLayerNet(10, 1)
# optimizerを追加
optimizer = optimizers.SGD(lr)
optimizer.setup(model)

for i in range(iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    # for p in model.params():
    #     p.data-= le * p.grad.data
    optimizer.update()


    if i % 1000 == 0:
        print(loss.data)

# Plot
plt.scatter(x, y, s=10)
plt.xlabel('x')
plt.ylabel('y')
t = np.arange(0, 1, .01)[:, np.newaxis]
y_pred = model(t)
plt.plot(t, y_pred.data, color='r')
plt.show()

