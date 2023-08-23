import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functions as F

# トイ・データセット
np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)  # 省略可能

x.name = f'input'
y.name = f'target'

W = Variable(np.zeros((1, 1)), name = "W")
W = Variable(np.zeros(1))
b = Variable(np.zeros(1), name = "b")
# b = Variable(np.zeros(1))

def predict(x):
    y = F.matmul(x, W) + b
    return y

def mean_squared_error(x0, x1):
    diff = x0 - x1
    return F.sum(diff ** 2) / len(diff)

lr = 0.1
iters = 1

for i in range(iters):
    y_pred = predict(x)
    
    y_pred.name = f'output_{i}'
    
    loss = mean_squared_error(y, y_pred)


    W.cleargrad()
    b.cleargrad()
    loss.backward()
    loss.name = 'loss'

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data

    if i % 10 == 0:
        print(loss.data)

print('====')
print('W =', W.data)
print('b =', b.data)

# Plot
plt.scatter(x.data, y.data, s=10)
plt.xlabel('x')
plt.ylabel('y')
t = np.arange(0, 1, .01)[:, np.newaxis]
y_pred = predict(t)
plt.plot(t, y_pred.data, color='r')
plt.show()