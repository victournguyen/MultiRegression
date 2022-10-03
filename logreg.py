from math import exp, log
import random
import matplotlib.pyplot as plt
import numpy as np

x = [random.random() for i in range(1000)]
y = [1 if z > 0.5 else 0 for z in x]

n = len(x)
b0 = b1 = 0

def prob(i):
    global b0, b1, x
    return 1 / (1 + exp(-(b0 + b1 * x[i])))

def log_loss():
    global y, n
    loss = 0
    for i in range(n):
        loss += log(prob(i)) if y[i] == 1 else log(1 - prob(i))
    loss *= -1 / n
    return loss

def iterate(num, lr):
    global b0, b1, x, y, n
    print(log_loss())
    for ep in range(num):
        db0 = db1 = 0
        for i in range(n):
            # print(b0, b1, x[i])
            temp = prob(i) * (exp(-(b0 + b1 * x[i])) if y[i] == 1 else -1)
            db0 += temp
            db1 += x[i] * temp
        db0 *= -1 / n
        db1 *= -1 / n
        b0 -= lr * db0
        b1 -= lr * db1
        try:
            print(log_loss())
        except ValueError:
            print('diverges, decrease learning rate')
            return False
    return True

iters = iterate(100, 0.5)
print(b0, b1)
if iters:
    a = np.linspace(start=0, stop=1, num=100)

    b = [1 / (1 + exp(-(b0 + b1 * xval))) for xval in a]

    plt.scatter(x, y, s=2)
    plt.plot(a, b)
    plt.show()
