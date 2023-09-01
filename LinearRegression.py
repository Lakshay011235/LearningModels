import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

X = np.array([1,2,3,4,5,6,7,8,9], dtype=float)
_Y = np.array([2,4,6,8,10,12,14,16,18], dtype=float)
Y = np.array([3,3,5,10,10,13,12,15,19], dtype=float)

plt.figure()
plt.plot(X, Y, marker="D")
# plt.show()

def loss_fn(Y_calc, Y_obs):
    result = 0
    for y1,y2 in zip(Y_calc, Y_obs):
        result += (y1-y2)**2
    return result

losses = []
indexes = []
# print(loss_fn(Y, _Y))
for w in range(20):
    Y1 = [(1.4 + w / 20)*y for y in range(1,10)]
    indexes.append((1.4 + w / 20))
    losses.append(loss_fn(Y1, _Y))
    plt.plot(X, Y1, marker="D")

# plt.plot(indexes, losses, marker="D")
plt.show()
"""
Linear Regression::

Y = W1 * X1 + B

dY = W1 * dX
(y2-y1)/(x2-x1) = W1  | for all pairs of x,y

"""
#
# x0,y0 = X[0],Y[0]
# old_mean = 0
# for i in range(1,len(X)):
#     dY = Y[i]-y0
#     dX = X[i]-x0
#     old_mean = dY//dX
#     print(dX,dY, dY//dX)
#     x0,y0 = X[i],Y[i]
# print(X * Y.T)
