import math
from mxnet import nd, autograd
from mxnet.gluon import nn
import numpy as np
import time

def corr2d(X, K):
    n, m = K.shape
    Y = nd.zeros((X.shape[0] - n + 1, X.shape[1] - m + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i : i + n, j : j + m] * K).sum()
    return Y


def corr2d_multi_in(X, K):
    # 我们首先沿着 X 和 K 的第 0 维（通道维）遍历。然后使用 * 将结果列表 (list) 变成
    # add_n 的位置参数（positional argument）来进行相加。
    return nd.add_n(*[corr2d(x, k) for x, k in zip(X, K)])

def corr2d_multi_in_out(X, K):
    # 对 K 的第 0 维遍历，每次同输入 X 做相关计算。所有结果使用 nd.stack 合并在一起。
    return nd.stack(*[corr2d_multi_in(X, k) for k in K])

print("---1---")
X = nd.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
              [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = nd.array([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])
# K = nd.array([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])
Y = corr2d_multi_in(X, K)
print(X.shape, K.shape, Y.shape)
print(Y)
print("---2---")
K2 = nd.stack(K, K + 1, K + 2, K + 3)
print(K2.shape)
print(corr2d_multi_in_out(X, K2))
print("---3---")
X = nd.arange(60).reshape((5, 4, 3))
K = nd.array([[[0]], [[0]], [[0]], [[0]], [[0]]])
K = nd.stack(K + 1, K + 2)
Y = corr2d_multi_in_out(X, K)
print(X.shape, K.shape, Y.shape)
print(X, K, Y)
print(Y + Y)