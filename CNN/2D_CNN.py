import mxnet as mx
from mxnet import nd, autograd
from mxnet.gluon import nn

def corr2d(x, k):
    h, w = k.shape
    y = nd.zeros(shape=(x.shape[0] - h + 1, x.shape[1] - w + 1))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i][j] = (x[i: i + h, j: j + w] * k).sum()
    return y

x = nd.arange(9).reshape((3, 3))
k = nd.arange(4).reshape((2, 2))
y = corr2d(x, k)
print(x, k, y)
print('---1---')

class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super(Conv2D, self).__init__(**kwargs)

        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()


X = nd.ones((6, 8))
X[:,2:6] = 0
print(X)
K = nd.array([[1, -1]])
Y = corr2d(X, K)
print(Y)
print('---2---')
conv2d = nn.Conv2D(1, kernel_size=(1, 2))
conv2d.initialize()

X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

for i in range(10):
    with autograd.record():
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
        if i % 2 == 1:
            print('epoch %d loss = %.3f' % (i, l.sum().asscalar()))
    l.backward()

    conv2d.weight.data()[:] -= 3e-2 * conv2d.weight.grad()

print(conv2d.weight.data().reshape((1, 2)))

print('---3---')
# Qustion 3  return nd.Convolution(X, self.weight.data(), self.bias.data())
# myConv2d = Conv2D((1, 2))
# myConv2d.initialize()
#
# X = X.reshape((1, 1, 6, 8))
# Y = Y.reshape((1, 1, 6, 7))
#
# for i in range(10):
#     with autograd.record():
#         Y_hat = myConv2d(X)
#         l = (Y_hat - Y) ** 2
#         if i % 2 == 1:
#             print('epoch %d loss = %.3f' % (i, l.sum().asscalar()))
#     l.backward()
#
#     myConv2d.weight.data()[:] -= 3e-2 * myConv2d.weight.grad()
#
# print(myConv2d.weight.data().reshape((1, 2)))

print('---4---')
X = nd.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
              [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = nd.array([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])

def corr2d_multi_in(X, K):
    # 我们首先沿着 X 和 K 的第 0 维（通道维）遍历。然后使用 * 将结果列表 (list) 变成
    # add_n 的位置参数（positional argument）来进行相加。
    return nd.add_n(*[corr2d(x, k) for x, k in zip(X, K)])

print(corr2d_multi_in(X, K))
print(X.shape)
print(K.shape)
K = nd.stack(K, K + 1, K + 2)
print(K)

nn.MaxPool2D()