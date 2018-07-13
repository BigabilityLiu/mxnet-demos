from mxnet import autograd, nd
import random
from matplotlib import pyplot as plt
import gluonbook as gb


num_inputs = 2
num_example = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_example, num_inputs))
#y = wx + b
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels = labels + nd.random.normal(scale=0.01, shape=num_example)
print(features[0], labels[0])

# plt.rcParams['figure.figsize'] = (3.5, 2.5)
# plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1)
# plt.show()

batch_size = 10
def data_iter(batch_size, features, labels):
    num_example = len(features)
    indices = list(range(num_example))
    random.shuffle(indices)
    for i in range(0, num_example, batch_size):
        j = nd.array(indices[i: min(i + batch_size, num_example)])
        yield features.take(j), labels.take(j)

# for X, y in data_iter(batch_size, features, labels):
#     print(X, y)
#     break


w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
b = nd.zeros(shape=(1,))
params = [w, b]
for param in params:
    param.attach_grad()


def linreg(X, w, b):
    return nd.dot(X, w) + b


def squard_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size


lr = 0.03
num_epochs = 3
net = linreg
loss = squard_loss
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)
        l.backward()
        sgd([w, b], lr, batch_size)
    m = loss(net(features, w, b), labels).mean()
    mn = m.asnumpy()
    print("epoch %d, loss %f" % (epoch, mn))

print(true_w, w)
print(true_b, b)