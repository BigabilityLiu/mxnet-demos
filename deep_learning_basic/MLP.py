import sys
sys.path.append('..')
import gluonbook as gb
from mxnet import autograd, gluon, nd
from mxnet.gluon import loss as gloss

batch_size = 256
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10
num_hiddens = 256

W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens))
b1 = nd.zeros(num_hiddens)
W2 = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs))
b2 = nd.zeros(num_outputs)

params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()


def relu(X):
    return nd.maximum(X, 0)


def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(nd.dot(X, W1) + b1)
    return nd.dot(H, W2) + b2

loss = gloss.SoftmaxCrossEntropyLoss()

num_epochs = 5
lr = 0.5

gb.train_cpu(net, train_iter, test_iter, loss, num_epochs, batch_size,
             params, lr)
