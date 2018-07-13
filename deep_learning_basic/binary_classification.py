import sys
import mxnet as mx
from mxnet import gluon, autograd
from mxnet import ndarray as nd
import gluonbook as gb

def transform(feature, label):
    return feature.astype('float32') / 255, label.astype('float32')

mnist_train = gluon.data.vision.FashionMNIST(train = True, transform=transform)
mnist_test = gluon.data.vision.FashionMNIST(train=False, transform=transform)

# feature, label = mnist_train[0]
# print('feature shape: ', feature.shape, 'label: ', label)

def get_text_labels(labels):
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in labels]

batch_size = 256
train_iter = gluon.data.DataLoader(mnist_train, batch_size, True)
test_iter = gluon.data.DataLoader(mnist_test, batch_size, False)

num_inputs = 784
num_outputs = 10

W = nd.random.normal(scale=0.01, shape=(num_inputs, num_outputs))
b = nd.zeros(num_outputs)

params = [W, b]

for param in params:
    param.attach_grad()


def softmax(X):
    exp = X.exp()
    partition = exp.sum(axis=1, keepdims=True)
    return exp / partition


def net(X):
    return softmax(nd.dot(X.reshape((-1, num_inputs)), W) + b)


def cross_entropy(y_hat, y):
    return -nd.pick(y_hat.log(), y)


def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y).mean().asscalar()


def evaluate_accuracy(data_iter, net):
    acc = 0
    for X, y in data_iter:
        acc += accuracy(net(X), y)
    return acc / len(data_iter)


num_epochs = 10
lr = 0.1
loss = cross_entropy

def train_cpu(net, train_iter, test_iter, loss, num_epochs, batch_size,params=None, lr=None, trainer=None):
    for epoch in range(0, num_epochs):
        train_l_sum = 0
        train_acc_sum = 0
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            if trainer is None:
                gb.sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size)
            train_l_sum += l.mean().asscalar()
            train_acc_sum += accuracy(y_hat, y)
        test_acc = evaluate_accuracy(test_iter, net)
        print("epoch %d, loss %.4f, train acc %.3f, test acc %.3f"
              % (epoch, train_l_sum / len(train_iter),
                 train_acc_sum / len(train_iter), test_acc))
        print(W, b)

train_cpu(net, train_iter, test_iter, loss, num_epochs, batch_size, params,lr)