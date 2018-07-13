import sys

sys.path.append('..')
import gluonbook as gb
from mxnet import autograd, gluon, nd

n_train = 20
n_test = 100

num_inputs = 200
true_w = nd.ones((num_inputs, 1)) * 0.01
true_b = 0.05

features = nd.random.normal(shape=(n_train + n_test, num_inputs))
labels = nd.dot(features, true_w) + true_b
labels += nd.random.normal(scale=0.01,shape=labels.shape)

train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]


def init_params():
    w = nd.random.normal(scale=1, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    params = [w, b]
    for param in params:
        param.attach_grad()
    return params


def l2_penalty(w):
    return (w**2).sum()/2


batch_size = 1
num_epochs = 10
lr = 0.003
net = gb.linreg
loss = gb.squared_loss


def fit_and_plot(lambd):
    w, b = params = init_params()
    train_ls = []
    test_ls = []
    for _ in range(num_epochs):
        for X, y in gb.data_iter(batch_size, features.shape[0], features, labels):
            with autograd.record():
                l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            l.backward()
            gb.sgd(params, lr, batch_size)
        train_ls.append(loss(net(train_features, w, b), train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features, w, b), test_labels).mean().asscalar())
    gb.semilogy(range(1, num_epochs+1), train_ls, 'epochs', 'loss',
                range(1, num_epochs+1), test_ls, ['train', 'test'])
    return 'w[:10]:', w[:10].T, 'b:', b


print(fit_and_plot(lambd=0))
print(fit_and_plot(lambd=5))
print(fit_and_plot(lambd=10))