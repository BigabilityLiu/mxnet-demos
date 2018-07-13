import sys
sys.path.append('..')
import gluonbook as gb
from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss, nn

# y=1.2(X)−3.4X(2)+5.6X(3)+5+ϵ

n_train = 100
n_test = 100
true_w = [1.2, 3.4, 5.6]
true_b = 5

features = nd.random.normal(shape=(n_train + n_test, 1))
poly_features = nd.concat(features, nd.power(features, 2), nd.power(features, 3))

labels = true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1] + true_w[2] * poly_features[:, 2] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

from IPython.display import set_matplotlib_formats
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    gb.plt.rcParams['figure.figsize'] = figsize
    set_matplotlib_formats('retina')
    gb.plt.xlabel(x_label)
    gb.plt.ylabel(y_label)
    gb.plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        gb.plt.semilogy(x2_vals, y2_vals)
        gb.plt.legend(legend)
    gb.plt.show()


num_epochs = 100
loss = gloss.L2Loss()

def fit_and_plot(train_features, test_features, train_labels, test_labels):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    batch_size = min(100, train_labels.shape[0])
    train_iter = gdata.DataLoader(
        gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle= True
    )
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})

    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(loss(net(train_features), train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features), test_labels).mean().asscalar())
    print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
    semilogy(range(1, num_epochs+1), train_ls, 'epochs', 'loss',
             range(1, num_epochs+1), test_ls, ['train', 'test'])
    return (net[0].weight.data(), net[0].bias.data())


(weight, bias) = fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:] )
print('weight:', weight, 'bias:', bias)

(weight, bias) = fit_and_plot(features[:n_train, :], features[n_train:, :], labels[:n_train], labels[n_train:])
print('weight:', weight, 'bias:', bias)

(weight, bias) = fit_and_plot(poly_features[0:2, :], poly_features[n_train:, :], labels[0:2], labels[n_train:] )
print('weight:', weight, 'bias:', bias)
