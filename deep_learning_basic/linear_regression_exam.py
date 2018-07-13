import mxnet as mx
import random
from mxnet import autograd

# y = Xw + b + nose

true_w = [2, -3.4]
true_b = 4.2
# 生成测试数据
num_inputs = 2
num_examples = 1000
features = mx.nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = features[:, 0] * true_w[0] + features[:, 1] * true_w[1] + true_b
labels += mx.nd.random.normal(scale=0.01, shape=labels.shape)

# read data
batch_size = 10
dataset = mx.gluon.data.ArrayDataset(features, labels)
data_iter = mx.gluon.data.DataLoader(dataset, batch_size, shuffle=True)

net = mx.gluon.nn.Sequential()
net.add(mx.gluon.nn.Dense(1))

net.initialize(mx.init.Normal(sigma=0.01))

#train

lr = 0.03
num_epochs = 3
loss = mx.gluon.loss.L2Loss()

trainer = mx.gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})

for i in range(1, num_epochs + 1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    print('epoch %d : loss = %f ' % (i, loss(net(features), labels).mean().asnumpy()))