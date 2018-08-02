import sys
sys.path.append('..')
import gluonbook as gb
from mxnet import nd, gluon, init
from mxnet.gluon import loss as gloss, nn

net = nn.Sequential()
net.add(
    nn.Conv2D(6, kernel_size=5),
    nn.BatchNorm(),
    nn.Activation('sigmoid'),
    nn.MaxPool2D(pool_size=2, strides=2),
    nn.Conv2D(16, kernel_size=5),
    nn.BatchNorm(),
    nn.Activation('sigmoid'),
    nn.MaxPool2D(pool_size=2, strides=2),
    nn.Dense(120),
    nn.BatchNorm(),
    nn.Activation('sigmoid'),
    nn.Dense(84),
    nn.BatchNorm(),
    nn.Activation('sigmoid'),
    nn.Dense(10)
)

lr = 1.0
num_epochs = 5
batch_size = 256
ctx = gb.try_gpu()
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
loss = gloss.SoftmaxCrossEntropyLoss()
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)
# gb.train_ch5(net, train_iter, test_iter, loss, batch_size, trainer, ctx, num_epochs)
gb.train_cpu(net, train_iter, test_iter, loss, num_epochs, batch_size, None, lr, trainer)