import sys
sys.path.append('..')
import gluonbook as gb
import mxnet as mx
from mxnet import nd, gluon, init
from mxnet.gluon import loss as gloss, nn

net = nn.Sequential()
net.add(
    nn.Conv2D(channels=6, kernel_size=5, activation='sigmoid'),
    nn.MaxPool2D(pool_size=2, strides=2),
    nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
    nn.MaxPool2D(pool_size=2, strides=2),
    # Dense 会默认将（批量大小，通道，高，宽）形状的输入转换成
    #（批量大小，通道 x 高 x 宽）形状的输入。
    nn.Dense(120, activation='sigmoid'),
    nn.Dense(84, activation='sigmoid'),
    nn.Dense(10)
)
# epoch 1, loss 2.3202, train acc 0.099, test acc 0.100, time 14.3 sec
# epoch 2, loss 2.0335, train acc 0.216, test acc 0.483, time 14.4 sec
# epoch 3, loss 0.9872, train acc 0.606, test acc 0.653, time 15.2 sec
# epoch 4, loss 0.7591, train acc 0.705, test acc 0.720, time 14.7 sec
# epoch 5, loss 0.6686, train acc 0.735, test acc 0.738, time 14.5 sec
X = nd.random.uniform(shape=(1, 1, 28, 28))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)


train_iter, test_iter = gb.load_data_fashion_mnist(batch_size=256)
def try_gpu():
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx

ctx = try_gpu()
lr = 1.0
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
loss = gloss.SoftmaxCrossEntropyLoss()
gb.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs=5)