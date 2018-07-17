import sys
sys.path.insert(0, '..')
import gluonbook as gb
from mxnet import nd, gluon, init
from mxnet.gluon import loss as gloss, nn

def nin_block(num_channels, kernel_size, strides, padding):
    blk = nn.Sequential()
    blk.add(
        nn.Conv2D(channels=num_channels, kernel_size=kernel_size, strides=strides, padding=padding),
        nn.Conv2D(channels=num_channels, kernel_size=1, activation='relu'),
        nn.Conv2D(channels=num_channels, kernel_size=1, activation='relu')
    )
    return blk

net = nn.Sequential()

net.add(
    nin_block(96, 11, 4, 0),
    nn.MaxPool2D(pool_size=3, strides=2),
    nin_block(256, 5, 1, 2),
    nn.MaxPool2D(pool_size=3, strides=2),
    nin_block(384, 3, 1, 1),
    nn.MaxPool2D(pool_size=3, strides=2),
    nn.Dropout(0.5),

    nin_block(10, 3, 1, 1),
    nn.GlobalAvgPool2D(),
    nn.Flatten()
)

X = nd.random.uniform(shape=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)


lr = 0.1
ctx = gb.try_gpu()
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
loss = gloss.SoftmaxCrossEntropyLoss()
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size=128, resize=224)
gb.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs=3)