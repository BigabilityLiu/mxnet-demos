import mxnet as mx
from mxnet import nd, init, gluon
from mxnet.gluon import nn, data as gdata, loss as gloss
import os
import gluonbook as gb

def vgg_block(num_convs, num_channels):
    blk = nn.Sequential()
    for _ in range(num_convs):
        blk.add(
            nn.Conv2D(channels=num_channels, kernel_size=3, padding=1, activation='relu')
        )
    blk.add(
        nn.MaxPool2D(pool_size=2, strides=2)
    )
    return blk


conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

def vgg(conv_arch):
    net = nn.Sequential()

    for num_convs, num_channels in conv_arch:
        net.add(vgg_block(num_convs, num_channels))

    net.add(
        nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
        nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
        nn.Dense(10)
    )
    return net

X = nd.random.uniform(shape=(1, 1, 224, 224))

# 出于测试的目的我们构造一个通道数更小，或者说更窄的网络来训练FashionMNIST。
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)

# net = vgg(conv_arch)

net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)


lr = 0.05
ctx = gb.try_gpu()
net.initialize(force_reinit=True, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size=128, resize=224)
loss = gloss.SoftmaxCrossEntropyLoss()
gb.train(train_iter, test_iter, net, loss, trainer, mx.cpu(), num_epochs=3)

net.load_params()
