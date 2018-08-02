import sys
sys.path.append('..')
import gluonbook as gb
from mxnet import nd, gluon, init
from mxnet.gluon import loss as gloss, nn

def conv_block(num_channels):
    blk = nn.Sequential()
    blk .add(nn.BatchNorm(),
             nn.Activation('relu'),
             nn.Conv2D(num_channels, kernel_size=3, padding=1))
    return blk

def transition_block(num_channels):
    blk = nn.Sequential()
    blk.add(
        nn.BatchNorm(),
        nn.Activation('relu'),
        nn.Conv2D(num_channels, kernel_size=1),
        nn.AvgPool2D(pool_size=2, strides=2)
    )
    return blk

class DenseBlock(nn.Block):
    def __init__(self, num_convs, num_channels, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add(conv_block(num_channels))

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = nd.concat(X, Y, dim=1)
        return Y

# blk = DenseBlock(2, 10)
# blk.initialize()
# X = nd.random.uniform(shape=(4, 3, 8, 8))
# Y = blk(X)
# print(Y.shape)
#
# blk = transition_block(10)
# blk.initialize()
# print(blk(Y).shape)

net = nn.Sequential()
net.add(
    nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
    nn.BatchNorm(),
    nn.Activation('relu'),
    nn.MaxPool2D(pool_size=3, strides=2, padding=1)
)
num_channels = 64
growth_rate = 32
num_convs_in_dense_block = [4, 4, 4, 4]

for i, num_convs in enumerate(num_convs_in_dense_block):
    net.add(DenseBlock(num_convs, growth_rate))
    num_channels += num_convs * growth_rate
    if i != len(num_convs_in_dense_block) - 1:
        net.add(transition_block(num_channels // 2))

net.add(
    nn.BatchNorm(),
    nn.Activation('relu'),
    nn.GlobalAvgPool2D(),
    nn.Dense(10)
)
lr = 0.1
num_epochs = 5
batch_size = 256
ctx = gb.try_gpu()
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
loss = gloss.SoftmaxCrossEntropyLoss()
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size=batch_size,
                                                   resize=96)
gb.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)