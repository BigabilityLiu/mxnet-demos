import mxnet as mx
from mxnet import nd, init, gluon
from mxnet.gluon import nn, data as gdata, loss as gloss
import os
import gluonbook as gb

drop_prob1 = 0.2
drop_prob2 = 0.5

net = nn.Sequential()

net.add(
    nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
    nn.MaxPool2D(pool_size=3, strides=2),
    nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
    nn.MaxPool2D(pool_size=3, strides=2),
    nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
    nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
    nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
    nn.MaxPool2D(pool_size=3, strides=2),

    nn.Dense(4096, activation='relu'),
    nn.Dropout(0.5),
    nn.Dense(4096, activation='relu'),
    nn.Dropout(0.5),
    nn.Dense(10)
)

X = nd.random.uniform(shape=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)

def load_data_fashion_mnist(batch_size, resize=None,
                            root=os.path.join('~','.mxnet','datasets','fashion-mnist')):

    # root ~/.mxnet/datasets/fashion-mnist
    root = os.path.expanduser(root)
    # /Users/techcul/.mxnet/datasets/fashion-mnist
    transformer = []
    if resize:
        transformer += [gdata.vision.transforms.Resize(resize)]
    transformer += [gdata.vision.transforms.ToTensor()]
    transformer = gdata.vision.transforms.Compose(transformer)
    mnist_train = gdata.vision.FashionMNIST(root=root, train=True)
    mnist_test = gdata.vision.FashionMNIST(root=root, train=False)
    train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),
                                  batch_size, shuffle=True, num_workers=4)
    test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
                                 batch_size, shuffle=False, num_workers=4)
    return train_iter, test_iter

train_iter, test_iter = load_data_fashion_mnist(batch_size=128, resize=224)

lr = 0.01
net.initialize(force_reinit=True, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
loss = gloss.SoftmaxCrossEntropyLoss()
gb.train(train_iter, test_iter, net, loss, trainer,mx.cpu(), num_epochs=5)

