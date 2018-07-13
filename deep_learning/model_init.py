from mxnet import nd
from mxnet.gluon import nn


class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.hidden = nn.Dense(256, 'relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))


class MySequential(nn.Block):
    def __init__(self, **kwargs):
        super(MySequential, self).__init__(**kwargs)

    def add(self, block):
        self._children[block.name] = block

    def forward(self, x):
        for block in self._children.values():
            x = block(x)
        return x


class FancyMLP(nn.Block):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        self.rand_weight = self.params.get_constant(
            'rand_weight', nd.random.uniform(shape=(8, 8))
        )
        self.dense = nn.Dense(8, activation='relu')

    def forward(self, X):
        x = self.dense(X)

        x = nd.relu(nd.dot(x, self.rand_weight.data()) + 1)
        x = self.dense(x)

        while x.norm().asscalar() > 1:
            x /= 2
        if x.norm().asscalar() < 0.8:
            x *= 10
        return x.sum()


class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)

        self.net = nn.Sequential()
        self.net.add(
            nn.Dense(64, 'relu'),
            nn.Dense(32, 'relu')
        )
        self.dense = nn.Dense(16, 'relu')

    def forward(self, x):
        return self.dense(self.net(x))


x = nd.random.uniform(shape=(2, 10))
net = nn.Sequential()
net.add(
    NestMLP(), nn.Dense(8), FancyMLP()
)
net.initialize()
print(net(x))
