from mxnet import nd, gluon, init
from mxnet.gluon import nn


class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()


layer = CenteredLayer()
print(layer(nd.array([1, 2, 3, 4, 5])))

net = nn.Sequential()
net.add(nn.Dense(128))
net.add(nn.Dense(10))
net.add(CenteredLayer())

net.initialize(init=init.Normal(sigma=10))
y = net(nd.random.uniform(shape=(4, 8)))
print(y.mean())
print(net[0].params['dense0_weight'])

print('---0---')
params = gluon.ParameterDict()
print(params)
params.get('param2', shape=(2, 3))
print(params)

print('---1---')
class MyDense(nn.Block):
    def __init__(self, units, in_units, **kwargs):
        super(MyDense, self).__init__(**kwargs)

        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = nd.dot(x, self.weight.data()) + self.bias.data()
        return nd.relu(linear)


dense = MyDense(units=5, in_units=10)
print(dense.params)

dense.initialize()
print(dense(nd.random.uniform(shape=(2, 10))))

print('---3---')
net = nn.Sequential()
net.add(MyDense(32, in_units=64))
net.add(MyDense(2, in_units=32))
net.initialize()
print(net(nd.random.uniform(shape=(2, 64))))
