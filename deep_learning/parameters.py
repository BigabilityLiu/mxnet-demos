from mxnet import init, nd
from mxnet.gluon import nn

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
print(net[0].weight)
net.initialize()

x = nd.random.uniform(shape=(20, 2))
y = net(x)


print(y)
print(net[0].params['dense0_weight'])
print(net[1].weight)
print(net[0].weight.data()[0])

net.initialize(init=init.Normal(0.01), force_reinit=True)
print(net[0].weight.data()[0])

net[0].initialize(init=init.Normal(1), force_reinit=True)
print(net[0].weight.data()[0])


class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = nd.random.uniform(low=-10, high=10, shape=data.shape)
        data *= data.abs() >= 5


net.initialize(MyInit(), force_reinit=True)
print(net[0].weight.data()[0])

net[0].weight.set_data(net[0].weight.data() + 1)
print(net[0].weight.data()[0])


net = nn.Sequential()
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

x = nd.random.uniform(shape=(2,20))
net(x)

print(net[1].weight.data()[0] == net[2].weight.data()[0])