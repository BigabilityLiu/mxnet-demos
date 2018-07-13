from mxnet import nd
from mxnet.gluon import nn

x = nd.ones(3)
nd.save('x', [x])

x2 = nd.load('x')
print(x2)
print('---1---')
y = nd.zeros(4)
nd.save('xy', [x, y])

xy = nd.load('xy')
print(xy)
print('---2---')

mydict = {'x': x, 'y': y}
nd.save('../mydict', mydict)
mydict2 = nd.load('../mydict')
print(mydict2)
print('---3---')

class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.hidden = nn.Dense(256, 'relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))


x = nd.random.uniform(shape=(2, 10))

net = MLP()
net.initialize()
y = net(x)

print(y)
print('---4---')
filename = 'myMLP.params'

net.save_params(filename)

net2 = MLP()
net2.load_params(filename)
y2 = net2(x)

print(y == y2)