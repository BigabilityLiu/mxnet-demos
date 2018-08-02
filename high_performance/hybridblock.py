from mxnet import nd
from mxnet.gluon import nn

class HybridNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(HybridNet, self).__init__(**kwargs)
        self.fc1 = nn.Dense(10)
        self.fc2 = nn.Dense(2)

    def hybrid_forward(self, F, x):
        # print(x.asnumpy())
        print('F:', F)
        x = F.relu(self.fc1(x))
        print('x:', x)
        return self.fc2(x)

net = HybridNet()

X = nd.random.uniform(shape=(10,))
print('X:', X)
net.initialize()
net.hybridize()
print(net(X))
print(net(X))