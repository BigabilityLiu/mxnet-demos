import numpy as np
# y = 0.2x - 0.3

true_w = 0.2
true_b = - 0.3

w = 0.1
b = 0.1
learning_rate = 0.01

X = np.random.uniform(-10, 10, 10)
Y = X * true_w + true_b


def net(x):
    return x * w + b

def loss(x, Y):
    y_hat = net(x)
    print(Y, y_hat)
    print('Y = %f y_hat = %f, w = %f, b = %f' % (Y, y_hat, w, b))
    # return ((Y - y_hat) ** 2) / 2
    return Y - y_hat

def sgd(x, Y):
    delta = loss(x, Y)
    l = delta * learning_rate
    global w, b
    w += l * x
    b += l

def one_train(x, Y):
    sgd(x, Y)

for i in range(100):
    for j in range(len(X)):
        one_train(X[j], Y[j])

print(w, b)