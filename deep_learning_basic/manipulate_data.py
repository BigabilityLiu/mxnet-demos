from mxnet import nd, autograd
import numpy as np
# print(caffe.__version__)

x = nd.arange(12).reshape((4,3))
print(x)
print(x[:])
print(x[1:])
print(x[:3])
#
# x.attach_grad()
# with autograd.record():
#     y = 2 * x # nd.dot(x.T, x)
#     z = y * x
#     z.backward()
#     print('x.grad: ',x.grad)
#     print(x.grad == 4 * x)

# def f(a):
#     b = a * 2
#     while b.norm().asscalar() < 1000:
#         b = b * 2
#     if b.sum().asscalar() > 0:
#         c = b
#     else:
#         c = 100 * b
#     return c
#
# a = nd.array([-1])
# print(a)
# a.attach_grad()
# with autograd.record():
#     c = f(a)
# c.backward()
# print(a.grad)
# print(a.grad == c / a)

