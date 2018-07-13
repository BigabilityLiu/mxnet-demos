import math
from mxnet import nd, autograd
import numpy as np
import time

data = nd.arange(12).reshape((3,4))
print(data)

print(data[1][0])
