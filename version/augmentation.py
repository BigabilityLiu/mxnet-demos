import sys
sys.path.insert(0, '..')

import gluonbook as gb
import mxnet as mx
from mxnet import autograd, gluon, image, init, nd
from mxnet.gluon import data as gdata, loss as gloss, utils as gutils
import sys
from time import time
# import matplotlib.pyplot as plt

gb.set_matplotlib_formats('retina')

img = image.imread('../data/kitten.jpg')
# gb.plt.imshow(img.asnumpy())

def show_images(imgs, num_rows, num_cols, scala = 2):
    figsize = (num_cols * scala, num_rows * 2)
    _, axes = gb.plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j].asnumpy())
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes

def apply(img, aug, num_rows=2, num_cols=4, scala=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scala)


# apply(img, gdata.vision.transforms.RandomFlipLeftRight())
# apply(img, gdata.vision.transforms.RandomFlipTopBottom())
# apply(img, gdata.vision.transforms.RandomResizedCrop((200, 200), (0.1, 1), ratio=(0.5, 2)))
# apply(img, gdata.vision.transforms.RandomBrightness(0.5))
# apply(img, gdata.vision.transforms.RandomHue(0.5))
# apply(img, gdata.vision.transforms.RandomColorJitter(0.5))

show_images(gdata.vision.CIFAR10(train=True)[0:32][0], 4, 8, scala=0.8)


gb.plt.show()
