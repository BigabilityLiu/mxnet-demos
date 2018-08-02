import gluonbook as gb
from mxnet import contrib, gluon, image, nd
import numpy as np
import matplotlib.pyplot as plt


def bbox_to_rect(bbox, color):
    return gb.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)

def show_bboxes(axes, bboxes, labels=None, colors=None):
    def _make_list(obj, default_value=None):
        if obj is None:
            return default_value
        elif not isinstance(obj, (list, tuple)):
            return [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'k'])

    for i, bboxe in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = bbox_to_rect(bboxe.asnumpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))


np.set_printoptions(2)

img = image.imread('../data/kitten.jpg').asnumpy()
fig = gb.plt.imshow(img)

h, w = img.shape[0:2]
x = nd.random.uniform(shape=(1, 3, h, w))
y = contrib.nd.MultiBoxPrior(x, sizes=[0.75, 0.5, 0.25], ratios = [1, 2, 0.5])

boxes = y.reshape((h, w, 5, 4))

bbox_scale = nd.array((w, h, w, h))
# show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2', 's=0.75, r=0.5'])

ground_truth = nd.array([[0, 0.1, 0.08, 0.52, 0.92],
                         [1, 0.55, 0.2, 0.9, 0.88]])
anchors = nd.array([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.5, 0.6],
                    [0.5, 0.25, 0.85, 0.85], [0.57, 0.45, 0.85, 0.85]])

# show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
# show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3'] );
#
# out = contrib.nd.MultiBoxTarget(anchors.expand_dims(axis=0),
#                                 ground_truth.expand_dims(axis=0),
#                                 nd.zeros((1, 3, 4)))
# print(out)
anchors = nd.array([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                    [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
loc_preds = nd.array([0] * anchors.size)
cls_probs = nd.array([[0] * 4,  # 是背景的概率。
                      [0.9, 0.8, 0.7, 0.1],  # 是狗的概率 。
                      [0.1, 0.2, 0.3, 0.9]])  # 是猫的概率。
# show_bboxes(fig.axes, anchors * bbox_scale,['dog=0.9', 'dog=0.8', 'dog=0.7',' cat=0.9'])
ret = contrib.ndarray.MultiBoxDetection(cls_probs.expand_dims(axis=0),
                                        loc_preds.expand_dims(axis=0),
                                        anchors.expand_dims(axis=0),
                                        nms_threshold=.5)
print(ret)
for i in ret[0].asnumpy():
    if i[0] == -1:
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    show_bboxes(fig.axes, [nd.array(i[2:]) * bbox_scale], label)

gb.plt.show()


