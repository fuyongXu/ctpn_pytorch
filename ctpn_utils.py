"""
Author  : Xu fuyong
Time    : created by 2019/6/28 17:03

"""
import numpy as np
import cv2
from config import *

# INTER_AREA:基于区域像素关系的一种重采样或者插值方式.该方法是图像抽取的首选方法,
# 它可以产生更少的波纹,但是当图像放大时,它的效果与INTER_NEAREST效果相似.https://zhuanlan.zhihu.com/p/38493205
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # 初始化要调整大小的图像的尺寸并且获取图像尺寸
    dim = None
    (h ,w) = image.shape[: 2]
    # 如果传入的参数width和height是None,则返回原图
    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized

def gen_anchor(featuresize, scale):
    """
    从feature map [h×W][9][4] 生成基础base anchor
    将[H×W][9][4]reshape为[H×W×9][4]
    """
    heights = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]  # 除于0.7得到，固定width
    widths = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16]

    # 产生 k =9 种anchor size(h, w)
    heights = np.array(heights).reshape(len(heights), 1)
    widths = np.array(widths).reshape(len(widths), 1)

    base_anchor = np.array([0, 0, 15, 15])

    # center x,y
    xt = (base_anchor[0] + base_anchor[2]) * 0.5
    yt = (base_anchor[1] + base_anchor[3]) * 0.5

    # x1 y1 x2 y2
    x1 = xt - widths * 0.5
    y1 = yt - heights * 0.5
    x2 = xt + widths * 0.5
    y2 = yt + heights * 0.5
    base_anchor = np.hstack((x1, y1, x2, y2))

    h, w = featuresize
    shift_x = np.arange(0, w) * scale
    shift_y = np.arange(0, h) * scale

    anchor = []
    for i in shift_y:
        for j in shift_x:
            anchor.append(base_anchor + [j, i, j, i])
    return np.array(anchor).reshape((-1, 4))

def cal_iou(box1, box1_area, boxes2, boxes2_area):
    """
    box1 [x1,y1,x2,y2]
    boxes[Msample,x1,y1,x2,y2]
    """
    x1 = np.maximum(box1[0], boxes2[:, 0])
    x2 = np.maximum(box1[2], boxes2[:, 2])
    y1 = np.maximum(box1[1], boxes2[:, 1])
    y2 = np.maximum(box1[3], boxes2[:, 3])

    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    iou = intersection / (box1_area + boxes2_area[:] - intersection)
    return iou

def cal_overlap(boxes1, boxes2):
    """
    boxes1 [Nsample,x1,y1,x2,y2] anchor
    boxes2[Msample,x1,y1,x2,y2] grouth-box
    """
    area1 = (boxes1[:, 0] - boxes1[:, 2]) * (boxes1[:, 1] - boxes1[:, 3])
    area2 = (boxes2[:, 0] - boxes2[:, 2]) * (boxes2[:, 1] - boxes2[:, 3])

    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))

    # 计算boxes1和boxes2(GT box)的intersection
    for i in range(boxes1.shape[0]):
        overlaps[i][:] = cal_iou(boxes1[i], area1[i], boxes2, area2)
    return overlaps

def bbox_transform(anchors, gtboxes):
    """
    计算相对预测垂直坐标Vc,Vh相对于 anchor的bounding box 位置
    """
    regr = np.zeros((anchors.shape[0], 2))
    Cy = (gtboxes[:, 1] + gtboxes[:, 3]) * 0.5
    Cya = (anchors[:, 1] + anchors[:, 3]) * 0.5         #表示论文中cya
    h = gtboxes[:, 3] - gtboxes[:, 1] + 1.0
    ha = gtboxes[:, 3] - gtboxes[:, 1] + 1.0

    Vc = (Cy - Cya) / ha
    Vh = np.log(h / ha)

    return np.vstack((Vc, Vh)).transpose()

def bbox_transfor_inv(anchor, regr):
    """
    返回预测的bbox
    """
    Cya = (anchor[: 1] + anchor[:, 3]) * 0.5
    ha = anchor[:, 3] - anchor[:, 1] + 1

    Vcx = regr[0, :, 0]
    Vhx = regr[0, :, 1]

    Cyx = Vcx * ha + Cya
    hx = np.exp(Vhx) * ha
    xt = (anchor[:, 0] + anchor[:, 2]) * 0.5

    x1 = xt - 16 * 0.5
    y1 = Cyx - hx * 0.5
    x2 = xt + 16 * 0.5
    y2 = Cyx + hx * 0.5
    bbox = np.vstack((x1, y1, x2, y2)).transpose()

    return bbox

# 裁剪box
def clip_box(bbox, im_shape):
    # x1 >= 0
    bbox[:, 0] = np.maximum(np.minimum(bbox[:, 0], im_shape[1] - 1), 0)  # 宽度方向
    # y1 >= 0
    bbox[:, 1] = np.maximum(np.minimum(bbox[:, 1], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    bbox[:, 2] = np.maximum(np.minimum(bbox[:, 2], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    bbox[:, 3] = np.maximum(np.minimum(bbox[:, 3], im_shape[0] - 1), 0)
    return bbox

def filter_bbox(bbox, minsize):
    ws = bbox[:, 2] - bbox[:, 0] + 1
    hs = bbox[:, 3] - bbox[:, 1] + 1
    keep = np.where((ws >= minsize) & (hs >= minsize))[0]
    return keep

def cal_rpn(imgsize, featuresize, scale, gtboxes):
    imgh, imgw = imgsize

    # gen base anchor
    base_anchor = gen_anchor(featuresize, scale)

    # calculate iou
    overlaps = cal_overlap(base_anchor, gtboxes)

    # init labels -1 don't care 0 is negative 1 is positive
    labels = np.empty(base_anchor.shape[0])
    labels.fill(-1)

    # for each GT box corresponds to an anchor which has highest IOU
    gt_argmax_overlaps = overlaps.argmax(axis=0)

    # the anchor with the highest IOU overlap with a GT box
    anchor_argmax_overlaps = overlaps.argmax(axis=1)
    anchor_max_overlaps = overlaps[range(overlaps.shape[0]), anchor_argmax_overlaps]

    # IOU > IOU_POSITIVE
    labels[anchor_max_overlaps > IOU_POSITIVE] = 1
    # IOU < IOU_NEGATIVE
    labels[anchor_max_overlaps > IOU_NEGATIVE] = 0
    # ensure that every GT box has at least one positive RPN region
    labels[gt_argmax_overlaps] = 1

    # only keep anchors inside the image
    outside_anchor = np.where(
        (base_anchor[:, 0] < 0)|
        (base_anchor[:, 1] < 0)|
        (base_anchor[:, 2] >= imgw)|
        (base_anchor[:, 3] <= imgh)
    )[0]
    labels[outside_anchor] = -1

    # subsample positive labels ,if greater than RPN_POSITIVE_NUM(default 128)
    fg_index = np.where(labels == 1)[0]
    if(len(fg_index) > RPN_POSITIVE_NUM):
        labels[np.random.choice(fg_index, len(fg_index) - RPN_POSITIVE_NUM, replace=False)] = -1

    # subsample positive labels
    bg_index = np.where(labels == 0)[0]
    num_bg = RPN_TOTAL_NUM - np.sum(labels == 1)
    if(len(bg_index) > num_bg):
        labels[np.random.choice(bg_index, len(bg_index) - num_bg, replace=False)] = -1

    # calculate bbox targets, debug here
    bbox_targets = bbox_transform(base_anchor, gtboxes[anchor_argmax_overlaps, :])
    # bbox_targets = []
    return [labels, bbox_targets], base_anchor

def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # [::-1]是反向排列

    keep = []
    while order.size() > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.maximum(x2[i], y2[order[1:]])
        yy2 = np.maximum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        over = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(over <= thresh)[0]
        order = order[inds + 1]
    return keep