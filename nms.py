import numpy as np
from numpy import array
import cv2

TEST = 0
IMG_SIZE = 640

# w1,h1 w2,h2 w3,h3
if(TEST == 0):
    if(IMG_SIZE == 320):
        # 320x320
        anchor_grid0 = (21.17188, 30.53126, 36.71875, 66.375, 97.31252, 79.31252)
        anchor_grid1 = (64.56250, 148.125, 128.5, 187.62505, 257, 205.625)
        box_grid0 = (20, 20)
        box_grid1 = (10, 10)
        IMG_ORG_COL = 1280#960#640
        IMG_ORG_ROW = 720#480
        IMG_COL = 320
        IMG_ROW = 320
    else:
        # 640x320
        anchor_grid0 = (5, 6, 9, 7, 11, 12)
        anchor_grid1 = (21, 18, 43, 31, 98, 72)
        box_grid0 = (40, 20)
        box_grid1 = (20, 10)
        IMG_ORG_COL = 1280#960#640
        IMG_ORG_ROW = 720#480
        IMG_COL = 640
        IMG_ROW = 320
else:
    anchor_grid0 = (5, 6, 9, 7, 11, 12)
    anchor_grid1 = (21, 18, 43, 31, 98, 72)
    box_grid0 = (4, 2)
    box_grid1 = (2, 1)
    IMG_COL = 64
    IMG_ROW = 32

conf_threshold = 0.30
iou_threshold = 0.45    
    
# video resize
gain_x, gain_y = IMG_COL/IMG_ORG_COL, IMG_ROW/IMG_ORG_ROW # gain = old/new
# pad_y, pad_x = (IMG_ROW-IMG_ORG_ROW*gain) / 2, (IMG_COL-IMG_ORG_COL*gain) / 2 # wh padding
# print("gain=", gain)
# print("pad x = ", pad_x)
# print("pad y = ", pad_y)

grid_len0 = box_grid0[0]*box_grid0[1]
grid_len1 = box_grid1[0]*box_grid1[1]

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def box_restore(boxes, box_grid, anchor_grid):
    scale = [0, 0]
    scale[0] = IMG_COL/box_grid[0]
    scale[1] = IMG_ROW/box_grid[1]
    # 12channels = (xywh)*3 anchors
    cx = np.arange(boxes.shape[0]) % box_grid[0]
    cy = np.arange(boxes.shape[0]) // box_grid[0]
    cx = cx.reshape(cx.shape[0], 1)
    cy = cy.reshape(cy.shape[0], 1)
    # print(cx)
    # print(cy)
    x_sigmod = sigmoid(boxes[..., 0])
    y_sigmod = sigmoid(boxes[..., 1])

#     x = (x_sigmod + cx) * scale[0]
#     y = (y_sigmod + cy) * scale[1]
#     w = np.exp(boxes[..., 2]) * anchor_grid[0:6:2]
#     h = np.exp(boxes[..., 3]) * anchor_grid[1:6:2]
    x = (x_sigmod*2-0.5 + cx) * scale[0]
    y = (y_sigmod*2-0.5 + cy) * scale[1]
    w = (sigmoid(boxes[..., 2])*2)**2 * anchor_grid[0:6:2]
    h = (sigmoid(boxes[..., 3])*2)**2 * anchor_grid[1:6:2]
    
    # xywh to xyxy   
    restored_boxes = np.zeros(boxes.shape)
    restored_boxes[..., 0] = (x - w/2).astype(np.int32)  # xmin
    restored_boxes[..., 1] = (y - h/2).astype(np.int32)  # ymin
    restored_boxes[..., 2] = (x + w/2).astype(np.int32)  # xmax
    restored_boxes[..., 3] = (y + h/2).astype(np.int32)  # ymax

    return restored_boxes


def box_area(boxes: array):
    """
    :param boxes: [N, 4]
    :return: [N]
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(box1: array, box2: array):
    """
    :param box1: [N, 4]
    :param box2: [M, 4]
    :return: [N, M]
    """
    area1 = box_area(box1)  # N
    area2 = box_area(box2)  # M
    # broadcasting, 两个数组各维度大小 从后往前对比一致， 或者 有一维度值为1；
    lt = np.maximum(box1[:, np.newaxis, :2], box2[:, :2])
    rb = np.minimum(box1[:, np.newaxis, 2:], box2[:, 2:])
    wh = rb - lt
    wh = np.maximum(0, wh)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]
    iou = inter / (area1[:, np.newaxis] + area2 - inter)
    return iou  # NxM


def numpy_nms(boxes: array, scores: array, classes: array, conf_threshold: float, iou_threshold: float):

    idxs = scores.argsort()  # 按分数 降序排列的索引 [N]
    keep = []
    while idxs.size > 0:  # 统计数组中元素的个数
        max_score_index = idxs[-1]
        max_score_box = boxes[max_score_index][None, :]
        keep.append(max_score_index)

        if idxs.size == 1:
            break
        idxs = idxs[:-1]  # 将得分最大框 从索引中删除； 剩余索引对应的框 和 得分最大框 计算IoU；
        other_boxes = boxes[idxs]  # [?, 4]
        ious = box_iou(max_score_box, other_boxes)  # 一个框和其余框比较 1XM
        idxs = idxs[ious[0] <= iou_threshold]

    keep = np.array(keep)  # Tensor
    keep = keep[scores[keep] > conf_threshold]  # 最后处理阈值

    boxes = boxes[keep]  # 保留下来的框
    scores = scores[keep]  # soft nms抑制后得分
    classes = classes[keep]
    return boxes, scores, classes


