#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# This code is based on
# https://github.com/ultralytics/yolov5/blob/master/utils/dataloaders.py
import copy
import math
import random
import torch
import cv2
import numpy as np
names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]
dycache_copypaste = []
import albumentations as A

def albumentation(im, labels,size=640):
    T = [
        A.RandomResizedCrop(height=size, width=size, scale=(0.8, 1.0), ratio=(0.9, 1.11), p=0.0),
        A.Blur(p=0.01),
        A.MedianBlur(p=0.01),
        A.ToGray(p=0.01),
        A.CLAHE(p=0.01),
        A.RandomBrightnessContrast(p=0.0),
        A.RandomGamma(p=0.0),
        A.ImageCompression(quality_lower=75, p=0.0)]  # transforms
    transform = A.Compose(T, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    new = transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
    im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])

    return im,labels
def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    '''HSV color-space augmentation.'''
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32, return_int=False):
    '''Resize and pad image while meeting stride-multiple constraints.'''
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    if not return_int:
        return im, r, (dw, dh)
    else:
        return im, r, (left, top)


def mixup(im, labels, im2, labels2):
    '''Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf.'''
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    # print(len(labels))
    # print(len(labels2))
    nl = len(labels2)
    nl1 = len(labels)
    # if nl and nl1:
    labels = np.concatenate((labels, labels2), 0)
    # if not nl:
    #     labels = labels
    # if not nl1:
    #     labels = labels2

    return im, labels


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    '''Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio.'''
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


def random_affine(img, labels=(), segments =(),degrees=10, translate=.1, scale=.1, shear=10,
                  new_shape=(640, 640),mask_refine = False):
    '''Applies Random affine transformation.'''
    n = len(labels)
    height, width = new_shape

    M, s = get_transform_matrix(img.shape[:2], (height, width), degrees, scale, shear, translate)
    if (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Transform label coordinates
    # Transform label coordinates
    if n:
        perspective = False
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))
        if use_segments and mask_refine:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)
        else:
            xy = np.ones((n * 4, 3))
            xy[:, :2] = labels[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = xy[:, :2].reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=labels[:, 1:5].T * s, box2=new.T, area_thr=0.1)
        labels = labels[i]
        labels[:, 1:5] = new[i]

    return img, labels


def get_transform_matrix(img_shape, new_shape, degrees, scale, shear, translate):
    new_height, new_width = new_shape
    # Center
    C = np.eye(3)
    C[0, 2] = -img_shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img_shape[0] / 2  # y translation (pixels)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * new_width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * new_height  # y transla ion (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT
    return M, s


def mosaic_augmentation(img_size, imgs, hs, ws, labels, hyp,segments = None):
    '''Applies Mosaic augmentation.'''
    assert len(imgs) == 4, "Mosaic augmentation of current version only supports 4 images."

    labels4 = []
    segment4 = []
    s = img_size
    yc, xc = (int(random.uniform(s//2, 3*s//2)) for _ in range(2))  # mosaic center x, y
    for i in range(len(imgs)):
        # Load image
        img, h, w = imgs[i], hs[i], ws[i]
        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels_per_img = labels[i].copy()
        segment = segments[i][0].copy()
        if labels_per_img.size:
            boxes = np.copy(labels_per_img[:, 1:])
            boxes[:, 0] = w * (labels_per_img[:, 1] - labels_per_img[:, 3] / 2) + padw  # top left x
            boxes[:, 1] = h * (labels_per_img[:, 2] - labels_per_img[:, 4] / 2) + padh  # top left y
            boxes[:, 2] = w * (labels_per_img[:, 1] + labels_per_img[:, 3] / 2) + padw  # bottom right x
            boxes[:, 3] = h * (labels_per_img[:, 2] + labels_per_img[:, 4] / 2) + padh  # bottom right y
            labels_per_img[:, 1:] = boxes
            segment = [xyn2xy(x, w, h, padw, padh) for x in segment]

        labels4.append(labels_per_img)
        segment4.extend(segment)

    # Concat/clip labels
    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    for x in (labels4[:, 1:], *segment4):
        np.clip(x, 0, 2 * s, out=x)
    if hyp['copy_paste']:
        img4, labels4, segment4 = copy_paste(img4, labels4, segment4, p=hyp['copy_paste'])
    #picture(img4, labels4)


    # Augment
    img4, labels4 = random_affine(img4, labels4, segment4,
                                  degrees=hyp['degrees'],
                                  translate=hyp['translate'],
                                  scale=hyp['scale'],
                                  shear=hyp['shear'],
                                  new_shape=(img_size, img_size),
                                  mask_refine = hyp['mask_refine'])

    return img4, labels4
def picture(img,labels,img_name = "exp1"):
    for label in labels:
        x_min = int(label[1])
        y_min = int(label[2])
        x_max = int(label[3])
        y_max = int(label[4])
        name = int(label[0])
        name = names[name]
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 0), 2)
        cv2.putText(img, name, (x_min, y_min), 1, 1, (0, 0, 0))
    #cv2.imwrite(img_name+".png", img)
    cv2.imshow("myImage",img)
    cv2.waitKey(0)
def RandomeResizeCrop(img,labels,img_size = 1280,scale = [0.1, 2]):

    img_size = random.randint(img_size*scale[0],img_size*scale[1])
    T = [
        A.Resize(width=img_size,height=img_size),
        A.RandomCrop(height=640, width=640, p=1.0)]
    transformed_1 = A.RandomResizedCrop(height=640, width=640, scale=scale, ratio=(0.9, 1.11), p=1.0)
    transformed_2 = A.Compose(transformed_1, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    new = transformed_2(image=img, bboxes=labels[:, 1:], class_labels=labels[:, 0])
    im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])
    return im,labels
def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # Convert normalized segments into pixel segments, shape (n,2)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * x[..., 0] + padw  # top left x
    y[..., 1] = h * x[..., 1] + padh  # top left y
    return y
def copy_paste(im, labels, segments, p=0.5):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    n = len(segments)
    if p and n:
        h, w, c = im.shape  # height, width, channels
        im_new = np.zeros(im.shape, np.uint8)

        # calculate ioa first then select indexes randomly
        boxes = np.stack([w - labels[:, 3], labels[:, 2], w - labels[:, 1], labels[:, 4]], axis=-1)  # (n, 4)
        ioa = bbox_ioa(boxes, labels[:, 1:5])  # intersection over area
        indexes = np.nonzero((ioa < 0.30).all(1))[0]  # (N, )
        n = len(indexes)
        for j in random.sample(list(indexes), k=round(p * n)):
            l, box, s = labels[j], boxes[j], segments[j]
            labels = np.concatenate((labels, [[l[0], *box]]), 0)
            segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
            cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (1, 1, 1), cv2.FILLED)

        result = cv2.flip(im, 1)  # augment segments (flip left-right)
        i = cv2.flip(im_new, 1).astype(bool)
        im[i] = result[i]  # cv2.imwrite('debug.jpg', im)  # debug

    return im, labels, segments
def dy_cache_copy_paste(im, labels, segments, p=0.5):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    n = len(segments)
    if p and n:
        h, w, c = im.shape  # height, width, channels
        im_new = np.zeros(im.shape, np.uint8)

        # calculate ioa first then select indexes randomly
        boxes = np.stack([w - labels[:, 3], labels[:, 2], w - labels[:, 1], labels[:, 4]], axis=-1)  # (n, 4)
        ioa = bbox_ioa(boxes, labels[:, 1:5])  # intersection over area
        indexes = np.nonzero((ioa < 0.30).all(1))[0]  # (N, )
        n = len(indexes)
        for j in random.sample(list(indexes), k=round(p * n)):
            img_information = [labels[j], boxes[j], segments[j]]
            dycache_copypaste.append(copy.deepcopy(img_information))
        for number in random.sample(list(indexes), k=round(p * n)):
            copy_index = random.randint(0, len(dycache_copypaste) - 1)
            l, box, s = dycache_copypaste[copy_index][0], dycache_copypaste[copy_index][1], dycache_copypaste[copy_index][2]
            labels = np.concatenate((labels, [[l[0], *box]]), 0)
            segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
            cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (1, 1, 1), cv2.FILLED)
            if len(dycache_copypaste) > 40:
                pop_index = random.randint(0, len(dycache_copypaste) - 1)
                dycache_copypaste.pop(pop_index)


        result = cv2.flip(im, 1)  # augment segments (flip left-right)
        i = cv2.flip(im_new, 1).astype(bool)
        im[i] = result[i]  # cv2.imwrite('debug.jpg', im)  # debug

    return im, labels, segments
def bbox_ioa(box1, box2, eps=1e-7):
    """Returns the intersection over box2 area given box1, box2. Boxes are x1y1x2y2
    box1:       np.array of shape(nx4)
    box2:       np.array of shape(mx4)
    returns:    np.array of shape(nxm)
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * \
                 (np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area
def resample_segments(segments, n=1000):
    # Up-sample an (n,2) segment
    for i, s in enumerate(segments):
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # segment xy
    return segments
def segment2box(segment, width=640, height=640):
    # Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y, = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))  # xyxy