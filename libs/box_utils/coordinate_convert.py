# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np


def forward_convert(coordinate, with_label=True):
    """
    :param coordinate: format [y_c, x_c, h, w, theta]
    :return: format [y1, x1, y2, x2, y3, x3, y4, x4]
    """
    boxes = []
    if with_label:
        for rect in coordinate:
            box = cv2.boxPoints(((rect[1], rect[0]), (rect[3], rect[2]), rect[4]))
            box = np.reshape(box, [-1, ])
            boxes.append([box[1], box[0], box[3], box[2], box[5], box[4], box[7], box[6], rect[5]])
    else:
        for rect in coordinate:
            box = cv2.boxPoints(((rect[1], rect[0]), (rect[3], rect[2]), rect[4]))
            box = np.reshape(box, [-1, ])
            boxes.append([box[1], box[0], box[3], box[2], box[5], box[4], box[7], box[6]])

    return np.array(boxes, dtype=np.float32)


def back_forward_convert(coordinate, with_label=True):
    """
    :param coordinate: format [x1, y1, x2, y2, x3, y3, x4, y4, (label)] 
    :param with_label: default True
    :return: format [y_c, x_c, h, w, theta, (label)]
    """

    boxes = []
    if with_label:
        for rect in coordinate:
            box = np.int0(rect[:-1])
            box = box.reshape([4, 2])
            rect1 = cv2.minAreaRect(box)

            x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
            boxes.append([y, x, h, w, theta, rect[-1]])

    else:
        for rect in coordinate:
            box = np.int0(rect)
            box = box.reshape([4, 2])
            rect1 = cv2.minAreaRect(box)

            x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
            boxes.append([y, x, h, w, theta])

    return np.array(boxes, dtype=np.float32)


if __name__ == '__main__':
    coord = np.array([[150, 150, 50, 100, -90, 1],
                      [150, 150, 100, 50, -90, 1],
                      [150, 150, 50, 100, -45, 1],
                      [150, 150, 100, 50, -45, 1]])

    coord1 = np.array([[150, 150, 100, 50, 0],
                      [150, 150, 100, 50, -90],
                      [150, 150, 100, 50, 45],
                      [150, 150, 100, 50, -45]])

    coord2 = forward_convert(coord)
    # coord3 = forward_convert(coord1, mode=-1)
    print(coord2)
    # print(coord3-coord2)
    # coord_label = np.array([[167., 203., 96., 132., 132., 96., 203., 167., 1.]])
    #
    # coord4 = back_forward_convert(coord_label, mode=1)
    # coord5 = back_forward_convert(coord_label)

    # print(coord4)
    # print(coord5)

    # coord3 = coordinate_present_convert(coord, -1)
    # print(coord3)
    # coord4 = coordinate_present_convert(coord3, mode=1)
    # print(coord4)

