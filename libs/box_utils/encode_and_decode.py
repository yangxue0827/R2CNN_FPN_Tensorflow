# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

import numpy as np
import math


def decode_boxes(encode_boxes, reference_boxes, scale_factors=None, name='decode'):
    '''

    :param encode_boxes:[N, 4]
    :param reference_boxes: [N, 4] .
    :param scale_factors: use for scale
    in the first stage, reference_boxes  are anchors
    in the second stage, reference boxes are proposals(decode) produced by rpn stage
    :return:decode boxes [N, 4]
    '''

    with tf.variable_scope(name):
        t_ycenter, t_xcenter, t_h, t_w = tf.unstack(encode_boxes, axis=1)
        if scale_factors:
            t_xcenter /= scale_factors[0]
            t_ycenter /= scale_factors[1]
            t_w /= scale_factors[2]
            t_h /= scale_factors[3]

        reference_ymin, reference_xmin, reference_ymax, reference_xmax = tf.unstack(reference_boxes, axis=1)

        reference_xcenter = (reference_xmin + reference_xmax) / 2.
        reference_ycenter = (reference_ymin + reference_ymax) / 2.
        reference_w = reference_xmax - reference_xmin
        reference_h = reference_ymax - reference_ymin

        predict_xcenter = t_xcenter * reference_w + reference_xcenter
        predict_ycenter = t_ycenter * reference_h + reference_ycenter
        predict_w = tf.exp(t_w) * reference_w
        predict_h = tf.exp(t_h) * reference_h

        predict_xmin = predict_xcenter - predict_w / 2.
        predict_xmax = predict_xcenter + predict_w / 2.
        predict_ymin = predict_ycenter - predict_h / 2.
        predict_ymax = predict_ycenter + predict_h / 2.

        return tf.transpose(tf.stack([predict_ymin, predict_xmin,
                                      predict_ymax, predict_xmax]))


def decode_boxes_rotate(encode_boxes, reference_boxes, scale_factors=None, name='decode'):
    '''

    :param encode_boxes:[N, 5]
    :param reference_boxes: [N, 5] .
    :param scale_factors: use for scale
    in the rpn stage, reference_boxes are anchors
    in the fast_rcnn stage, reference boxes are proposals(decode) produced by rpn stage
    :return:decode boxes [N, 5]
    '''

    with tf.variable_scope(name):
        t_ycenter, t_xcenter, t_h, t_w, t_theta = tf.unstack(encode_boxes, axis=1)
        if scale_factors:
            t_xcenter /= scale_factors[0]
            t_ycenter /= scale_factors[1]
            t_w /= scale_factors[2]
            t_h /= scale_factors[3]
            t_theta /= scale_factors[4]

        reference_ymin, reference_xmin, reference_ymax, reference_xmax = tf.unstack(reference_boxes, axis=1)
        reference_x_center = (reference_xmin + reference_xmax) / 2.
        reference_y_center = (reference_ymin + reference_ymax) / 2.
        reference_w = reference_xmax - reference_xmin
        reference_h = reference_ymax - reference_ymin
        reference_theta = tf.ones(tf.shape(reference_xmin)) * -90

        predict_x_center = t_xcenter * reference_w + reference_x_center
        predict_y_center = t_ycenter * reference_h + reference_y_center
        predict_w = tf.exp(t_w) * reference_w
        predict_h = tf.exp(t_h) * reference_h

        predict_theta = t_theta * 180 / math.pi + reference_theta

        mask1 = tf.less(predict_theta, -90)
        mask2 = tf.greater_equal(predict_theta, -180)
        mask7 = tf.less(predict_theta, -180)
        mask8 = tf.greater_equal(predict_theta, -270)

        mask3 = tf.greater_equal(predict_theta, 0)
        mask4 = tf.less(predict_theta, 90)
        mask5 = tf.greater_equal(predict_theta, 90)
        mask6 = tf.less(predict_theta, 180)

        # to keep range in [-90, 0)
        # [-180, -90)
        convert_mask = tf.logical_and(mask1, mask2)
        remain_mask = tf.logical_not(convert_mask)
        predict_theta += tf.cast(convert_mask, tf.float32) * 90.

        remain_h = tf.cast(remain_mask, tf.float32) * predict_h
        remain_w = tf.cast(remain_mask, tf.float32) * predict_w
        convert_h = tf.cast(convert_mask, tf.float32) * predict_h
        convert_w = tf.cast(convert_mask, tf.float32) * predict_w

        predict_h = remain_h + convert_w
        predict_w = remain_w + convert_h

        # [-270, -180)
        cond4 = tf.cast(tf.logical_and(mask7, mask8), tf.float32) * 180.
        predict_theta += cond4

        # [0, 90)
        # cond2 = tf.cast(tf.logical_and(mask3, mask4), tf.float32) * 90.
        # predict_theta -= cond2

        convert_mask1 = tf.logical_and(mask3, mask4)
        remain_mask1 = tf.logical_not(convert_mask1)
        predict_theta -= tf.cast(convert_mask1, tf.float32) * 90.

        remain_h = tf.cast(remain_mask1, tf.float32) * predict_h
        remain_w = tf.cast(remain_mask1, tf.float32) * predict_w
        convert_h = tf.cast(convert_mask1, tf.float32) * predict_h
        convert_w = tf.cast(convert_mask1, tf.float32) * predict_w

        predict_h = remain_h + convert_w
        predict_w = remain_w + convert_h

        # [90, 180)
        cond3 = tf.cast(tf.logical_and(mask5, mask6), tf.float32) * 180.
        predict_theta -= cond3

        decode_boxes = tf.transpose(tf.stack([predict_y_center, predict_x_center,
                                              predict_h, predict_w, predict_theta]))

        return decode_boxes


def encode_boxes(unencode_boxes, reference_boxes, scale_factors=None, name='encode'):
    '''

    :param unencode_boxes: [batch_size*H*W*num_anchors_per_location, 4]
    :param reference_boxes: [H*W*num_anchors_per_location, 4]
    :return: encode_boxes [-1, 4]
    '''

    with tf.variable_scope(name):
        ymin, xmin, ymax, xmax = tf.unstack(unencode_boxes, axis=1)

        reference_ymin, reference_xmin, reference_ymax, reference_xmax = tf.unstack(reference_boxes, axis=1)

        x_center = (xmin + xmax) / 2.
        y_center = (ymin + ymax) / 2.
        w = xmax - xmin
        h = ymax - ymin

        reference_xcenter = (reference_xmin + reference_xmax) / 2.
        reference_ycenter = (reference_ymin + reference_ymax) / 2.
        reference_w = reference_xmax - reference_xmin
        reference_h = reference_ymax - reference_ymin

        reference_w += 1e-8
        reference_h += 1e-8
        w += 1e-8
        h += 1e-8  # to avoid NaN in division and log below

        t_xcenter = (x_center - reference_xcenter) / reference_w
        t_ycenter = (y_center - reference_ycenter) / reference_h
        t_w = tf.log(w / reference_w)
        t_h = tf.log(h / reference_h)

        if scale_factors:
            t_xcenter *= scale_factors[0]
            t_ycenter *= scale_factors[1]
            t_w *= scale_factors[2]
            t_h *= scale_factors[3]

        return tf.transpose(tf.stack([t_ycenter, t_xcenter, t_h, t_w]))


def encode_boxes_rotate(unencode_boxes, reference_boxes, scale_factors=None, name='encode'):
    '''
    :param unencode_boxes: [batch_size*H*W*num_anchors_per_location, 5]
    :param reference_boxes: [H*W*num_anchors_per_location, 5]
    :return: encode_boxes [-1, 5]
    '''

    with tf.variable_scope(name):
        y_center, x_center, h, w, theta = tf.unstack(unencode_boxes, axis=1)

        reference_ymin, reference_xmin, reference_ymax, reference_xmax = tf.unstack(reference_boxes, axis=1)

        reference_x_center = (reference_xmin + reference_xmax) / 2.
        reference_y_center = (reference_ymin + reference_ymax) / 2.
        # here maybe have logical error, reference_w and reference_h should exchange,
        # but it doesn't seem to affect the result.
        reference_w = reference_xmax - reference_xmin
        reference_h = reference_ymax - reference_ymin
        reference_theta = tf.ones(tf.shape(reference_xmin)) * -90

        reference_w += 1e-8
        reference_h += 1e-8
        w += 1e-8
        h += 1e-8  # to avoid NaN in division and log below

        t_xcenter = (x_center - reference_x_center) / reference_w
        t_ycenter = (y_center - reference_y_center) / reference_h
        t_w = tf.log(w / reference_w)
        t_h = tf.log(h / reference_h)
        t_theta = (theta - reference_theta) * math.pi / 180

        if scale_factors:
            t_xcenter *= scale_factors[0]
            t_ycenter *= scale_factors[1]
            t_w *= scale_factors[2]
            t_h *= scale_factors[3]
            t_theta *= scale_factors[4]

        return tf.transpose(tf.stack([t_ycenter, t_xcenter, t_h, t_w, t_theta]))