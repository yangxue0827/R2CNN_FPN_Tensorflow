# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def enum_scales(base_anchor, anchor_scales, name='enum_scales'):

    '''
    :param base_anchor: [y_center, x_center, h, w]
    :param anchor_scales: different scales, like [0.5, 1., 2.0]
    :return: return base anchors in different scales.
            Example:[[0, 0, 128, 128],[0, 0, 256, 256],[0, 0, 512, 512]]
    '''
    with tf.variable_scope(name):
        anchor_scales = tf.reshape(anchor_scales, [-1, 1])

        return base_anchor * anchor_scales


def enum_ratios(anchors, anchor_ratios, name='enum_ratios'):

    '''
    :param anchors: base anchors in different scales
    :param anchor_ratios:  ratio = h / w
    :return: base anchors in different scales and ratios
    '''

    with tf.variable_scope(name):
        _, _, hs, ws = tf.unstack(anchors, axis=1)  # for base anchor, w == h
        sqrt_ratios = tf.sqrt(anchor_ratios)
        sqrt_ratios = tf.expand_dims(sqrt_ratios, axis=1)
        ws = tf.reshape(ws / sqrt_ratios, [-1])
        hs = tf.reshape(hs * sqrt_ratios, [-1])
        # assert tf.shape(ws) == tf.shape(hs), 'h shape is not equal w shape'

        num_anchors_per_location = tf.shape(ws)[0]

        return tf.transpose(tf.stack([tf.zeros([num_anchors_per_location, ]),
                                      tf.zeros([num_anchors_per_location,]),
                                      ws, hs]))


def make_anchors(base_anchor_size, anchor_scales, anchor_ratios, featuremaps_height,
                 featuremaps_width, stride, name='make_anchors'):

    '''
    :param base_anchor_size: base anchor size in different scales
    :param anchor_scales: anchor scales
    :param anchor_ratios: anchor ratios
    :param featuremaps_width: width of featuremaps
    :param featuremaps_height: height of featuremaps
    :return: anchors of shape [w * h * len(anchor_scales) * len(anchor_ratios), 4]
    '''

    with tf.variable_scope(name):
        # [y_center, x_center, h, w]
        base_anchor = tf.constant([0, 0, base_anchor_size, base_anchor_size], dtype=tf.float32)
        base_anchors = enum_ratios(enum_scales(base_anchor, anchor_scales), anchor_ratios)

        _, _, ws, hs = tf.unstack(base_anchors, axis=1)

        x_centers = tf.range(tf.cast(featuremaps_width, tf.float32), dtype=tf.float32) * stride
        y_centers = tf.range(tf.cast(featuremaps_height, tf.float32), dtype=tf.float32) * stride

        x_centers, y_centers = tf.meshgrid(x_centers, y_centers)

        ws, x_centers = tf.meshgrid(ws, x_centers)
        hs, y_centers = tf.meshgrid(hs, y_centers)

        box_centers = tf.stack([y_centers, x_centers], axis=2)
        box_centers = tf.reshape(box_centers, [-1, 2])

        box_sizes = tf.stack([hs, ws], axis=2)
        box_sizes = tf.reshape(box_sizes, [-1, 2])
        final_anchors = tf.concat([box_centers - 0.5*box_sizes, box_centers+0.5*box_sizes], axis=1)
        return final_anchors

if __name__ == '__main__':
    base_anchor = tf.constant([256], dtype=tf.float32)
    anchor_scales = tf.constant([1.0], dtype=tf.float32)
    anchor_ratios = tf.constant([0.5, 1.0, 2.0], dtype=tf.float32)
    # print(enum_scales(base_anchor, anchor_scales))
    sess = tf.Session()
    # print(sess.run(enum_ratios(enum_scales(base_anchor, anchor_scales), anchor_ratios)))
    anchors = make_anchors(256, anchor_scales, anchor_ratios,
                           featuremaps_height=38,
                           featuremaps_width=50, stride=16)

    _anchors = sess.run(anchors)
    print(_anchors)

