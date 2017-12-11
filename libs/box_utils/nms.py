# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def non_maximal_suppression(boxes, scores, iou_threshold, max_output_size, name='non_maximal_suppression'):
    with tf.variable_scope(name):
        nms_index = tf.image.non_max_suppression(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            name=name
        )
        return nms_index