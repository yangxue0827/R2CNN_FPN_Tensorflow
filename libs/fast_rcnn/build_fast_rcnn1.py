# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from libs.box_utils import encode_and_decode
from libs.box_utils import boxes_utils
from libs.box_utils import iou
from libs.losses import losses
from help_utils.help_utils import print_tensors
from libs.box_utils import nms_rotate
import numpy as np
from libs.configs import cfgs

DEBUG = False


class FastRCNN(object):
    def __init__(self,
                 feature_pyramid, rpn_proposals_boxes, rpn_proposals_scores,
                 img_shape,
                 roi_size,
                 scale_factors,
                 roi_pool_kernel_size,  # roi size = initial_crop_size / max_pool_kernel size
                 gtboxes_and_label,
                 gtboxes_and_label_minAreaRectangle,  # [M, 5]
                 fast_rcnn_nms_iou_threshold,
                 fast_rcnn_maximum_boxes_per_img,
                 fast_rcnn_nms_max_boxes_per_class,
                 show_detections_score_threshold,  # show box scores larger than this threshold

                 num_classes,  # exclude background
                 fast_rcnn_minibatch_size,
                 fast_rcnn_positives_ratio,
                 fast_rcnn_positives_iou_threshold,
                 use_dropout,
                 is_training,
                 weight_decay,
                 level):

        self.feature_pyramid = feature_pyramid
        self.rpn_proposals_boxes = rpn_proposals_boxes  # [N, 4]
        self.rpn_proposals_scores = rpn_proposals_scores

        self.img_shape = img_shape
        self.roi_size = roi_size
        self.roi_pool_kernel_size = roi_pool_kernel_size
        self.level = level
        self.min_level = int(level[0][1])
        self.max_level = min(int(level[-1][1]), 5)

        self.fast_rcnn_nms_iou_threshold = fast_rcnn_nms_iou_threshold
        self.fast_rcnn_nms_max_boxes_per_class = fast_rcnn_nms_max_boxes_per_class
        self.fast_rcnn_maximum_boxes_per_img = fast_rcnn_maximum_boxes_per_img
        self.show_detections_score_threshold = show_detections_score_threshold

        self.scale_factors = scale_factors
        # larger than 0.5 is positive, others are negative
        self.fast_rcnn_positives_iou_threshold = fast_rcnn_positives_iou_threshold

        self.fast_rcnn_minibatch_size = fast_rcnn_minibatch_size
        self.fast_rcnn_positives_ratio = fast_rcnn_positives_ratio

        self.gtboxes_and_label = gtboxes_and_label
        self.gtboxes_and_label_minAreaRectangle = gtboxes_and_label_minAreaRectangle
        self.num_classes = num_classes
        self.use_dropout = use_dropout
        self.is_training = is_training
        self.weight_decay = weight_decay

        self.fast_rcnn_all_level_rois, self.fast_rcnn_all_level_proposals = self.get_rois()
        self.fast_rcnn_encode_boxes, self.fast_rcnn_scores, \
        self.fast_rcnn_encode_boxes_rotate, self.fast_rcnn_scores_rotate = self.fast_rcnn_net()

    def assign_level(self):
        with tf.name_scope('assign_levels'):
            ymin, xmin, ymax, xmax = tf.unstack(self.rpn_proposals_boxes, axis=1)

            w = tf.maximum(xmax - xmin, 0.)  # avoid w is negative
            h = tf.maximum(ymax - ymin, 0.)  # avoid h is negative

            levels = tf.round(4. + tf.log(tf.sqrt(w*h + 1e-8)/224.0) / tf.log(2.))

            levels = tf.maximum(levels, tf.ones_like(levels) * (np.float32(self.min_level)))  # level minimum is 2
            levels = tf.minimum(levels, tf.ones_like(levels) * (np.float32(self.max_level)))  # level maximum is 5

            return tf.cast(levels, tf.int32)

    def get_rois(self):
        '''
           1)get roi from feature map
           2)roi align or roi pooling. Here is roi align
           :return:
           all_level_rois: [N, 7, 7, C]
           all_level_proposals : [N, 4]
           all_level_proposals is matched with all_level_rois
        '''
        levels = self.assign_level()

        all_level_roi_list = []
        all_level_proposal_list = []
        if DEBUG:
            print_tensors(levels, 'levels')
        with tf.variable_scope('fast_rcnn_roi'):
            # P6 is not used by the Fast R-CNN detector.
            for i in range(self.min_level, self.max_level + 1):
                level_i_proposal_indices = tf.reshape(tf.where(tf.equal(levels, i)), [-1])
                level_i_proposals = tf.gather(self.rpn_proposals_boxes, level_i_proposal_indices)

                level_i_proposals = tf.cond(
                    tf.equal(tf.shape(level_i_proposals)[0], 0),
                    lambda: tf.constant([[0, 0, 0, 0]], dtype=tf.float32),
                    lambda: level_i_proposals
                )  # to avoid level_i_proposals batch is 0, or it will broken when gradient BP

                all_level_proposal_list.append(level_i_proposals)

                ymin, xmin, ymax, xmax = tf.unstack(level_i_proposals, axis=1)
                img_h, img_w = tf.cast(self.img_shape[1], tf.float32), tf.cast(self.img_shape[2], tf.float32)
                normalize_ymin = ymin / img_h
                normalize_xmin = xmin / img_w
                normalize_ymax = ymax / img_h
                normalize_xmax = xmax / img_w

                level_i_cropped_rois = tf.image.crop_and_resize(self.feature_pyramid['P%d' % i],
                                                                boxes=tf.transpose(tf.stack([normalize_ymin, normalize_xmin,
                                                                                             normalize_ymax, normalize_xmax])),
                                                                box_ind=tf.zeros(shape=[tf.shape(level_i_proposals)[0], ],
                                                                                 dtype=tf.int32),
                                                                crop_size=[self.roi_size, self.roi_size]
                                                                )
                level_i_rois = slim.max_pool2d(level_i_cropped_rois,
                                               [self.roi_pool_kernel_size, self.roi_pool_kernel_size],
                                               stride=self.roi_pool_kernel_size)
                all_level_roi_list.append(level_i_rois)

            all_level_rois = tf.concat(all_level_roi_list, axis=0)
            all_level_proposals = tf.concat(all_level_proposal_list, axis=0)
            return all_level_rois, all_level_proposals

    def fast_rcnn_net(self):

        with tf.variable_scope('fast_rcnn_net'):
            with slim.arg_scope([slim.fully_connected], weights_regularizer=slim.l2_regularizer(self.weight_decay)):

                flatten_rois_features = slim.flatten(self.fast_rcnn_all_level_rois)

                net = slim.fully_connected(flatten_rois_features, 1024, scope='fc_1')
                if self.use_dropout:
                    net = slim.dropout(net, keep_prob=0.5, is_training=self.is_training, scope='dropout')

                net = slim.fully_connected(net, 1024, scope='fc_2')

                fast_rcnn_scores = slim.fully_connected(net, self.num_classes + 1, activation_fn=None,
                                                          scope='classifier')

                fast_rcnn_encode_boxes = slim.fully_connected(net, self.num_classes * 4, activation_fn=None,
                                                                 scope='regressor')
            if DEBUG:
                print_tensors(fast_rcnn_encode_boxes, 'fast_rcnn_encode_bxes')

        with tf.variable_scope('fast_rcnn_net_rotate'):
            with slim.arg_scope([slim.fully_connected], weights_regularizer=slim.l2_regularizer(self.weight_decay)):

                flatten_rois_features_rotate = slim.flatten(self.fast_rcnn_all_level_rois)

                net_rotate = slim.fully_connected(flatten_rois_features_rotate, 1024, scope='fc_1')
                if self.use_dropout:
                    net_rotate = slim.dropout(net_rotate, keep_prob=0.5, is_training=self.is_training, scope='dropout')

                net_rotate = slim.fully_connected(net_rotate, 1024, scope='fc_2')

                fast_rcnn_scores_rotate = slim.fully_connected(net_rotate, self.num_classes + 1, activation_fn=None,
                                                               scope='classifier')

                fast_rcnn_encode_boxes_rotate = slim.fully_connected(net_rotate, self.num_classes * 5,
                                                                     activation_fn=None,
                                                                     scope='regressor')

            return fast_rcnn_encode_boxes, fast_rcnn_scores, fast_rcnn_encode_boxes_rotate, fast_rcnn_scores_rotate

    def fast_rcnn_find_positive_negative_samples(self, reference_boxes):
        '''
        when training, we should know each reference box's label and gtbox,
        in second stage
        iou >= 0.5 is object
        iou < 0.5 is background
        :param reference_boxes: [num_of_input_boxes, 4]
        :return:
        reference_boxes_mattached_gtboxes: each reference box mattched gtbox, shape: [num_of_input_boxes, 4]
        object_mask: indicate box(a row) weather is a object, 1 is object, 0 is background
        category_label: indicate box's class, one hot encoding. shape: [num_of_input_boxes, num_classes+1]
        '''

        with tf.variable_scope('fast_rcnn_find_positive_negative_samples'):
            gtboxes = tf.cast(
                tf.reshape(self.gtboxes_and_label_minAreaRectangle[:, :-1], [-1, 4]), tf.float32)  # [M, 4]

            gtboxes_rotate = tf.cast(
                tf.reshape(self.gtboxes_and_label[:, :-1], [-1, 5]), tf.float32)  # [M, 5]

            ious = iou.iou_calculate(reference_boxes, gtboxes)  # [N, M]

            matchs = tf.cast(tf.argmax(ious, axis=1), tf.int32)  # [N, ]
            max_iou_each_row = tf.reduce_max(ious, axis=1)
            # [N, ]
            positives = tf.cast(tf.greater_equal(max_iou_each_row, self.fast_rcnn_positives_iou_threshold), tf.int32)

            reference_boxes_mattached_gtboxes = tf.gather(gtboxes, matchs)  # [N, 4]
            reference_boxes_mattached_gtboxes_rotate = tf.gather(gtboxes_rotate, matchs)

            object_mask = tf.cast(positives, tf.float32)  # [N, ]

            label = tf.gather(self.gtboxes_and_label_minAreaRectangle[:, -1], matchs)  # [N, ]
            label = tf.cast(label, tf.int32) * positives  # background is 0

            return reference_boxes_mattached_gtboxes, reference_boxes_mattached_gtboxes_rotate, object_mask, label

    def fast_rcnn_minibatch(self, reference_boxes):
        with tf.variable_scope('fast_rcnn_minibatch'):

            reference_boxes_mattached_gtboxes, reference_boxes_mattached_gtboxes_rotate, object_mask, label = \
                self.fast_rcnn_find_positive_negative_samples(reference_boxes)

            positive_indices = tf.reshape(tf.where(tf.not_equal(object_mask, 0.)), [-1])

            num_of_positives = tf.minimum(tf.shape(positive_indices)[0],
                                          tf.cast(self.fast_rcnn_minibatch_size*self.fast_rcnn_positives_ratio, tf.int32))

            positive_indices = tf.random_shuffle(positive_indices)
            positive_indices = tf.slice(positive_indices, begin=[0], size=[num_of_positives])

            negative_indices = tf.reshape(tf.where(tf.equal(object_mask, 0.)), [-1])
            num_of_negatives = tf.minimum(tf.shape(negative_indices)[0],
                                          self.fast_rcnn_minibatch_size - num_of_positives)

            negative_indices = tf.random_shuffle(negative_indices)
            negative_indices = tf.slice(negative_indices, begin=[0], size=[num_of_negatives])

            minibatch_indices = tf.concat([positive_indices, negative_indices], axis=0)
            minibatch_indices = tf.random_shuffle(minibatch_indices)

            minibatch_reference_boxes_mattached_gtboxes = tf.gather(reference_boxes_mattached_gtboxes,
                                                                    minibatch_indices)

            minibatch_reference_boxes_mattached_gtboxes_rotate \
                = tf.gather(reference_boxes_mattached_gtboxes_rotate, minibatch_indices)

            object_mask = tf.gather(object_mask, minibatch_indices)
            label = tf.gather(label, minibatch_indices)
            label_one_hot = tf.one_hot(label, self.num_classes + 1)

            return minibatch_indices, minibatch_reference_boxes_mattached_gtboxes, \
                   minibatch_reference_boxes_mattached_gtboxes_rotate, object_mask, label_one_hot

    def fast_rcnn_loss(self):
        with tf.variable_scope('fast_rcnn_loss'):
            minibatch_indices, minibatch_reference_boxes_mattached_gtboxes, \
            minibatch_reference_boxes_mattached_gtboxes_rotate, minibatch_object_mask, \
            minibatch_label_one_hot = self.fast_rcnn_minibatch(self.fast_rcnn_all_level_proposals)

            minibatch_reference_boxes = tf.gather(self.fast_rcnn_all_level_proposals, minibatch_indices)

            minibatch_encode_boxes = tf.gather(self.fast_rcnn_encode_boxes,
                                               minibatch_indices)  # [minibatch_size, num_classes*4]

            minibatch_encode_boxes_rotate = tf.gather(self.fast_rcnn_encode_boxes_rotate,
                                                      minibatch_indices)  # [minibatch_size, num_classes*5]

            minibatch_scores = tf.gather(self.fast_rcnn_scores, minibatch_indices)
            minibatch_scores_rotate = tf.gather(self.fast_rcnn_scores_rotate, minibatch_indices)

            # encode gtboxes
            minibatch_encode_gtboxes = \
                encode_and_decode.encode_boxes(
                    unencode_boxes=minibatch_reference_boxes_mattached_gtboxes,
                    reference_boxes=minibatch_reference_boxes,
                    scale_factors=self.scale_factors
                )

            minibatch_encode_gtboxes_rotate = encode_and_decode.encode_boxes_rotate(
                unencode_boxes=minibatch_reference_boxes_mattached_gtboxes_rotate,
                reference_boxes=minibatch_reference_boxes,
                scale_factors=self.scale_factors
            )

            # [minibatch_size, num_classes*4]
            minibatch_encode_gtboxes = tf.tile(minibatch_encode_gtboxes, [1, self.num_classes])
            # [minibatch_size, num_classes*5]
            minibatch_encode_gtboxes_rotate = tf.tile(minibatch_encode_gtboxes_rotate, [1, self.num_classes])

            class_weights_list = []
            category_list = tf.unstack(minibatch_label_one_hot, axis=1)
            for i in range(1, self.num_classes+1):
                tmp_class_weights = tf.ones(shape=[tf.shape(minibatch_encode_boxes)[0], 4], dtype=tf.float32)
                tmp_class_weights = tmp_class_weights * tf.expand_dims(category_list[i], axis=1)
                class_weights_list.append(tmp_class_weights)
            class_weights = tf.concat(class_weights_list, axis=1)  # [minibatch_size, num_classes*4]

            class_weights_list_rotate = []
            category_list_rotate = tf.unstack(minibatch_label_one_hot, axis=1)
            for i in range(1, self.num_classes + 1):
                tmp_class_weights_rotate = tf.ones(shape=[tf.shape(minibatch_encode_boxes_rotate)[0], 5], dtype=tf.float32)
                tmp_class_weights_rotate = tmp_class_weights_rotate * tf.expand_dims(category_list_rotate[i], axis=1)
                class_weights_list_rotate.append(tmp_class_weights_rotate)
            class_weights_rotate = tf.concat(class_weights_list_rotate, axis=1)  # [minibatch_size, num_classes*5]

            # loss
            with tf.variable_scope('fast_rcnn_classification_loss'):
                fast_rcnn_classification_loss = slim.losses.softmax_cross_entropy(logits=minibatch_scores,
                                                                                  onehot_labels=minibatch_label_one_hot)
            with tf.variable_scope('fast_rcnn_location_loss'):
                fast_rcnn_location_loss = losses.l1_smooth_losses(predict_boxes=minibatch_encode_boxes,
                                                                  gtboxes=minibatch_encode_gtboxes,
                                                                  object_weights=minibatch_object_mask,
                                                                  classes_weights=class_weights)
                slim.losses.add_loss(fast_rcnn_location_loss)

            with tf.variable_scope('fast_rcnn_classification_rotate_loss'):
                fast_rcnn_classification_rotate_loss = slim.losses.softmax_cross_entropy(logits=minibatch_scores_rotate,
                                                                                         onehot_labels=minibatch_label_one_hot)

            with tf.variable_scope('fast_rcnn_location_rotate_loss'):
                fast_rcnn_location_rotate_loss = losses.l1_smooth_losses(predict_boxes=minibatch_encode_boxes_rotate,
                                                                         gtboxes=minibatch_encode_gtboxes_rotate,
                                                                         object_weights=minibatch_object_mask,
                                                                         classes_weights=class_weights_rotate)
                slim.losses.add_loss(fast_rcnn_location_rotate_loss)

            return fast_rcnn_location_loss, fast_rcnn_classification_loss, \
                   fast_rcnn_location_rotate_loss, fast_rcnn_classification_rotate_loss

    def fast_rcnn_proposals(self, decode_boxes, scores):
        '''
        mutilclass NMS
        :param decode_boxes: [N, num_classes*4]
        :param scores: [N, num_classes+1]
        :return:
        detection_boxes : [-1, 4]
        scores : [-1, ]

        '''

        with tf.variable_scope('fast_rcnn_proposals'):
            category = tf.argmax(scores, axis=1)

            object_mask = tf.cast(tf.not_equal(category, 0), tf.float32)

            decode_boxes = decode_boxes * tf.expand_dims(object_mask, axis=1)  # make background box is [0 0 0 0]
            scores = scores * tf.expand_dims(object_mask, axis=1)

            decode_boxes = tf.reshape(decode_boxes, [-1, self.num_classes, 4])  # [N, num_classes, 4]

            decode_boxes_list = tf.unstack(decode_boxes, axis=1)
            score_list = tf.unstack(scores[:, 1:], axis=1)
            after_nms_boxes = []
            after_nms_scores = []
            category_list = []
            for per_class_decode_boxes, per_class_scores in zip(decode_boxes_list, score_list):

                valid_indices = boxes_utils.nms_boxes(per_class_decode_boxes, per_class_scores,
                                                      iou_threshold=self.fast_rcnn_nms_iou_threshold,
                                                      max_output_size=self.fast_rcnn_nms_max_boxes_per_class,
                                                      name='second_stage_NMS')

                after_nms_boxes.append(tf.gather(per_class_decode_boxes, valid_indices))
                after_nms_scores.append(tf.gather(per_class_scores, valid_indices))
                tmp_category = tf.gather(category, valid_indices)

                category_list.append(tmp_category)

            all_nms_boxes = tf.concat(after_nms_boxes, axis=0)
            all_nms_scores = tf.concat(after_nms_scores, axis=0)
            all_category = tf.concat(category_list, axis=0)

            all_nms_boxes = boxes_utils.clip_boxes_to_img_boundaries(all_nms_boxes,
                                                                     img_shape=self.img_shape)

            scores_large_than_threshold_indices = tf.reshape(tf.where(tf.greater(all_nms_scores,
                                                                                 self.show_detections_score_threshold)), [-1])

            all_nms_boxes = tf.gather(all_nms_boxes, scores_large_than_threshold_indices)
            all_nms_scores = tf.gather(all_nms_scores, scores_large_than_threshold_indices)
            all_category = tf.gather(all_category, scores_large_than_threshold_indices)

            return all_nms_boxes, all_nms_scores, tf.shape(all_nms_boxes)[0], all_category

    def fast_rcnn_proposals_rotate(self, decode_boxes, scores):
        '''
        mutilclass NMS
        :param decode_boxes: [N, num_classes*5]
        :param scores: [N, num_classes+1]
        :return:
        detection_boxes : [-1, 5]
        scores : [-1, ]

        '''

        with tf.variable_scope('fast_rcnn_proposals'):
            category = tf.argmax(scores, axis=1)

            object_mask = tf.cast(tf.not_equal(category, 0), tf.float32)

            decode_boxes = decode_boxes * tf.expand_dims(object_mask, axis=1)  # make background box is [0 0 0 0, 0]
            scores = scores * tf.expand_dims(object_mask, axis=1)

            decode_boxes = tf.reshape(decode_boxes, [-1, self.num_classes, 5])  # [N, num_classes, 5]

            decode_boxes_list = tf.unstack(decode_boxes, axis=1)
            score_list = tf.unstack(scores[:, 1:], axis=1)
            after_nms_boxes = []
            after_nms_scores = []
            category_list = []
            for per_class_decode_boxes, per_class_scores in zip(decode_boxes_list, score_list):

                valid_indices = nms_rotate.nms_rotate(decode_boxes=per_class_decode_boxes,
                                                      scores=per_class_scores,
                                                      iou_threshold=self.fast_rcnn_nms_iou_threshold,
                                                      max_output_size=self.fast_rcnn_nms_max_boxes_per_class,
                                                      use_angle_condition=False,
                                                      angle_threshold=15,
                                                      use_gpu=cfgs.ROTATE_NMS_USE_GPU)

                after_nms_boxes.append(tf.gather(per_class_decode_boxes, valid_indices))
                after_nms_scores.append(tf.gather(per_class_scores, valid_indices))
                tmp_category = tf.gather(category, valid_indices)

                category_list.append(tmp_category)

            all_nms_boxes = tf.concat(after_nms_boxes, axis=0)
            all_nms_scores = tf.concat(after_nms_scores, axis=0)
            all_category = tf.concat(category_list, axis=0)

            # all_nms_boxes = boxes_utils.clip_boxes_to_img_boundaries(all_nms_boxes,
            #                                                          img_shape=self.img_shape)

            scores_large_than_threshold_indices = \
                tf.reshape(tf.where(tf.greater(all_nms_scores, self.show_detections_score_threshold)), [-1])

            all_nms_boxes = tf.gather(all_nms_boxes, scores_large_than_threshold_indices)
            all_nms_scores = tf.gather(all_nms_scores, scores_large_than_threshold_indices)
            all_category = tf.gather(all_category, scores_large_than_threshold_indices)

            return all_nms_boxes, all_nms_scores, tf.shape(all_nms_boxes)[0], all_category

    def fast_rcnn_predict(self):

        with tf.variable_scope('fast_rcnn_predict'):
            fast_rcnn_softmax_scores = slim.softmax(self.fast_rcnn_scores)  # [-1, num_classes+1]
            fast_rcnn_softmax_scores_rotate = slim.softmax(self.fast_rcnn_scores_rotate)  # [-1, num_classes+1]

            fast_rcnn_encode_boxes = tf.reshape(self.fast_rcnn_encode_boxes, [-1, 4])
            fast_rcnn_encode_boxes_rotate = tf.reshape(self.fast_rcnn_encode_boxes_rotate, [-1, 5])

            reference_boxes = tf.tile(self.fast_rcnn_all_level_proposals, [1, self.num_classes])  # [N, 4*num_classes]
            reference_boxes = tf.reshape(reference_boxes, [-1, 4])   # [N*num_classes, 4]
            fast_rcnn_decode_boxes = encode_and_decode.decode_boxes(encode_boxes=fast_rcnn_encode_boxes,
                                                                    reference_boxes=reference_boxes,
                                                                    scale_factors=self.scale_factors)

            fast_rcnn_decode_boxes_rotate = \
                encode_and_decode.decode_boxes_rotate(encode_boxes=fast_rcnn_encode_boxes_rotate,
                                                      reference_boxes=reference_boxes,
                                                      scale_factors=self.scale_factors)

            fast_rcnn_decode_boxes = boxes_utils.clip_boxes_to_img_boundaries(fast_rcnn_decode_boxes,
                                                                              img_shape=self.img_shape)

            # mutilclass NMS
            fast_rcnn_decode_boxes = tf.reshape(fast_rcnn_decode_boxes, [-1, self.num_classes*4])
            fast_rcnn_decode_boxes_rotate = tf.reshape(fast_rcnn_decode_boxes_rotate, [-1, self.num_classes * 5])

            fast_rcnn_decode_boxes, fast_rcnn_score, num_of_objects, detection_category = \
                self.fast_rcnn_proposals(fast_rcnn_decode_boxes, scores=fast_rcnn_softmax_scores)

            fast_rcnn_decode_boxes_rotate, fast_rcnn_score_rotate, num_of_objects_rotate, detection_category_rotate = \
                self.fast_rcnn_proposals_rotate(fast_rcnn_decode_boxes_rotate, scores=fast_rcnn_softmax_scores_rotate)

            return fast_rcnn_decode_boxes, fast_rcnn_score, num_of_objects, detection_category,\
                   fast_rcnn_decode_boxes_rotate, fast_rcnn_score_rotate, num_of_objects_rotate, detection_category_rotate












