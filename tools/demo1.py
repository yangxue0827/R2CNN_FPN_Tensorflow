# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import sys
sys.path.append('../')
import random
import matplotlib.pyplot as plt
# from osgeo import gdal, gdalconst
import xml.dom.minidom
import time
from timeit import default_timer as timer
import cv2
from data.io import image_preprocess
from libs.networks.network_factory import get_network_byname
from libs.rpn import build_rpn
from libs.fast_rcnn import build_fast_rcnn1
from tools import restore_model
from libs.configs import cfgs
from help_utils.tools import *
from help_utils.help_utils import *
import argparse
from libs.box_utils import nms_rotate

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_file_paths_recursive(folder=None, file_ext=None):
    """ Get the absolute path of all files in given folder recursively
    :param folder:
    :param file_ext:
    :return:
    """
    file_list = []
    if folder is None:
        return file_list

    for dir_path, dir_names, file_names in os.walk(folder):
        for file_name in file_names:
            if file_ext is None:
                file_list.append(os.path.join(dir_path, file_name))
                continue
            if file_name.endswith(file_ext):
                file_list.append(os.path.join(dir_path, file_name))
    return file_list


def clip_obj_imgs(src_img, boxes, classes, scores, des_folder):
    """ Clip image by target information
    :param src_img:
    :param boxes:
    :param classes:
    :param scores:
    :param des_folder:
    :return:
    """
    box_num = len(boxes)
    ii = 0
    off_size = 20
    img_height = src_img.shape[0]
    img_width = src_img.shape[1]

    while ii < box_num:
        box = boxes[ii]
        xpos = max(box[0] - off_size, 0)
        ypos = max(box[1] - off_size, 0)
        clip_w = min(box[2]-box[0]+2*off_size, img_width-xpos)
        clip_h = min(box[3]-box[1]+2*off_size, img_height-ypos)
        img = np.zeros((clip_h, clip_w, 3))
        img[0:clip_h, 0:clip_w, :] = src_img[ypos:ypos+clip_h, xpos:xpos+clip_w, :]
        #plt.imshow(img)
        #plt.show()
        clip_path = os.path.join(des_folder, '%s-%d_%.3f.jpg' % (classes[ii], ii, scores[ii]))
        cv2.imwrite(clip_path, img)
        ii = ii + 1


def detect_img(file_paths, des_folder, det_th, h_len, w_len, h_overlap, w_overlap, show_res=False):
    with tf.Graph().as_default():

        img_plac = tf.placeholder(shape=[None, None, 3], dtype=tf.uint8)

        img_tensor = tf.cast(img_plac, tf.float32) - tf.constant([103.939, 116.779, 123.68])
        img_batch = image_preprocess.short_side_resize_for_inference_data(img_tensor,
                                                                          target_shortside_len=cfgs.SHORT_SIDE_LEN,
                                                                          is_resize=False)

        # ***********************************************************************************************
        # *                                         share net                                           *
        # ***********************************************************************************************
        _, share_net = get_network_byname(net_name=cfgs.NET_NAME,
                                          inputs=img_batch,
                                          num_classes=None,
                                          is_training=True,
                                          output_stride=None,
                                          global_pool=False,
                                          spatial_squeeze=False)
        # ***********************************************************************************************
        # *                                            RPN                                              *
        # ***********************************************************************************************
        rpn = build_rpn.RPN(net_name=cfgs.NET_NAME,
                            inputs=img_batch,
                            gtboxes_and_label=None,
                            is_training=False,
                            share_head=cfgs.SHARE_HEAD,
                            share_net=share_net,
                            stride=cfgs.STRIDE,
                            anchor_ratios=cfgs.ANCHOR_RATIOS,
                            anchor_scales=cfgs.ANCHOR_SCALES,
                            scale_factors=cfgs.SCALE_FACTORS,
                            base_anchor_size_list=cfgs.BASE_ANCHOR_SIZE_LIST,  # P2, P3, P4, P5, P6
                            level=cfgs.LEVEL,
                            top_k_nms=cfgs.RPN_TOP_K_NMS,
                            rpn_nms_iou_threshold=cfgs.RPN_NMS_IOU_THRESHOLD,
                            max_proposals_num=cfgs.MAX_PROPOSAL_NUM,
                            rpn_iou_positive_threshold=cfgs.RPN_IOU_POSITIVE_THRESHOLD,
                            rpn_iou_negative_threshold=cfgs.RPN_IOU_NEGATIVE_THRESHOLD,
                            rpn_mini_batch_size=cfgs.RPN_MINIBATCH_SIZE,
                            rpn_positives_ratio=cfgs.RPN_POSITIVE_RATE,
                            remove_outside_anchors=False,  # whether remove anchors outside
                            rpn_weight_decay=cfgs.WEIGHT_DECAY[cfgs.NET_NAME])

        # rpn predict proposals
        rpn_proposals_boxes, rpn_proposals_scores = rpn.rpn_proposals()  # rpn_score shape: [300, ]

        # ***********************************************************************************************
        # *                                         Fast RCNN                                           *
        # ***********************************************************************************************
        fast_rcnn = build_fast_rcnn1.FastRCNN(feature_pyramid=rpn.feature_pyramid,
                                              rpn_proposals_boxes=rpn_proposals_boxes,
                                              rpn_proposals_scores=rpn_proposals_scores,
                                              img_shape=tf.shape(img_batch),
                                              roi_size=cfgs.ROI_SIZE,
                                              roi_pool_kernel_size=cfgs.ROI_POOL_KERNEL_SIZE,
                                              scale_factors=cfgs.SCALE_FACTORS,
                                              gtboxes_and_label=None,
                                              gtboxes_and_label_minAreaRectangle=None,
                                              fast_rcnn_nms_iou_threshold=cfgs.FAST_RCNN_NMS_IOU_THRESHOLD,
                                              fast_rcnn_maximum_boxes_per_img=100,
                                              fast_rcnn_nms_max_boxes_per_class=cfgs.FAST_RCNN_NMS_MAX_BOXES_PER_CLASS,
                                              show_detections_score_threshold=det_th,
                                              # show detections which score >= 0.6
                                              num_classes=cfgs.CLASS_NUM,
                                              fast_rcnn_minibatch_size=cfgs.FAST_RCNN_MINIBATCH_SIZE,
                                              fast_rcnn_positives_ratio=cfgs.FAST_RCNN_POSITIVE_RATE,
                                              fast_rcnn_positives_iou_threshold=cfgs.FAST_RCNN_IOU_POSITIVE_THRESHOLD,
                                              # iou>0.5 is positive, iou<0.5 is negative
                                              use_dropout=cfgs.USE_DROPOUT,
                                              weight_decay=cfgs.WEIGHT_DECAY[cfgs.NET_NAME],
                                              is_training=False,
                                              level=cfgs.LEVEL)

        fast_rcnn_decode_boxes, fast_rcnn_score, num_of_objects, detection_category, \
        fast_rcnn_decode_boxes_rotate, fast_rcnn_score_rotate, num_of_objects_rotate, detection_category_rotate = \
            fast_rcnn.fast_rcnn_predict()

        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )

        restorer, restore_ckpt = restore_model.get_restorer()

        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(init_op)
            if not restorer is None:
                restorer.restore(sess, restore_ckpt)
                print('restore model')

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)

            for img_path in file_paths:
                start = timer()
                img = cv2.imread(img_path)

                box_res = []
                label_res = []
                score_res = []
                box_res_rotate = []
                label_res_rotate = []
                score_res_rotate = []

                imgH = img.shape[0]
                imgW = img.shape[1]
                for hh in range(0, imgH, h_len - h_overlap):
                    h_size = min(h_len, imgH - hh)
                    if h_size < 10:
                        break
                    for ww in range(0, imgW, w_len - w_overlap):
                        w_size = min(w_len, imgW - ww)
                        if w_size < 10:
                            break

                        src_img = img[hh:(hh + h_size), ww:(ww + w_size), :]

                        boxes, labels, scores = sess.run([fast_rcnn_decode_boxes, detection_category, fast_rcnn_score],
                                                         feed_dict={img_plac: src_img})

                        boxes_rotate, labels_rotate, scores_rotate = sess.run([fast_rcnn_decode_boxes_rotate,
                                                                               detection_category_rotate,
                                                                               fast_rcnn_score_rotate],
                                                                              feed_dict={img_plac: src_img})

                        if len(boxes) > 0:
                            for ii in range(len(boxes)):
                                box = boxes[ii]
                                box[0] = box[0] + hh
                                box[1] = box[1] + ww
                                box[2] = box[2] + hh
                                box[3] = box[3] + ww
                                box_res.append(box)
                                label_res.append(labels[ii])
                                score_res.append(scores[ii])
                        if len(boxes_rotate) > 0:
                            for ii in range(len(boxes_rotate)):
                                box_rotate = boxes_rotate[ii]
                                box_rotate[0] = box_rotate[0] + hh
                                box_rotate[1] = box_rotate[1] + ww
                                box_res_rotate.append(box_rotate)
                                label_res_rotate.append(labels_rotate[ii])
                                score_res_rotate.append(scores_rotate[ii])

                # inx = nms_rotate.nms_rotate_cpu(boxes=np.array(box_res_rotate), scores=np.array(score_res_rotate),
                #                                 iou_threshold=0.5, max_output_size=100)
                # box_res_rotate = np.array(box_res_rotate)[inx]
                # score_res_rotate = np.array(score_res_rotate)[inx]
                # label_res_rotate = np.array(label_res_rotate)[inx]

                time_elapsed = timer() - start
                print("{} detection time : {:.4f} sec".format(img_path.split('/')[-1].split('.')[0], time_elapsed))

                mkdir(des_folder)
                img_np = draw_box_cv(np.array(img, np.float32) - np.array([103.939, 116.779, 123.68]),
                                     boxes=np.array(box_res),
                                     labels=np.array(label_res),
                                     scores=np.array(score_res))
                img_np_rotate = draw_rotate_box_cv(np.array(img, np.float32) - np.array([103.939, 116.779, 123.68]),
                                                   boxes=np.array(box_res_rotate),
                                                   labels=np.array(label_res_rotate),
                                                   scores=np.array(score_res_rotate))
                cv2.imwrite(des_folder + '/{}_horizontal_fpn.jpg'.format(img_path.split('/')[-1].split('.')[0]), img_np)
                cv2.imwrite(des_folder + '/{}_rotate_fpn.jpg'.format(img_path.split('/')[-1].split('.')[0]), img_np_rotate)

            coord.request_stop()
            coord.join(threads)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--src_folder', dest='src_folder',
                        help='images path',
                        default=None, type=str)
    parser.add_argument('--des_folder', dest='des_folder',
                        help='output path',
                        default=None, type=str)
    parser.add_argument('--det_th', dest='det_th',
                        help='detection threshold',
                        default=0.7,
                        type=float)
    parser.add_argument('--h_len', dest='h_len',
                        help='image height',
                        default=600, type=int)
    parser.add_argument('--w_len', dest='w_len',
                        help='image width',
                        default=1000, type=int)
    parser.add_argument('--h_overlap', dest='h_overlap',
                        help='height overlap',
                        default=0, type=int)
    parser.add_argument('--w_overlap', dest='w_overlap',
                        help='width overlap',
                        default=0, type=int)
    parser.add_argument('--image_ext', dest='image_ext',
                        help='image format',
                        default='.tif', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print('Called with args:')
    print(args)
    file_paths = get_file_paths_recursive(args.src_folder, args.image_ext)

    detect_img(file_paths, args.des_folder, args.det_th, args.h_len, args.w_len,
               args.h_overlap, args.w_overlap, False)

