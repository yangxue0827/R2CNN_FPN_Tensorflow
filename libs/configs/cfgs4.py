# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os

"""
v4-dense_feature_pyramid, v5-feature_pyramid
Test(快速训练):
v4
*********horizontal eval*********
水平， 2000张  3000 nms
R: 0.803773584906
P: 0.963236661407
mAP: 0.776797872684
F: 0.876309784923

***********rotate eval***********
旋转， 2000张  3000 nms
R: 0.769496855346
P: 0.89145996997
mAP: 0.690753310464
F: 0.826000571602


v5
*********horizontal eval*********
水平， 2000张  3000 nms
R: 0.81320754717
P: 0.962663777256
mAP: 0.785794751386
F: 0.881646590363
***********rotate eval***********
旋转， 2000张  3000 nms
R: 0.786163522013
P: 0.885101755072
mAP: 0.700995138344
F: 0.832704086715


水平标准：
*********horizontal eval*********
水平， 2000张  3000 nms
R: 0.81320754717
P: 0.962663777256
mAP: 0.785794751386
F: 0.881646590363
***********rotate eval***********
旋转， 2000张  3000 nms
R: 0.825471698113
P: 0.933186075468
mAP: 0.773740173667
F: 0.876030238451


"""




# root path
ROO_PATH = os.path.abspath('/yangxue/FPN_v18')

# pretrain weights path
MODEL_PATH = ROO_PATH + '/output/model'
SUMMARY_PATH = ROO_PATH + '/output/summary'

TEST_SAVE_PATH = ROO_PATH + '/tools/test_result'
INFERENCE_IMAGE_PATH = ROO_PATH + '/tools/inference_image'
INFERENCE_SAVE_PATH = ROO_PATH + '/tools/inference_result'

NET_NAME = 'resnet_v1_101'
VERSION = 'v5'
CLASS_NUM = 1
LEVEL = ['P2', 'P3', 'P4', 'P5', 'P6']
BASE_ANCHOR_SIZE_LIST = [32, 64, 128, 256, 512]
STRIDE = [4, 8, 16, 32, 64]
ANCHOR_SCALES = [1.]
ANCHOR_RATIOS = [1 / 3., 1., 3.0]
SCALE_FACTORS = [10., 10., 5., 5., 5.]
OUTPUT_STRIDE = 16
SHORT_SIDE_LEN = 600
DATASET_NAME = 'ship'

BATCH_SIZE = 1
WEIGHT_DECAY = {'vggnet16': 0.0005, 'resnet_v1_50': 0.0001, 'resnet_v1_101': 0.0001}
EPSILON = 1e-5
MOMENTUM = 0.9
MAX_ITERATION = 40000
GPU_GROUP = "1"

# rpn
RPN_NMS_IOU_THRESHOLD = 0.7
MAX_PROPOSAL_NUM = 300
RPN_IOU_POSITIVE_THRESHOLD = 0.7
RPN_IOU_NEGATIVE_THRESHOLD = 0.3
RPN_MINIBATCH_SIZE = 256
RPN_POSITIVE_RATE = 0.5
IS_FILTER_OUTSIDE_BOXES = True
RPN_TOP_K_NMS = 3000
FEATURE_PYRAMID_MODE = 0  # {0: 'feature_pyramid', 1: 'dense_feature_pyramid'}

# fast rcnn
FAST_RCNN_MODE = 'build_fast_rcnn1'
ROI_SIZE = 14
ROI_POOL_KERNEL_SIZE = 2
USE_DROPOUT = False
KEEP_PROB = 0.5
FAST_RCNN_NMS_IOU_THRESHOLD = 0.15
FAST_RCNN_NMS_MAX_BOXES_PER_CLASS = 20
FINAL_SCORE_THRESHOLD = 0.9
FAST_RCNN_IOU_POSITIVE_THRESHOLD = 0.5
FAST_RCNN_MINIBATCH_SIZE = 128
FAST_RCNN_POSITIVE_RATE = 0.25
