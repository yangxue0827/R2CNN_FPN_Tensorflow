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
from libs.fast_rcnn import build_fast_rcnn
from tools import restore_model
from libs.configs import cfgs
from help_utils.tools import *
from help_utils.help_utils import *
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def chw2hwc(img):
    """ Convert image data from [channel, height, weight] to [height, weight, channel],
    and the origianl image should be [channel, height, weight]
    :param img:
    :return:
    """
    res = np.swapaxes(img, 0, 2) # w,h,c
    res = np.swapaxes(res, 0, 1) # h,w,c
    return res


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


def visualize_detection(src_img, boxes, scores):
    """ visualize detections in one image
    :param src_img: numpy.array
    :param boxes: [[x1, y1, x2, y2]...], each row is one object
    :param class_names: class names, and each row is one object
    :param scores: score for each object (each row)
    :return:
    """
    plt.imshow(src_img)
    box_num = len(boxes)
    color = (1.0, 0.0, 0.0)
    for i in range(box_num):
        box = boxes[i]
        rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False,
                             edgecolor=color, linewidth=1.5)
        plt.gca().add_patch(rect)
        plt.gca().text(box[0], box[1] - 2, '{:.3f}'.format(scores[i]),
                       bbox=dict(facecolor=color, alpha=0.5), fontsize=6, color='white')
    plt.show()


# def obj_to_det_xml(img_path, box_res, label_res, score_res, det_xml_path):
#     """ Save detection results to det.xml
#     :param img_path:
#     :param box_res:
#     :param label_res:
#     :param score_res:
#     :param det_xml_path:
#     :return:
#     """
#     gdal.AllRegister()
#     ds = gdal.Open(img_path, gdalconst.GA_Update)
#     if ds is None:
#         print("Image %s open failed!" % img_path)
#         sys.exit()
#     proj_str = ds.GetProjection()
#     geoTf = ds.GetGeoTransform()
#
#     obj_n = len(box_res)
#
#     doc = xml.dom.minidom.Document()
#     root_node = doc.createElement("ImageInfo")
#     root_node.setAttribute("resolution", str(geoTf[1]))
#     root_node.setAttribute("imagingtime", time.strftime('%Y-%m-%dT%H:%M:%S'))
#     doc.appendChild(root_node)
#
#     BaseInfo_node = doc.createElement("BaseInfo")
#     BaseInfo_node.setAttribute("description", " ")
#     BaseInfo_node.setAttribute("ID", " ")
#     BaseInfo_node.setAttribute("name", "sewage")
#     root_node.appendChild(BaseInfo_node)
#
#     result_node = doc.createElement("result")
#     DetectNumber_node = doc.createElement("DetectNumber")
#     DetectNumber_value = doc.createTextNode(str(obj_n))
#     DetectNumber_node.appendChild(DetectNumber_value)
#     result_node.appendChild(DetectNumber_node)
#
#     for ii in range(obj_n):
#         box = box_res[ii]
#         xmin = geoTf[0] + geoTf[1] * box[1]
#         ymin = geoTf[3] + geoTf[5] * box[0]
#         xmax = geoTf[0] + geoTf[1] * box[3]
#         ymax = geoTf[3] + geoTf[5] * box[2]
#
#         DetectResult_node = doc.createElement("DetectResult")
#
#         ResultID_node = doc.createElement("ResultID")
#         ResultID_value = doc.createTextNode(str(ii))
#         ResultID_node.appendChild(ResultID_value)
#
#         Shape_node = doc.createElement("Shape")
#         Point1_node = doc.createElement("Point")
#         Point1_value = doc.createTextNode("%.6f, %.6f" % (xmin, ymin))
#         Point1_node.appendChild(Point1_value)
#
#         Point2_node = doc.createElement("Point")
#         Point2_value = doc.createTextNode("%.6f, %.6f" % (xmax, ymin))
#         Point2_node.appendChild(Point2_value)
#
#         Point3_node = doc.createElement("Point")
#         Point3_value = doc.createTextNode("%.6f, %.6f" % (xmax, ymax))
#         Point3_node.appendChild(Point3_value)
#
#         Point4_node = doc.createElement("Point")
#         Point4_value = doc.createTextNode("%.6f, %.6f" % (xmin, ymax))
#         Point4_node.appendChild(Point4_value)
#
#         Point5_node = doc.createElement("Point")
#         Point5_value = doc.createTextNode("%.6f, %.6f" % (xmin, ymin))
#         Point5_node.appendChild(Point5_value)
#
#         Shape_node.appendChild(Point1_node)
#         Shape_node.appendChild(Point2_node)
#         Shape_node.appendChild(Point3_node)
#         Shape_node.appendChild(Point4_node)
#         Shape_node.appendChild(Point5_node)
#
#         Location_node = doc.createElement("Location")
#         Location_value = doc.createTextNode("unknown")
#         Location_node.appendChild(Location_value)
#
#         CenterLonLat_node = doc.createElement("CenterLonLat")
#         CenterLonLat_value = doc.createTextNode("0.000000, 0.000000")
#         CenterLonLat_node.appendChild(CenterLonLat_value)
#
#         Length_node = doc.createElement("Length")
#         Length_value = doc.createTextNode("0")
#         Length_node.appendChild(Length_value)
#
#         Width_node = doc.createElement("Width")
#         Width_value = doc.createTextNode("0")
#         Width_node.appendChild(Width_value)
#
#         Area_node = doc.createElement("Area")
#         Area_value = doc.createTextNode("0")
#         Area_node.appendChild(Area_value)
#
#         Angle_node = doc.createElement("Angle")
#         Angle_value = doc.createTextNode("0")
#         Angle_node.appendChild(Angle_value)
#
#         Probability_node = doc.createElement("Probability")
#         Probability_value = doc.createTextNode("1.0")
#         Probability_node.appendChild(Probability_value)
#
#         ResultImagePath_node = doc.createElement("ResultImagePath")
#         ResultImagePath_value = doc.createTextNode(" ")
#         ResultImagePath_node.appendChild(ResultImagePath_value)
#
#         ValidationName_node = doc.createElement("ValidationName")
#         ValidationName_value = doc.createTextNode(" ")
#         ValidationName_node.appendChild(ValidationName_value)
#
#         PossibleResults_node = doc.createElement("PossibleResults")
#
#         Type_node = doc.createElement("Type")
#         Type_value = doc.createTextNode("%s" % label_res[ii])
#         Type_node.appendChild(Type_value)
#
#         Reliability_node = doc.createElement("Reliability")
#         Reliability_value = doc.createTextNode("%.3f" % score_res[ii])
#         Reliability_node.appendChild(Reliability_value)
#
#         PossibleResults_node.appendChild(Type_node)
#         PossibleResults_node.appendChild(Reliability_node)
#
#         DetectResult_node.appendChild(ResultID_node)
#         DetectResult_node.appendChild(Shape_node)
#         DetectResult_node.appendChild(Location_node)
#         DetectResult_node.appendChild(CenterLonLat_node)
#         DetectResult_node.appendChild(Length_node)
#         DetectResult_node.appendChild(Width_node)
#         DetectResult_node.appendChild(Area_node)
#         DetectResult_node.appendChild(Angle_node)
#         DetectResult_node.appendChild(Probability_node)
#         DetectResult_node.appendChild(ResultImagePath_node)
#         DetectResult_node.appendChild(ValidationName_node)
#         DetectResult_node.appendChild(PossibleResults_node)
#
#         result_node.appendChild(DetectResult_node)
#     root_node.appendChild(result_node)
#
#     with open(det_xml_path, "w+") as f:
#         f.write(doc.toprettyxml(indent="\t", newl="\n", encoding="utf-8"))


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


def detect_img(file_paths, des_folder, det_th, h_len, w_len, show_res=False):
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
        fast_rcnn = build_fast_rcnn.FastRCNN(feature_pyramid=rpn.feature_pyramid,
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

        fast_rcnn_decode_boxes, fast_rcnn_score, num_of_objects, detection_category = \
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
                # gdal.AllRegister()
                # ds = gdal.Open(img_path, gdalconst.GA_ReadOnly)
                # if ds is None:
                #     print("Image %s open failed!" % img_path)
                #     sys.exit()
                img = cv2.imread(img_path)

                box_res = []
                label_res = []
                score_res = []
                # imgH = ds.RasterYSize
                # imgW = ds.RasterXSize
                imgH = img.shape[0]
                imgW = img.shape[1]
                for hh in range(0, imgH, h_len):
                    h_size = min(h_len, imgH - hh)
                    if h_size < 10:
                        break
                    for ww in range(0, imgW, w_len):
                        w_size = min(w_len, imgW - ww)
                        if w_size < 10:
                            break

                        # src_img = ds.ReadAsArray(ww, hh, w_size, h_size)
                        src_img = img[hh:(hh + h_size), ww:(ww + w_size), :]
                        # if len(src_img.shape) == 2:
                        #     src_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2RGB)
                        # else:
                        #     src_img = chw2hwc(src_img)

                        boxes, labels, scores = sess.run([fast_rcnn_decode_boxes, detection_category, fast_rcnn_score],
                                                         feed_dict={img_plac: src_img})

                        if show_res:
                            visualize_detection(src_img, boxes, scores)
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
                # ds = None
                time_elapsed = timer() - start
                print("{} detection time : {:.4f} sec".format(img_path.split('/')[-1].split('.')[0], time_elapsed))

                # if target_name == 'aircraft':
                # img = cv2.imread(img_path)
                # if len(img.shape) == 2:
                #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                # elif len(img.shape) == 3:
                #     img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                #     img[:, :, 0] = img[:, :, 1] = img[:, :, 2] = img_gray
                mkdir(des_folder)
                img_np = draw_box_cv(img,
                                     boxes=np.array(box_res),
                                     labels=np.array(label_res),
                                     scores=np.array(score_res)) - np.array([103.939, 116.779, 123.68])
                cv2.imwrite(des_folder + '/{}_fpn.jpg'.format(img_path.split('/')[-1].split('.')[0]), img_np)
                # clip_obj_imgs(src_img, box_res, label_res, score_res, des_folder)
                # print(img_path)
                # det_xml_path =img_path.replace(".tif", ".det.xml")
                # obj_to_det_xml(img_path, box_res, label_res, score_res, det_xml_path)

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

    detect_img(file_paths, args.des_folder, args.det_th, args.h_len, args.w_len, False)

