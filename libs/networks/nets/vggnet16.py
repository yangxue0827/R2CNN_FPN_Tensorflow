# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import inspect
import os

import numpy as np
import tensorflow as tf
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
from libs.configs import cfgs

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg16:
    def __init__(self, vgg16_npy_path=cfgs.VGG16_WEIGHT_PATH):
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16_part.npy")
            vgg16_npy_path = path

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("vgg.npy file loaded")

    def conv_op(self, input_op, name, kh, kw, n_out, dh, dw):
        n_in = input_op.get_shape()[-1].value

        with tf.variable_scope(name):
            weights = tf.get_variable(name='weights',
                                      dtype=tf.float32,
                                      initializer=tf.constant(self.data_dict[name]['weights'], dtype=tf.float32))
            conv = tf.nn.conv2d(input_op, weights, (1, dh, dw, 1), padding='SAME')
            biases = tf.get_variable(name='biases',
                                     dtype=tf.float32,
                                     initializer=tf.constant(self.data_dict[name]['biases'], dtype=tf.float32))
            return tf.nn.relu(tf.nn.bias_add(conv, biases))

    def fc_op(self, input_op, name, n_out):
        n_in = input_op.get_shape()[-1].value

        with tf.variable_scope(name):
            weights = tf.get_variable(name='weights',
                                      dtype=tf.float32,
                                      initializer=tf.Variable(tf.constant(self.data_dict[name]['weights'],
                                                                          name="weights")))

            biases = tf.get_variable(name='biases',
                                     dtype=tf.float32,
                                     initializer=tf.Variable(tf.constant(self.data_dict[name]['biases'],
                                                                         name="biases")))

            fc = tf.nn.relu_layer(input_op, weights, biases)

            return fc

    def mpool_op(self, input_op, name, kh, kw, dh, dw):
        return tf.nn.max_pool(input_op,
                              ksize=[1, kh, kw, 1],
                              strides=[1, dh, dw, 1],
                              padding='SAME',
                              name=name)

    def build(self, rgb, rgb2gbr=False):

        self.color = rgb

        # if use cv2 read image, the channel is gbr, others are rgb
        if rgb2gbr:
            # Convert RGB to BGR
            red, green, blue = tf.split(self.color, num_or_size_splits=3, axis=3)
            self.color = tf.concat([blue - VGG_MEAN[0],
                                    green - VGG_MEAN[1],
                                    red - VGG_MEAN[2]], axis=3)
            self.conv1_1 = self.conv_op(input_op=self.color, name="conv1_1", kh=3, kw=3,
                                        n_out=64, dh=1, dw=1)

        else:

            blue, green, red = tf.split(self.color, num_or_size_splits=3, axis=3)
            self.color = tf.concat([blue - VGG_MEAN[0],
                                    green - VGG_MEAN[1],
                                    red - VGG_MEAN[2]], axis=3)
            self.conv1_1 = self.conv_op(input_op=self.color, name="conv1_1", kh=3, kw=3,
                                        n_out=64, dh=1, dw=1)

        self.conv1_2 = self.conv_op(input_op=self.conv1_1, name="conv1_2", kh=3, kw=3,
                                    n_out=64, dh=1, dw=1)
        self.pool1 = self.mpool_op(input_op=self.conv1_2, name='pool1',
                                   kh=2, kw=2, dw=2, dh=2)

        self.conv2_1 = self.conv_op(input_op=self.pool1, name="conv2_1", kh=3, kw=3,
                                    n_out=128, dh=1, dw=1)
        self.conv2_2 = self.conv_op(input_op=self.conv2_1, name="conv2_2", kh=3, kw=3,
                                    n_out=128, dh=1, dw=1)
        self.pool2 = self.mpool_op(input_op=self.conv2_2, name='pool2',
                                   kh=2, kw=2, dw=2, dh=2)

        self.conv3_1 = self.conv_op(input_op=self.pool2, name="conv3_1", kh=3, kw=3,
                                    n_out=256, dh=1, dw=1)
        self.conv3_2 = self.conv_op(input_op=self.conv3_1, name="conv3_2", kh=3, kw=3,
                                    n_out=256, dh=1, dw=1)
        self.conv3_3 = self.conv_op(input_op=self.conv3_2, name="conv3_3", kh=3, kw=3,
                                    n_out=256, dh=1, dw=1)
        self.pool3 = self.mpool_op(input_op=self.conv3_3, name='pool3',
                                   kh=2, kw=2, dw=2, dh=2)

        self.conv4_1 = self.conv_op(input_op=self.pool3, name="conv4_1", kh=3, kw=3,
                                    n_out=512, dh=1, dw=1)
        self.conv4_2 = self.conv_op(input_op=self.conv4_1, name="conv4_2", kh=3, kw=3,
                                    n_out=512, dh=1, dw=1)
        self.conv4_3 = self.conv_op(input_op=self.conv4_2, name="conv4_3", kh=3, kw=3,
                                    n_out=512, dh=1, dw=1)
        self.pool4 = self.mpool_op(input_op=self.conv4_3, name='pool4',
                                   kh=2, kw=2, dw=2, dh=2)

        self.conv5_1 = self.conv_op(input_op=self.pool4, name="conv5_1", kh=3, kw=3,
                                    n_out=512, dh=1, dw=1)
        self.conv5_2 = self.conv_op(input_op=self.conv5_1, name="conv5_2", kh=3, kw=3,
                                    n_out=512, dh=1, dw=1)
        self.conv5_3 = self.conv_op(input_op=self.conv5_2, name="conv5_3", kh=3, kw=3,
                                    n_out=512, dh=1, dw=1)
        # self.pool5 = self.mpool_op(input_op=self.conv5_3, name='pool5',
        #                            kh=2, kw=2, dw=2, dh=2)
        #
        # shape = self.pool5.get_shape()
        # flattened_shape = shape[1].value * shape[2].value * shape[3].value
        #
        # flatten = tf.reshape(self.pool5, [-1, flattened_shape], name='flatten')
        #
        # self.fc6 = self.fc_op(input_op=self.pool5, name="fc6", n_out=4096)
        #
        # assert self.fc6.get_shape().as_list()[1:] == [4096]
        #
        # self.fc6_drop = tf.nn.dropout(self.fc6, 0.5, name='fc6_drop')
        #
        # self.fc7 = self.fc_op(input_op=self.fc6_drop, name="fc7", n_out=4096)

        # self.fc7_drop = tf.nn.dropout(self.fc7, 0.5, name='fc7_drop')
        #
        # self.fc8 = self.fc_op(input_op=self.fc7_drop, name="fc8", n_out=1000)
        # self.prob = tf.nn.softmax(self.fc8, name="prob")


# if __name__ == '__main__':
#     img_path = cfgs._ROO_PATH + '/demo/0000fdee4208b8b7e12074c920bc6166-0.jpg'
#     img1 = Image.open(img_path)
#     img1 = np.array(img1)
#     # plt.imshow(img1)
#     # plt.show()
#
#     # img2 = cv2.imread(img_path)
#     # plt.imshow(img2)
#     # plt.show()
#     images = tf.placeholder("float32", [1, 96, 96, 3])
#
#     vgg = Vgg16()
#
#     vgg.build(images, rgb2gbr=True)
#
#     with tf.Session() as sess:
#         init = tf.global_variables_initializer()
#         sess.run(init)
#
#         img = np.array([img1], dtype=np.float32)
#         _relu5_3, temp = sess.run([vgg.conv5_3, vgg.color], feed_dict={images: img})
#
#         print('ooo')