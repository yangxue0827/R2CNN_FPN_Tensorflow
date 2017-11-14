# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os

from libs.configs import cfgs
from libs.networks.network_factory import get_flags_byname


RESTORE_FROM_RPN = False
FLAGS = get_flags_byname(cfgs.NET_NAME)
os.environ["CUDA_VISIBLE_DEVICES"] = "6"


def get_restorer():

    checkpoint_path = tf.train.latest_checkpoint(os.path.join(FLAGS.trained_checkpoint, cfgs.VERSION))

    if checkpoint_path != None:
        if RESTORE_FROM_RPN:
            print('___restore from rpn___')
            model_variables = slim.get_model_variables()
            restore_variables = [var for var in model_variables if not var.name.startswith('Fast_Rcnn')] + [slim.get_or_create_global_step()]
            for var in restore_variables:
                print(var.name)
            restorer = tf.train.Saver(restore_variables)
        else:
            restorer = tf.train.Saver()
        print("model restore from :", checkpoint_path)
    else:
        checkpoint_path = FLAGS.pretrained_model_path
        print("model restore from pretrained mode, path is :", checkpoint_path)

        model_variables = slim.get_model_variables()

        restore_variables = [var for var in model_variables
                             if (var.name.startswith(cfgs.NET_NAME)
                                 and not var.name.startswith('{}/logits'.format(cfgs.NET_NAME)))]
        for var in restore_variables:
            print(var.name)
        restorer = tf.train.Saver(restore_variables)
    return restorer, checkpoint_path