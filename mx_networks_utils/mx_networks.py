import tensorflow as tf
import numpy as np
import cv2, time
from m4_networks_utils.ops import *
import utils.m4_Image_Preprocess as m4_Image_Preprocess
import tensorflow as tf
class YOLO_GAN_Structure:
    def __init__(self, cfg):
        self.cfg = cfg
        self.class_num = cfg.class_num
        self.anchors = m4_Image_Preprocess.get_anchor(self.cfg.anchors_path)
        self.img_size = tf.constant([self.cfg.img_size[0], self.cfg.img_size[1]])

    def darknet53_body(self, x):
        with tf.variable_scope('darknet53_body') as scope:
            net = m4_conv_layers(x, 32, k_h=3, k_w=3, s_h=1, s_w=1,
                               padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                               is_trainable=self.cfg.is_train, stddev=0.02, weight_decay=self.cfg.weight_decay, name='conv_1')
            net = m4_conv_layers(net, 64, k_h=3, k_w=3, s_h=2, s_w=2,
                                 padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                                 is_trainable=self.cfg.is_train, stddev=0.02, weight_decay=self.cfg.weight_decay,
                                 name='conv_2')

            net = self.res_block(net, 32, name='res_block_1')

            net = m4_conv_layers(net, 128, k_h=3, k_w=3, s_h=2, s_w=2,
                                 padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                                 is_trainable=self.cfg.is_train, stddev=0.02, weight_decay=self.cfg.weight_decay,
                                 name='conv_3')

            for i in range(2):
                net = self.res_block(net, 64, name='res_block1_' + str(i))

            net = m4_conv_layers(net, 256, k_h=3, k_w=3, s_h=2, s_w=2,
                                 padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                                 is_trainable=self.cfg.is_train, stddev=0.02, weight_decay=self.cfg.weight_decay,
                                 name='conv_4')

            for i in range(8):
                net = self.res_block(net, 128, name='res_block2_' + str(i))

            route1 = net

            net = m4_conv_layers(net, 512, k_h=3, k_w=3, s_h=2, s_w=2,
                                 padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                                 is_trainable=self.cfg.is_train, stddev=0.02, weight_decay=self.cfg.weight_decay,
                                 name='conv_5')

            for i in range(8):
                net = self.res_block(net, 256, name='res_block3_' + str(i))

            route2 = net

            net = m4_conv_layers(net, 1024, k_h=3, k_w=3, s_h=2, s_w=2,
                                 padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                                 is_trainable=self.cfg.is_train, stddev=0.02, weight_decay=self.cfg.weight_decay,
                                 name='conv_6')

            for i in range(4):
                net = self.res_block(net, 512, name='res_block4_' + str(i))

            route3 = net

            return route1, route2, route3

    def de_darknet53_body(self, route1, route2, route3):
        with tf.variable_scope('de_darknet53_body') as scope:

            net = route3

            for i in range(4):
                net = self.de_res_block(net, 1024, name='de_res_block4_' + str(i))

            shape = net.get_shape()
            new_shape = [shape[0], shape[1]*2, shape[2]*2, shape[3]]
            net = self.upsample_layer(net, new_shape)
            net = m4_conv_layers(net, 512, k_h=3, k_w=3, s_h=1, s_w=1,
                                 padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                                 is_trainable=self.cfg.is_train, stddev=0.02, weight_decay=self.cfg.weight_decay,
                                 name='conv_6')
            net = route2 + net

            for i in range(8):
                net = self.de_res_block(net, 512, name='de_res_block3_' + str(i))

            shape = net.get_shape()
            new_shape = [shape[0], shape[1] * 2, shape[2] * 2, shape[3]]
            net = self.upsample_layer(net, new_shape)
            net = m4_conv_layers(net, 256, k_h=3, k_w=3, s_h=1, s_w=1,
                                 padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                                 is_trainable=self.cfg.is_train, stddev=0.02, weight_decay=self.cfg.weight_decay,
                                 name='conv_5')

            net = route1 + net

            for i in range(8):
                net = self.de_res_block(net, 256, name='de_res_block2_' + str(i))

            shape = net.get_shape()
            new_shape = [shape[0], shape[1] * 2, shape[2] * 2, shape[3]]
            net = self.upsample_layer(net, new_shape)
            net = m4_conv_layers(net, 128, k_h=3, k_w=3, s_h=1, s_w=1,
                                 padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                                 is_trainable=self.cfg.is_train, stddev=0.02, weight_decay=self.cfg.weight_decay,
                                 name='conv_4')
            for i in range(2):
                net = self.de_res_block(net, 128, name='de_res_block1_' + str(i))

            shape = net.get_shape()
            new_shape = [shape[0], shape[1] * 2, shape[2] * 2, shape[3]]
            net = self.upsample_layer(net, new_shape)
            net = m4_conv_layers(net, 64, k_h=3, k_w=3, s_h=1, s_w=1,
                                 padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                                 is_trainable=self.cfg.is_train, stddev=0.02, weight_decay=self.cfg.weight_decay,
                                 name='conv_3')



            net = self.de_res_block(net, 64, name='de_res_block_1')

            shape = net.get_shape()
            new_shape = [shape[0], shape[1] * 2, shape[2] * 2, shape[3]]
            net = self.upsample_layer(net, new_shape)
            net = m4_conv_layers(net, 32, k_h=3, k_w=3, s_h=1, s_w=1,
                                 padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                                 is_trainable=self.cfg.is_train, stddev=0.02, weight_decay=self.cfg.weight_decay,
                                 name='conv_2')

            image = m4_conv_layers(net, 4, k_h=3, k_w=3, s_h=1, s_w=1,
                                 padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                                 is_trainable=self.cfg.is_train, stddev=0.02, weight_decay=self.cfg.weight_decay,
                                 name='conv_1')

            return image


    def forward(self,x):
        route_1, route_2, route_3 = self.darknet53_body(x)
        with tf.variable_scope('yolov3_head') as scope:
            inter1, net = self.yolo_block(route_3, 512, name='yolo_block_1')
            feature_map_1 = m4_conv_layers(net, 3 * (5 + self.class_num), k_h=1, k_w=1, s_h=1, s_w=1,
                                 padding="SAME", get_vars_name=False, active_func=None, norm=None,
                                 is_trainable=self.cfg.is_train, stddev=0.02, weight_decay=self.cfg.weight_decay,
                                 name='feature_map_1')
            inter1 = m4_conv_layers(inter1, 256, k_h=1, k_w=1, s_h=1, s_w=1,
                                           padding="SAME", get_vars_name=False, active_func=None, norm=None,
                                           is_trainable=self.cfg.is_train, stddev=0.02,
                                           weight_decay=self.cfg.weight_decay,
                                           name='conv_1')
            inter1 = self.upsample_layer(inter1, tf.shape(route_2))
            concat1 = tf.concat([inter1, route_2], axis=3)
            inter2, net = self.yolo_block(concat1, 256, name='yolo_block_2')
            feature_map_2 = m4_conv_layers(net, 3 * (5 + self.class_num), k_h=1, k_w=1, s_h=1, s_w=1,
                                           padding="SAME", get_vars_name=False, active_func=None, norm=None,
                                           is_trainable=self.cfg.is_train, stddev=0.02,
                                           weight_decay=self.cfg.weight_decay,
                                           name='feature_map_2')

            inter2 = m4_conv_layers(inter2, 128, k_h=1, k_w=1, s_h=1, s_w=1,
                                           padding="SAME", get_vars_name=False, active_func=None, norm=None,
                                           is_trainable=self.cfg.is_train, stddev=0.02,
                                           weight_decay=self.cfg.weight_decay,
                                           name='conv_2')
            inter2 = self.upsample_layer(inter2, tf.shape(route_1))
            concat2 = tf.concat([inter2, route_1], axis=3)

            _, net = self.yolo_block(concat2, 128, name='yolo_block_3')
            feature_map_3 = m4_conv_layers(net, 3 * (5 + self.class_num), k_h=1, k_w=1, s_h=1, s_w=1,
                                           padding="SAME", get_vars_name=False, active_func=None, norm=None,
                                           is_trainable=self.cfg.is_train, stddev=0.02,
                                           weight_decay=self.cfg.weight_decay,
                                           name='feature_map_3')
            return feature_map_1, feature_map_2, feature_map_3

    def Generator(self, image, z, reuse=False):
        with tf.variable_scope('Generator', reuse=reuse) as scope:
            input_ = tf.concat([image, z], axis=-1)
            feature_map_1_fake, feature_map_2_fake, feature_map_3_fake = self.forward(input_)
            return [feature_map_1_fake, feature_map_2_fake, feature_map_3_fake]

    def Discriminator(self, x, reuse=False):
        x_13, x_26, x_52 = x[0], x[1], x[2]
        x_13_shape, x_26_shape, x_52_shape = x_13.get_shape(), x_26.get_shape(), x_52.get_shape()
        feature_map_1 = tf.reshape(x_13, [-1, x_13_shape[1], x_13_shape[2], x_13_shape[-2] * x_13_shape[-1]])
        feature_map_2 = tf.reshape(x_26, [-1, x_26_shape[1], x_26_shape[2], x_26_shape[-2] * x_26_shape[-1]])
        feature_map_3 = tf.reshape(x_52, [-1, x_52_shape[1], x_52_shape[2], x_52_shape[-2] * x_52_shape[-1]])
        with tf.variable_scope('Discriminator', reuse=reuse) as scope:
            # encoder
            AE_x = self.BE_GAN_Encoder(feature_map_1, feature_map_2, feature_map_3,
                                                                  x_13_shape, x_26_shape, x_52_shape)
            AE_feature_map_1, AE_feature_map_2, AE_feature_map_3 = self.BE_GAN_Decoder(AE_x)
            return [AE_feature_map_1, AE_feature_map_2, AE_feature_map_3]


    def res_block(self, x, filters, active='relu', norm='batch_norm',name='res_block'):
        with tf.variable_scope(name) as scope:
            shortcut = x
            net = m4_conv_layers(x, filters * 1, k_h=1, k_w=1, s_h=1, s_w=1,
                           padding="SAME", get_vars_name=False, active_func=active, norm=norm,
                           is_trainable=self.cfg.is_train, stddev=0.02, weight_decay=self.cfg.weight_decay, name='conv_1')
            net = m4_conv_layers(net, filters * 2, k_h=3, k_w=3, s_h=1, s_w=1,
                                 padding="SAME", get_vars_name=False, active_func=None, norm=norm,
                                 is_trainable=self.cfg.is_train, stddev=0.02, weight_decay=self.cfg.weight_decay, name='conv_2')
            net = shortcut + net
            return net

    def de_res_block(self, x, filters, active='relu', norm='batch_norm',name='de_res_block'):
        with tf.variable_scope(name) as scope:
            shortcut = x
            net = m4_conv_layers(x, filters // 2, k_h=3, k_w=3, s_h=1, s_w=1,
                                 padding="SAME", get_vars_name=False, active_func=None, norm=norm,
                                 is_trainable=self.cfg.is_train, stddev=0.02, weight_decay=self.cfg.weight_decay,
                                 name='conv_2')
            net = m4_conv_layers(net, filters // 1, k_h=1, k_w=1, s_h=1, s_w=1,
                           padding="SAME", get_vars_name=False, active_func=active, norm=norm,
                           is_trainable=self.cfg.is_train, stddev=0.02, weight_decay=self.cfg.weight_decay, name='conv_1')
            net = shortcut + net
            return net

    def yolo_block(self,x, filters, name='yolo_block'):
        with tf.variable_scope(name) as scope:
            net = m4_conv_layers(x, filters * 1, k_h=1, k_w=1, s_h=1, s_w=1,
                                     padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                                     is_trainable=self.cfg.is_train, stddev=0.02, weight_decay=self.cfg.weight_decay,
                                     name='conv_1')
            net = m4_conv_layers(net, filters * 2, k_h=3, k_w=3, s_h=1, s_w=1,
                                 padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                                 is_trainable=self.cfg.is_train, stddev=0.02, weight_decay=self.cfg.weight_decay,
                                 name='conv_2')

            net = m4_conv_layers(net, filters * 1, k_h=1, k_w=1, s_h=1, s_w=1,
                                 padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                                 is_trainable=self.cfg.is_train, stddev=0.02, weight_decay=self.cfg.weight_decay,
                                 name='conv_3')
            net = m4_conv_layers(net, filters * 2, k_h=3, k_w=3, s_h=1, s_w=1,
                                 padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                                 is_trainable=self.cfg.is_train, stddev=0.02, weight_decay=self.cfg.weight_decay,
                                 name='conv_4')
            net = m4_conv_layers(net, filters * 1, k_h=1, k_w=1, s_h=1, s_w=1,
                                 padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                                 is_trainable=self.cfg.is_train, stddev=0.02, weight_decay=self.cfg.weight_decay,
                                 name='conv_5')
            route = net

            net = m4_conv_layers(net, filters * 2, k_h=3, k_w=3, s_h=1, s_w=1,
                                 padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                                 is_trainable=self.cfg.is_train, stddev=0.02, weight_decay=self.cfg.weight_decay,
                                 name='conv_6')

            return route, net

    def deyolo_block(self, x, route, nc_ouput, name='deyolo_block'):
        with tf.variable_scope(name) as scope:
            filters = x.get_shape()[-1]
            net = m4_conv_layers(x, filters // 2, k_h=3, k_w=3, s_h=1, s_w=1,
                                 padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                                 is_trainable=self.cfg.is_train, stddev=0.02, weight_decay=self.cfg.weight_decay,
                                 name='conv_6')
            if route != None:
                net = net + route
            net = m4_conv_layers(net, filters // 1, k_h=1, k_w=1, s_h=1, s_w=1,
                                 padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                                 is_trainable=self.cfg.is_train, stddev=0.02, weight_decay=self.cfg.weight_decay,
                                 name='conv_5')
            net = m4_conv_layers(net, filters // 2, k_h=3, k_w=3, s_h=1, s_w=1,
                                 padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                                 is_trainable=self.cfg.is_train, stddev=0.02, weight_decay=self.cfg.weight_decay,
                                 name='conv_4')
            net = m4_conv_layers(net, filters // 1, k_h=1, k_w=1, s_h=1, s_w=1,
                                 padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                                 is_trainable=self.cfg.is_train, stddev=0.02, weight_decay=self.cfg.weight_decay,
                                 name='conv_3')
            net = m4_conv_layers(net, filters // 2, k_h=3, k_w=3, s_h=1, s_w=1,
                                 padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                                 is_trainable=self.cfg.is_train, stddev=0.02, weight_decay=self.cfg.weight_decay,
                                 name='conv_2')
            net = m4_conv_layers(net, nc_ouput, k_h=1, k_w=1, s_h=1, s_w=1,
                                 padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                                 is_trainable=self.cfg.is_train, stddev=0.02, weight_decay=self.cfg.weight_decay,
                                 name='conv_1')
            return net

    def upsample_layer(self, inputs, out_shape):
        new_height, new_width = out_shape[1], out_shape[2]
        # NOTE: here height is the first
        # TODO: Do we need to set `align_corners` as True?
        inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width), name='upsampled')
        return inputs

    def downsample_layer(self, inputs, out_shape):
        new_height, new_width = out_shape[1], out_shape[2]
        # NOTE: here height is the first
        # TODO: Do we need to set `align_corners` as True?
        inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width), name='downsampled')
        return inputs

    def BE_GAN_Encoder(self, feature_map_1, feature_map_2, feature_map_3, x_13_shape, x_26_shape, x_52_shape):
        with tf.variable_scope('Encoder') as scope:
            net = m4_conv_layers(feature_map_3, 256, k_h=1, k_w=1, s_h=1, s_w=1,
                                 padding="SAME", get_vars_name=False, active_func=None, norm=None,
                                 is_trainable=self.cfg.is_train, stddev=0.02,
                                 weight_decay=self.cfg.weight_decay,
                                 name='de_feature_map_3')  # (-1, 52, 52, 256)
            net = self.deyolo_block(net, None, 384, name='deyolo_block3')
            net, de_route1 = tf.split(net, [128, 384 - 128], axis=-1)

            # de_route1

            net = self.downsample_layer(net, x_26_shape)
            de_inter2 = m4_conv_layers(net, 256, k_h=1, k_w=1, s_h=1, s_w=1,
                                       padding="SAME", get_vars_name=False, active_func=None, norm=None,
                                       is_trainable=self.cfg.is_train, stddev=0.02,
                                       weight_decay=self.cfg.weight_decay,
                                       name='deconv_2')  # (-1, 26, 26, 256)

            net = m4_conv_layers(feature_map_2, 512, k_h=1, k_w=1, s_h=1, s_w=1,
                                 padding="SAME", get_vars_name=False, active_func=None, norm=None,
                                 is_trainable=self.cfg.is_train, stddev=0.02,
                                 weight_decay=self.cfg.weight_decay,
                                 name='de_feature_map_2')  # (-1, 26, 26, 512)

            net = self.deyolo_block(net, de_inter2, 768, name='deyolo_block2')
            net, de_route2 = tf.split(net, [256, 768 - 256], axis=-1)

            # de_route2

            net = self.downsample_layer(net, x_13_shape)
            de_inter1 = m4_conv_layers(net, 512, k_h=1, k_w=1, s_h=1, s_w=1,
                                       padding="SAME", get_vars_name=False, active_func=None, norm=None,
                                       is_trainable=self.cfg.is_train, stddev=0.02,
                                       weight_decay=self.cfg.weight_decay,
                                       name='deconv_1')

            net = m4_conv_layers(feature_map_1, 1024, k_h=1, k_w=1, s_h=1, s_w=1,
                                 padding="SAME", get_vars_name=False, active_func=None, norm=None,
                                 is_trainable=self.cfg.is_train, stddev=0.02,
                                 weight_decay=self.cfg.weight_decay,
                                 name='de_feature_map_1')

            de_route3 = self.deyolo_block(net, de_inter1, 1024, name='deyolo_block1')

            image = self.de_darknet53_body(de_route1, de_route2, de_route3)

            return image

    def BE_GAN_Decoder(self, image):
        with tf.variable_scope('Dencoder') as scope:
            feature_map_1, feature_map_2, feature_map_3 = self.forward(image)
            shape_1 = feature_map_1.get_shape()
            shape_2 = feature_map_2.get_shape()
            shape_3 = feature_map_3.get_shape()
            feature_map_1 = tf.reshape(feature_map_1, [-1, shape_1[1], shape_1[2], 3, shape_1[3] // 3])
            feature_map_2 = tf.reshape(feature_map_2, [-1, shape_2[1], shape_2[2], 3, shape_2[3] // 3])
            feature_map_3 = tf.reshape(feature_map_3, [-1, shape_3[1], shape_3[2], 3, shape_3[3] // 3])
            return feature_map_1, feature_map_2, feature_map_3

    def compute_loss(self, y_pred, y_true):
        '''
        功能：
        :param y_pred: [feature_map_1, feature_map_2, feature_map_3], 其中feature_map的shape是：[-1, rows_map, cols_map, 3 * (5+class_num)]
        :param y_true: [y_true_13, y_true_26, y_true_52], 其中y_true的shape是： [-1, rows_map, cols_map, 3, (5+class_num+1)]
        :return:
        '''
        '''
        param:
            y_pred: returned feature_map list by `forward` function: [feature_map_1, feature_map_2, feature_map_3]
            y_true: input y_true by the tf.data pipeline
        '''
        loss_xy, loss_wh, loss_conf, loss_class = 0., 0., 0., 0.
        anchor_group = [self.anchors[6:9], self.anchors[3:6], self.anchors[0:3]]

        # calc loss in 3 scales
        for i in range(len(y_pred)):
            result = self.loss_layer(y_pred[i], y_true[i], anchor_group[i])
            loss_xy += result[0]
            loss_wh += result[1]
            loss_conf += result[2]
            loss_class += result[3]
        total_loss = loss_xy + loss_wh + loss_conf + loss_class
        return [total_loss, loss_xy, loss_wh, loss_conf, loss_class]

    def post_process(self, feature_map13_26_52_fake, y_true_all):
        anchor_group = [self.anchors[6:9], self.anchors[3:6], self.anchors[0:3]]
        list_pred = []
        list_true = []
        for i in range(3):

            grid_size = feature_map13_26_52_fake[i].get_shape()[1:3]
            # the downscale ratio in height and weight
            ratio = tf.cast(self.img_size / grid_size, tf.float32)
            anchors = anchor_group[i]

            x_y_offset, pred_boxes, conf_logits, prob_logits = self.reorg_layer(feature_map13_26_52_fake[i], anchors)
            y_true = y_true_all[i]

            # shape: [N, 13, 13, 3, 2]
            pred_box_xy = pred_boxes[..., 0:2]
            pred_box_wh = pred_boxes[..., 2:4]

            # get xy coordinates in one cell from the feature_map
            # numerical range: 0 ~ 1
            # shape: [N, 13, 13, 3, 2]
            true_xy = y_true[..., 0:2] / ratio[::-1] - x_y_offset
            pred_xy = pred_box_xy / ratio[::-1] - x_y_offset

            # get_tw_th
            # numerical range: 0 ~ 1
            # shape: [N, 13, 13, 3, 2]
            true_tw_th = y_true[..., 2:4] / anchors
            pred_tw_th = pred_box_wh / anchors
            # for numerical stability
            true_tw_th = tf.where(condition=tf.equal(true_tw_th, 0),
                                  x=tf.ones_like(true_tw_th), y=true_tw_th)
            pred_tw_th = tf.where(condition=tf.equal(pred_tw_th, 0),
                                  x=tf.ones_like(pred_tw_th), y=pred_tw_th)
            true_tw_th = tf.log(tf.clip_by_value(true_tw_th, 1e-9, 1e9))
            pred_tw_th = tf.log(tf.clip_by_value(pred_tw_th, 1e-9, 1e9))

            # conf_logits = tf.nn.sigmoid(conf_logits)
            # prob_logits = tf.nn.softmax(prob_logits)
            post_feature_map = tf.concat([pred_xy, pred_tw_th, conf_logits, prob_logits], axis=-1)

            conf_logits_y_true = y_true[..., 4:5]
            prob_logits_y_true = y_true[..., 5:-1]
            post_y_true = tf.concat([true_xy, true_tw_th, conf_logits_y_true, prob_logits_y_true], axis=-1)

            # post_feature_map = tf.cast(post_feature_map, tf.float32)
            # post_y_true = tf.cast(post_y_true, tf.float32)
            list_pred.append(post_feature_map)
            list_true.append(post_y_true)
        return list_pred, list_true



    def reorg_layer(self, feature_map, anchors):
        '''
        预测的box映射回原图
        :param feature_map: [-1, rows_map, cols_map, 3 * (5+class_num)]
        :param anchors: shape=(3,2)
        :return: x_y_offset: [13, 13, 1, 2]
                 boxes: [N, 13, 13, 3, 4], (center_x, center_y, w, h)， 映射回原图的坐标
                 conf_logits: [N, 13, 13, 3, 1] , no sigmod
                 prob_logits: [N, 13, 13, 3, class_num], no softmax
        '''

        # NOTE: size in [h, w] format! don't get messed up!
        '''
        以网格13*13举例： grid_size=[13, 13]
        '''
        grid_size = feature_map.get_shape()[1:3]  # [13, 13]
        # the downscale ratio in height and weight
        '''
        ratio: 32
        '''
        ratio = tf.cast(self.img_size / grid_size, tf.float32)
        # rescale the anchors to the feature_map
        # NOTE: the anchor is in [w, h] format!
        '''
        anchor: [w,h]
        '''
        rescaled_anchors = [(anchor[0] / ratio[1], anchor[1] / ratio[0]) for anchor in anchors]

        feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], 3, 5 + self.class_num])

        # split the feature_map along the last dimension
        # shape info: take 416x416 input image and the 13*13 feature_map for example:
        # box_centers: [N, 13, 13, 3, 2] last_dimension: [center_x, center_y]
        # box_sizes: [N, 13, 13, 3, 2] last_dimension: [width, height]
        # conf_logits: [N, 13, 13, 3, 1]
        # prob_logits: [N, 13, 13, 3, class_num]
        box_centers, box_sizes, conf_logits, prob_logits = tf.split(feature_map, [2, 2, 1, self.class_num], axis=-1)
        box_centers = tf.nn.sigmoid(box_centers)

        # use some broadcast tricks to get the mesh coordinates
        grid_x = tf.range(grid_size[1], dtype=tf.int32)
        grid_y = tf.range(grid_size[0], dtype=tf.int32)

        '''
        x, y = tf.meshgrid(grid_x, grid_y)
        1. 如果grid_x， grid_y的shape不是(n,)这样的格式， 则先拍平变成(n,)这样的格式
        2. 假设：grid_x=[1,2,3], grid_y=[1,2,3]
        x: [[1, 2, 3],
            [1, 2, 3],
            [1, 2, 3]] 3行， 3列
        y: [[1, 1, 1],
            [2, 2, 2],
            [3, 3, 3]] 3行， 3列
        '''
        grid_x, grid_y = tf.meshgrid(grid_x, grid_y)

        '''
        x_offset: [[1], [2], [3], [1], [2], [3], [1], [2], [3]]
        y_offset: [[1], [1], [1], [2], [2], [2], [3], [3], [3]]
        x_y_offset: [[1,1], 
                     [2,1], 
                     [3,1], 
                     [1,2], 
                     [2,2], 
                     [3,2], 
                     [1,3], 
                     [2,3], 
                     [3,3]]
        '''
        x_offset = tf.reshape(grid_x, (-1, 1))
        y_offset = tf.reshape(grid_y, (-1, 1))
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)

        # shape: [13, 13, 1, 2]
        '''
        x_y_offset:
                    [[[[1. 1.]]
                      [[2. 1.]]
                      [[3. 1.]]]
                     [[[1. 2.]]
                      [[2. 2.]]
                      [[3. 2.]]]
                     [[[1. 3.]]
                      [[2. 3.]]
                      [[3. 3.]]]]
        '''
        x_y_offset = tf.cast(tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2]), tf.float32)

        # get the absolute box coordinates on the feature_map
        box_centers = box_centers + x_y_offset

        # rescale to the original image scale
        # 中心坐标映射回原图
        box_centers = box_centers * ratio[::-1]

        # avoid getting possible nan value with tf.clip_by_value

        box_sizes = tf.exp(
            box_sizes) * rescaled_anchors  # 由于网络输出的boxe size是与feature mapanchor大小的比值， 因此，这里需要转化下（论文中给的是与网格大小的比值， 这里与原文不一样）

        # box_sizes = box_sizes * rescaled_anchors
        # box_sizes = tf.clip_by_value(tf.exp(box_sizes), 1e-9, 100) * rescaled_anchors
        # rescale to the original image scale
        box_sizes = box_sizes * ratio[::-1]

        # shape: [N, 13, 13, 3, 4]
        # last dimension: (center_x, center_y, w, h)
        boxes = tf.concat([box_centers, box_sizes], axis=-1)

        # shape:
        # x_y_offset: [13, 13, 1, 2]
        # boxes: [N, 13, 13, 3, 4], rescaled to the original image scale
        # conf_logits: [N, 13, 13, 3, 1]
        # prob_logits: [N, 13, 13, 3, class_num]
        return x_y_offset, boxes, conf_logits, prob_logits

    def predict(self, feature_maps):
        '''
        Receive the returned feature_maps from `forward` function,
        the produce the output predictions at the test stage.
        '''
        feature_map_1, feature_map_2, feature_map_3 = feature_maps

        feature_map_anchors = [(feature_map_1, self.anchors[6:9]),
                               (feature_map_2, self.anchors[3:6]),
                               (feature_map_3, self.anchors[0:3])]
        reorg_results = [self.reorg_layer(feature_map, anchors) for (feature_map, anchors) in feature_map_anchors]

        def _reshape(result):
            x_y_offset, boxes, conf_logits, prob_logits = result
            grid_size = tf.shape(x_y_offset)[:2]
            boxes = tf.reshape(boxes, [-1, grid_size[0] * grid_size[1] * 3, 4])
            conf_logits = tf.reshape(conf_logits, [-1, grid_size[0] * grid_size[1] * 3, 1])
            prob_logits = tf.reshape(prob_logits, [-1, grid_size[0] * grid_size[1] * 3, self.class_num])
            # shape: (take 416*416 input image and feature_map_1 for example)
            # boxes: [N, 13*13*3, 4]
            # conf_logits: [N, 13*13*3, 1]
            # prob_logits: [N, 13*13*3, class_num]
            return boxes, conf_logits, prob_logits

        boxes_list, confs_list, probs_list = [], [], []
        for result in reorg_results:
            boxes, conf_logits, prob_logits = _reshape(result)
            confs = tf.sigmoid(conf_logits)
            probs = tf.sigmoid(prob_logits)
            boxes_list.append(boxes)
            confs_list.append(confs)
            probs_list.append(probs)

        # collect results on three scales
        # take 416*416 input image for example:
        # shape: [N, (13*13+26*26+52*52)*3, 4]
        boxes = tf.concat(boxes_list, axis=1)
        # shape: [N, (13*13+26*26+52*52)*3, 1]
        confs = tf.concat(confs_list, axis=1)
        # shape: [N, (13*13+26*26+52*52)*3, class_num]
        probs = tf.concat(probs_list, axis=1)

        center_x, center_y, width, height = tf.split(boxes, [1, 1, 1, 1], axis=-1)
        x_min = center_x - width / 2
        y_min = center_y - height / 2
        x_max = center_x + width / 2
        y_max = center_y + height / 2

        boxes = tf.concat([x_min, y_min, x_max, y_max], axis=-1)

        return boxes, confs, probs

    def loss_layer(self, feature_map_i, y_true, anchors):
        '''
        功能：
        :param feature_map_i: [-1, rows_map, cols_map, 3 * (5+class_num)]
        :param y_true: [-1, rows_map, cols_map, 3, (5+class_num+1)]
        :param anchors: shape=(3,2)， 真实的，未resize
        :return:
        '''
        '''
        calc loss function from a certain scale
        input:
            feature_map_i: feature maps of a certain scale. shape: [N, 13, 13, 3*(5 + num_class)] etc.
            y_true: y_ture from a certain scale. shape: [N, 13, 13, 3, 5 + num_class + 1] etc.
            anchors: shape [9, 2]
        '''

        # size in [h, w] format! don't get messed up!
        '''
        假设： grid_size=（13,13）
        '''
        grid_size = tf.shape(feature_map_i)[1:3]
        # the downscale ratio in height and weight
        '''
        假设： img_size=(416,416)
        ratio: 32
        '''
        ratio = tf.cast(self.img_size / grid_size, tf.float32)
        # N: batch_size
        N = tf.cast(tf.shape(feature_map_i)[0], tf.float32)

        '''
        pred_boxes: [N, 13, 13, 3, 4]
        '''
        x_y_offset, pred_boxes, pred_conf_logits, pred_prob_logits = self.reorg_layer(feature_map_i, anchors)

        ###########
        # get mask
        ###########

        # shape: take 416x416 input image and 13*13 feature_map for example:
        # [N, 13, 13, 3, 1]
        object_mask = y_true[..., 4:5]  # 置信度

        # the calculation of ignore mask if referred from
        # https://github.com/pjreddie/darknet/blob/master/src/yolo_layer.c#L179
        ignore_mask = tf.TensorArray(tf.float32, size=0, dynamic_size=True)  # dynamic_size指定数组长度可变

        '''
        idx < N: 遍历一个batch中的所有图片
        '''

        def loop_cond(idx, ignore_mask):
            return tf.less(idx, tf.cast(N, tf.int32))

        def loop_body(idx, ignore_mask):
            '''
            功能： 计算13×13*3个预测的box与gt box的IOU， 如果IOU高于0.5， 则对应的[13, 13, 3]的mask设为0， 否则为1
            :param idx:
            :param ignore_mask:
            :return:
            '''
            # shape: [13, 13, 3, 4] & [13, 13, 3]  ==>  [V, 4]
            # V: num of true gt box of each image in a batch
            '''
            tf.boolean_mask(tensor, mask): 返回tensor与mask中True元素同下标的部分
            例：
            a = tf.constant([[1,2,3],
                             [4,5,6],
                             [7,8,9]])
            b = tf.constant([False,True, True])
            c = tf.boolean_mask(a,b):
                                     [[4 5 6]
                                      [7 8 9]]
            '...':指之间所有的维度,例；
                a = np.array([[[1,2],[3,4]],
                              [[5,6],[7,8]],
                              [[9,1],[2,3]]])， shape=(3,2,2)
                b = a[1, ..., 0:1]:
                                    [[5]
                                     [7]]
            y_true[idx, ..., 0:4]: shape=(13,13,3,4)
            object_mask[idx, ..., 0]: [13,13,3]
            valid_true_boxes: [V,4], label: 表示有没有目标。 这里将网格中有的目标取出来， 即V：表示gt box的数量
            '''
            valid_true_boxes = tf.boolean_mask(y_true[idx, ..., 0:4], tf.cast(object_mask[idx, ..., 0], 'bool'))

            # shape: [13, 13, 3, 4] & [V, 4] ==> [13, 13, 3, V]
            '''
            计算13×13*3中每个预测的box与所有gt box的IOU
            '''
            iou = self.box_iou(pred_boxes[idx], valid_true_boxes)
            # shape: [13, 13, 3]
            best_iou = tf.reduce_max(iou, axis=-1)
            # shape: [13, 13, 3]
            ignore_mask_tmp = tf.cast(best_iou < 0.5, tf.float32)
            # finally will be shape: [N, 13, 13, 3]
            ignore_mask = ignore_mask.write(idx, ignore_mask_tmp)  # 指定idx位置写入tensor
            return idx + 1, ignore_mask

        '''
        tf.while_loop(cond, body, loop_vars), 用法见例子：
        例如我们要用tensorflow实现这样的函数：
                                        i=0
                                        n=10
                                        while(i < n):
                                            i = i+1
        首先要有个判断语句： def cond(i,n):
                                return i < n
        之后是循环体： def body(i,n):
                        i= i + 1
                        return i, n
                注意： body函数中虽然没有与n有关的操作，但必须要传入参数n，如果不传入n下次就无法判断了
        最后合起来就是： 
        i, n = tf.while_loop(cond,body,[i,n])
        ignore_mask： shape=[N, 13, 13, 3]
        '''
        _, ignore_mask = tf.while_loop(cond=loop_cond, body=loop_body, loop_vars=[0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        # shape: [N, 13, 13, 3, 1]
        ignore_mask = tf.expand_dims(ignore_mask, -1)

        # shape: [N, 13, 13, 3, 2]
        pred_box_xy = pred_boxes[..., 0:2]
        pred_box_wh = pred_boxes[..., 2:4]

        # get xy coordinates in one cell from the feature_map
        # numerical range: 0 ~ 1
        # shape: [N, 13, 13, 3, 2]
        true_xy = y_true[..., 0:2] / ratio[::-1] - x_y_offset
        pred_xy = pred_box_xy / ratio[::-1] - x_y_offset

        # get_tw_th
        # numerical range: 0 ~ 1
        # shape: [N, 13, 13, 3, 2]
        true_tw_th = y_true[..., 2:4] / anchors
        pred_tw_th = pred_box_wh / anchors
        # for numerical stability
        true_tw_th = tf.where(condition=tf.equal(true_tw_th, 0),
                              x=tf.ones_like(true_tw_th), y=true_tw_th)
        pred_tw_th = tf.where(condition=tf.equal(pred_tw_th, 0),
                              x=tf.ones_like(pred_tw_th), y=pred_tw_th)
        true_tw_th = tf.log(tf.clip_by_value(true_tw_th, 1e-9, 1e9))
        pred_tw_th = tf.log(tf.clip_by_value(pred_tw_th, 1e-9, 1e9))

        # box size punishment:
        # box with smaller area has bigger weight. This is taken from the yolo darknet C source code.
        # shape: [N, 13, 13, 3, 1]
        box_loss_scale = 2. - (y_true[..., 2:3] / tf.cast(self.img_size[1], tf.float32)) * (
                    y_true[..., 3:4] / tf.cast(self.img_size[0], tf.float32))

        ############
        # loss_part
        ############
        # mix_up weight
        # [N, 13, 13, 3, 1]
        mix_w = y_true[..., -1:]
        # shape: [N, 13, 13, 3, 1]
        xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy) * object_mask * box_loss_scale * mix_w) / N
        wh_loss = tf.reduce_sum(tf.square(true_tw_th - pred_tw_th) * object_mask * box_loss_scale * mix_w) / N

        # shape: [N, 13, 13, 3, 1]
        conf_pos_mask = object_mask
        conf_neg_mask = (1 - object_mask) * ignore_mask
        conf_loss_pos = conf_pos_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                                                logits=pred_conf_logits)
        conf_loss_neg = conf_neg_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                                                logits=pred_conf_logits)
        # TODO: may need to balance the pos-neg by multiplying some weights
        conf_loss = conf_loss_pos + conf_loss_neg
        if True:
            alpha = 1.0
            gamma = 2.0
            # TODO: alpha should be a mask array if needed
            focal_mask = alpha * tf.pow(tf.abs(object_mask - tf.sigmoid(pred_conf_logits)), gamma)
            conf_loss *= focal_mask
        conf_loss = tf.reduce_sum(conf_loss * mix_w) / N

        # shape: [N, 13, 13, 3, 1]
        # whether to use label smooth
        if True:
            delta = 0.01
            label_target = (1 - delta) * y_true[..., 5:-1] + delta * 1. / self.class_num
        else:
            label_target = y_true[..., 5:-1]
        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_target,
                                                                           logits=pred_prob_logits) * mix_w
        class_loss = tf.reduce_sum(class_loss) / N

        return xy_loss, wh_loss, conf_loss, class_loss

    def box_iou(self, pred_boxes, valid_true_boxes):
        '''

        :param pred_boxes: [13, 13, 3, 4], (center_x, center_y, w, h), 映射回原图的
        :param valid_true_boxes: [V, 4]
        :return: iou
        '''

        # [13, 13, 3, 2]
        pred_box_xy = pred_boxes[..., 0:2]
        pred_box_wh = pred_boxes[..., 2:4]

        # shape: [13, 13, 3, 1, 2]
        pred_box_xy = tf.expand_dims(pred_box_xy, -2)
        pred_box_wh = tf.expand_dims(pred_box_wh, -2)

        # [V, 2]
        true_box_xy = valid_true_boxes[:, 0:2]
        true_box_wh = valid_true_boxes[:, 2:4]

        # [13, 13, 3, 1, 2] & [V, 2] ==> [13, 13, 3, V, 2]
        intersect_mins = tf.maximum(pred_box_xy - pred_box_wh / 2.,
                                    true_box_xy - true_box_wh / 2.)
        intersect_maxs = tf.minimum(pred_box_xy + pred_box_wh / 2.,
                                    true_box_xy + true_box_wh / 2.)
        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)

        # shape: [13, 13, 3, V]
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        # shape: [13, 13, 3, 1]
        pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
        # shape: [V]
        true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]
        # shape: [1, V]
        true_box_area = tf.expand_dims(true_box_area, axis=0)

        # [13, 13, 3, V]
        iou = intersect_area / (pred_box_area + true_box_area - intersect_area + 1e-10)

        return iou