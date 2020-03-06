import  os, time
import  numpy as np
import  tensorflow as tf
from    tensorflow import keras
from tensorflow import random

def celoss_ones(logits):
    # 计算属于与标签为1的交叉熵
    y = tf.ones_like(logits)
    loss = keras.losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)


def celoss_zeros(logits):
    # 计算属于与便签为0的交叉熵
    y = tf.zeros_like(logits)
    loss = keras.losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)

def d_loss_fn(d_fake_logits, d_real_logits):
    # 真实图片与1之间的误差
    d_loss_real = celoss_ones(d_real_logits)
    # 生成图片与0之间的误差
    d_loss_fake = celoss_zeros(d_fake_logits)
    # 合并误差
    loss = d_loss_fake + d_loss_real
    return loss


def g_loss_fn(d_fake_logits):
    # 计算生成图片与1之间的误差
    loss = celoss_ones(d_fake_logits)
    return loss

def w_loss_fn(d_fake_logits, d_real_logits):
    '''

    :param d_fake_logits:
    :param d_real_logits:
    :return:
    '''
    d_fake_logits = tf.reduce_mean(d_fake_logits, axis=1)
    d_real_logits = tf.reduce_mean(d_real_logits, axis=1)
    d_loss = -d_real_logits + d_fake_logits
    g_loss = -d_fake_logits
    return d_loss, g_loss

def w_AE_loss_fn(AE_real, AE_fake, real, fake, k_t):
    '''

    :param d_fake_logits:
    :param d_real_logits:
    :return:
    '''
    AE_real_loss = tf.reduce_mean(tf.abs(AE_real - real), axis=[1, 2, 3])
    AE_fake_loss = tf.reduce_mean(tf.abs(AE_fake - fake), axis=[1, 2, 3])
    d_loss = AE_real_loss - k_t * AE_fake_loss
    g_loss = AE_fake_loss
    return d_loss, g_loss, AE_real_loss

def gradient_penalty(D, real, fake, batch_size, is_train=True):
    alpha = random.uniform([batch_size, 1, 1, 1], 0., 1.)
    # diff = fake - real
    diff = real - fake
    inter = real + (alpha * diff)
    with tf.GradientTape() as t:
        t.watch(inter)
        pred = D(inter, is_train)
    grad = t.gradient(pred, [inter])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
    gp = tf.reduce_mean((slopes - 1.)**2, axis=1)
    return gp

def mx_generate_anchors(feature_shape, anchor_scale, anchor_ratio, image_size):
    '''
    Introduciton: 以feature-map的每个点为中心，各生成num_anchor_per_center个anchor-box，
                  因此共有feature_h * feature_w * num_anchor_per_center个。

    :param feature_shape: 特征层的大小，tensor, [h, w]， eg. feature_shape=[32, 16]
    :param anchor_scale: int型，eg. anchor_scale=32
    :param anchor_ratio: list, eg. anchor_ratio=[0.5, 1., 2.0]
    :param image_size:  输入网络图像的大小，tensor, eg. 设图像大小为 image_size=[128, 128], [h, w]
    :return: anchor_boxes, shape=[feature_h * feature_w * num_anchor_per_center, 4]， 注意：坐标为原图上的坐标
    '''
    # 若原图像大小为[128, 128]，映射的feature_map为[32, 16]，则feature_strides= [4,8]
    feature_stride = image_size / feature_shape

    # [A,B]=Meshgrid(a,b), meshgrid用于从数组a和b产生网格。生成的网格矩阵A和B大小是相同的, 它相当于a从一行重复增加到size(b)行，把b转置成一列再重复增加到size(a)列
    # tf.meshgrid([32.0], [0.5, 1., 2.0])
    # scales:                             ratios:
    #       [[32.]                               [[0.5]
    #        [32.]                                [1.0]
    #        [32.]]                               [2.0]]
    scales, ratios = tf.meshgrid([float(anchor_scale)], anchor_ratio)
    scales = tf.reshape(scales, [-1]) # [32, 32, 32]
    ratios = tf.reshape(ratios, [-1]) # [0.5, 1, 2]

    # Enumerate heights and widths from scales and ratios
    heights = scales / tf.sqrt(ratios)  # [45, 32, 22], square root
    widths = scales * tf.sqrt(ratios)  # [22, 32, 45]

    # Enumerate shifts in feature space
    shifts_y = tf.multiply(tf.range(tf.cast(feature_shape[0], tf.int32)), tf.cast(feature_stride[0], tf.int32)) # [0, 4, ... , 128-4]
    shifts_x = tf.multiply(tf.range(tf.cast(feature_shape[1], tf.int32)), tf.cast(feature_stride[1], tf.int32)) # [0, 8, ... , 128-8]

    shifts_x, shifts_y = tf.cast(shifts_x, tf.float32), tf.cast(shifts_y, tf.float32)
    shifts_x, shifts_y = tf.meshgrid(shifts_x, shifts_y)

    # mesh A: [3] B:[32*16]=> [32*16, 3]
    box_widths, box_centers_x = tf.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = tf.meshgrid(heights, shifts_y)

    # box_centers = tf.reshape(tf.stack([box_centers_y, box_centers_x], axis=2), (-1, 2))
    # box_sizes = tf.reshape(tf.stack([box_heights, box_widths], axis=2), (-1, 2))

    box_centers_y = tf.expand_dims(box_centers_y, axis=2)
    box_centers_x = tf.expand_dims(box_centers_x, axis=2)
    box_heights = tf.expand_dims(box_heights, axis=2)
    box_widths = tf.expand_dims(box_widths, axis=2)

    box_centers = tf.reshape(tf.concat([box_centers_y, box_centers_x], axis=-1), (-1, 2))
    box_sizes = tf.reshape(tf.concat([box_heights, box_widths], axis=-1), (-1, 2))

    # Convert to corner coordinates (y1, x1, y2, x2)
    anchor_boxes = tf.concat([box_centers - 0.5 * box_sizes,
                       box_centers + 0.5 * box_sizes], axis=1)

    return anchor_boxes

def mx_compute_overlap(anchors, gt_boxes):
    '''
    Introduction: 计算iou
    anchors:                        gt_boxes:
        [[0.0, 0.0, 1.0, 1.0],              [[0.5, 0.5, 1.0, 1.0],
         [0.2, 0.2, 1.5, 1.5],               [2.0, 1.5, 3.0, 4.0]]
         [2.2, 1.7, 2.8, 4.1]
    输出为iou:
            [[0.25       0.        ]
             [0.14792901 0.        ]
             [0.         0.53076917]]
    从上面可以看出：是将“anchors”中的“每行”依次与“gt_boxes”中的“所有行”计算iou，
                  最终的输出iou：其“每行”对应着“anchors”中的“每行”与“bbox_target”中的“所有行”计算出的iou
                  shape=[num_anchors, num_gt_boxes]
    :param anchors: tensor, [-1, 4], 每行为：[x0, y0, x1, y1]
    :param gt_boxex:tensor, [-1, 4], 每行为：[x0, y0, x1, y1]
    :return: shape=[num_anchors, num_gt_boxes]
    '''
    anchors = tf.expand_dims(anchors, 1)
    u_x0 = tf.where(anchors < gt_boxes, anchors, gt_boxes)[:,:,0]
    u_y0 = tf.where(anchors < gt_boxes, anchors, gt_boxes)[:,:,1]
    u_x1 = tf.where(anchors > gt_boxes, anchors, gt_boxes)[:,:,2]
    u_y1 = tf.where(anchors > gt_boxes, anchors, gt_boxes)[:,:,3]

    i_x0 = tf.where(anchors > gt_boxes, anchors, gt_boxes)[:,:,0]
    i_y0 = tf.where(anchors > gt_boxes, anchors, gt_boxes)[:,:,1]
    i_x1 = tf.where(anchors < gt_boxes, anchors, gt_boxes)[:,:,2]
    i_y1 = tf.where(anchors < gt_boxes, anchors, gt_boxes)[:,:,3]

    i_w = i_x1 - i_x0
    i_h = i_y1 - i_y0
    u_w = u_x1 - u_x0
    u_h = u_y1 - u_y0

    i_w = tf.clip_by_value(i_w, 0, 1e20)
    i_h = tf.clip_by_value(i_h, 0, 1e20)
    u_w = tf.clip_by_value(u_w, 0, 1e20)
    u_h = tf.clip_by_value(u_h, 0, 1e20)

    u_arera = u_w * u_h
    i_arera = i_w * i_h
    iou = i_arera / u_arera

    return iou


