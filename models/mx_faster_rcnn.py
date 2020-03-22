import numpy as np
import tensorflow as tf
# 导入 keras 模型，不能使用 import keras，它导入的是标准的 Keras 库
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import argparse, os, time, sys

sys.path.append('../')
import models.resnet as resnet
import mx_networks_utils.mx_ops as ops


class Faster_RCNN(keras.Model):
    def __init__(self, cfg,
                 num_classes,
                 anchor_scales,
                 anchor_ratios,
                 rpn_batch_size,
                 rpn_pos_frac,
                 rpn_pos_iou_thr,
                 rpn_neg_iou_thr,
                 rpn_proposal_count,
                 rpn_nms_thr,
                 pooling_size,
                 rcnn_batch_size,
                 rcnn_pos_frac,
                 rcnn_pos_iou_thr,
                 rcnn_neg_iou_thr,
                 rcnn_min_confidence,
                 rcnn_nms_thr,
                 rcnn_max_instance):
        super(Faster_RCNN, self).__init__()
        print('****************************fater rcnn****************************')
        print(' num_classes:{} \n anchor_scales:{} \n anchor_ratios:{} \n rpn_batch_size:{} \n rpn_pos_frac:{} \
              \n rpn_pos_iou_thr:{} \n rpn_neg_iou_thr:{} \n rpn_proposal_count:{} \n rpn_nms_thr:{} \
              \n pooling_size:{} \n rcnn_batch_size:{} \n rcnn_pos_frac:{} \n rcnn_pos_iou_thr:{} \n rcnn_neg_iou_thr:{} \
              \n rcnn_min_confidence:{} \n rcnn_nms_thr:{} \n rcnn_max_instance:{} \n'.format(
            num_classes, anchor_scales, anchor_ratios, rpn_batch_size, rpn_pos_frac,
            rpn_pos_iou_thr, rpn_neg_iou_thr, rpn_proposal_count, rpn_nms_thr, pooling_size,
            rcnn_batch_size, rcnn_pos_frac, rcnn_pos_iou_thr, rcnn_neg_iou_thr, rcnn_min_confidence, rcnn_nms_thr,
            rcnn_max_instance))
        self.cfg = cfg
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.rpn_pos_iou_thr = rpn_pos_iou_thr
        self.rpn_neg_iou_thr = rpn_neg_iou_thr
        self.rpn_batch_size = rpn_batch_size
        self.rpn_pos_frac = rpn_pos_frac
        self.image_size = tf.constant(cfg.img_size[0:2], dtype=tf.int32)
        self.backbone = resnet.resnet101(training=self.cfg.is_train)
        self.rpn = RPNHead(anchor_scales=anchor_scales,
                           anchor_ratios=anchor_ratios,
                           proposal_count=rpn_proposal_count,
                           nms_threshold=rpn_nms_thr,
                           batch_size=rpn_batch_size,
                           positive_fraction=rpn_pos_frac,
                           pos_iou_thr=rpn_pos_iou_thr,
                           neg_iou_thr=rpn_neg_iou_thr)

        self.rcnn = RCNNHead(num_classes,
                             rcnn_batch_size,
                             rcnn_pos_frac,
                             rcnn_pos_iou_thr,
                             rcnn_neg_iou_thr,
                             rcnn_min_confidence,
                             rcnn_nms_thr,
                             rcnn_max_instance,
                             res5=self.backbone.res5)

    def call(self, inputs):
        num_gt, batch_image_real, boxes, labels, img_width, img_height, batch_z = inputs
        # 1. build base network
        out2, out3, out4, out5 = self.backbone(batch_image_real)
        feature_to_cropped = out4

        # 1. rpn
        feature_shape = feature_to_cropped.shape[1:3]
        rpn_class_score, rpn_class_probs, rpn_bbox_pred = self.rpn(feature_to_cropped)

        # 1.1 generate_anchors
        anchors = self.rpn.mx_generate_anchors(feature_shape, self.anchor_scales, self.anchor_ratios, self.image_size)

        # 1.2 build target
        rpn_classes_target, rpn_bbox_target = self.rpn.build_target(anchors, gt_boxes=boxes,
                                                                    neg_iou_thr=self.rpn_neg_iou_thr,
                                                                    pos_iou_thr=self.rpn_pos_iou_thr,
                                                                    batch_size=self.rpn_batch_size,
                                                                    positive_fraction=self.rpn_pos_frac)
        self.rpn_class_acc = self.rpn.rpn_acc(rpn_class_probs, rpn_classes_target)

        # 1.3 rpn_loss
        rpn_class_loss = self.rpn.rpn_class_loss(rpn_class_score, rpn_classes_target)
        rpn_location_loss = self.rpn.rpn_location_loss(rpn_bbox=rpn_bbox_pred,
                                                       target_bbox=rpn_bbox_target,
                                                       target_class=rpn_classes_target)

        # 1.4 get proposals
        self.proposal_bbox, self.proposal_probs = self.rpn.get_proposals(rpn_class_probs, rpn_bbox_pred, anchors,
                                                                         self.image_size)

        # 2. rcnn
        # 2.1 biuld target
        rcnn_classes_target, rcnn_bbox_target, rois = self.rcnn.build_target(self.proposal_bbox, boxes, labels,
                                                                             self.image_size)
        # 2.2 roi pooling
        pooled_features = self.rcnn.roi_pooling(rois, feature_to_cropped)
        # 2.3 network output
        rcnn_class_score, rcnn_class_prob, rcnn_bbox_pred = self.rcnn(pooled_features)
        self.rcnn_class_acc = self.rcnn.rcnn_acc(rcnn_class_prob, rcnn_classes_target)

        # 2.3 rcnn loss
        rcnn_class_loss = self.rcnn.rcnn_class_loss(rcnn_class_score, rcnn_classes_target)
        rcnn_location_loss = self.rcnn.rcnn_location_loss(rcnn_bbox_pred, rcnn_bbox_target, rcnn_classes_target)

        # 2.4 detect image
        self.final_class, self.final_box, self.final_score = self.rcnn.detect_image(rcnn_class_prob, rcnn_bbox_pred, rois)

        return rpn_class_loss, rpn_location_loss, rcnn_class_loss, rcnn_location_loss

    def detect_single_image(self, image):
        out2, out3, out4, out5 = self.backbone(image)
        feature_to_cropped = out4

        # 1. rpn
        feature_shape = feature_to_cropped.shape[1:3]
        rpn_class_score, rpn_class_probs, rpn_bbox_pred = self.rpn(feature_to_cropped)

        # 1.1 generate_anchors
        anchors = self.rpn.mx_generate_anchors(feature_shape, self.anchor_scales, self.anchor_ratios, self.image_size)

        # 1.2 get proposals
        proposal_bbox, proposal_probs = self.rpn.get_proposals(rpn_class_probs, rpn_bbox_pred, anchors,
                                                                         self.image_size)

        # 2. rcnn

        # 2.2 roi pooling
        pooled_features = self.rcnn.roi_pooling(proposal_bbox, feature_to_cropped)
        # 2.3 network output
        rcnn_class_score, rcnn_class_prob, rcnn_bbox_pred = self.rcnn(pooled_features)
        # 2.4 detect image
        final_class, final_box, final_score = self.rcnn.detect_image(rcnn_class_prob, rcnn_bbox_pred, proposal_bbox)
        return final_class, final_box, final_score


class RPNHead(keras.Model):

    def __init__(self,
                 anchor_scales,
                 anchor_ratios,
                 proposal_count,
                 nms_threshold,
                 batch_size,
                 positive_fraction,
                 pos_iou_thr,
                 neg_iou_thr):
        super(RPNHead, self).__init__()
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold
        self.batch_size = batch_size
        self.positive_fraction = positive_fraction
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        print('**********************************RPNHead**********************************')
        print(' anchor_scales:{} \n anchor_ratios:{} \n proposal_count:{} \n nms_threshold:{} \n batch_size:{} \n positive_fraction:{} \
              \n pos_iou_thr:{} \n neg_iou_thr:{}'.format(
            self.anchor_scales, self.anchor_ratios, self.proposal_count, self.nms_threshold, self.batch_size,
            self.positive_fraction, self.pos_iou_thr, self.neg_iou_thr))

        # Shared convolutional base of the RPN
        self.rpn = layers.Conv2D(512, (3, 3), padding='same',
                                 kernel_initializer='he_normal',
                                 name='rpn_conv_shared')

        self.rpn_class_score = layers.Conv2D(len(anchor_ratios) * len(anchor_scales) * 2, (3, 3), padding='same',
                                             kernel_initializer='he_normal',
                                             name='rpn_class_raw')

        self.rpn_bbox_pred = layers.Conv2D(len(anchor_ratios) * len(anchor_scales) * 4, (3, 3), padding='same',
                                           kernel_initializer='he_normal',
                                           name='rpn_bbox_pred')

    def call(self, inputs):
        '''

        :param inputs:
        :return:
                rpn_probs: shape = [num_anchors, 2]
                rpn_deltas: shape = [num_anchors, 4]
        '''

        shared = self.rpn(inputs)
        self.feature_shape = shared.shape[1:3]
        shared = tf.nn.relu(shared)

        # class output
        # rpn_probs: shape----> [batch, num_anchors, 2]
        rpn_class_score = self.rpn_class_score(shared)
        rpn_class_score = tf.reshape(rpn_class_score, [-1, 2])
        rpn_class_probs = tf.nn.softmax(rpn_class_score)

        # center, size output
        # rpn_deltas: shape----> [batch, num_anchors, 4]
        rpn_bbox_pred = self.rpn_bbox_pred(shared)
        rpn_bbox_pred = tf.reshape(rpn_bbox_pred, [-1, 4])

        return rpn_class_score, rpn_class_probs, rpn_bbox_pred

    def mx_generate_anchors(self, feature_shape, anchor_scale, anchor_ratio, image_size):
        '''
        Introduciton: 以feature-map的每个点为中心，各生成num_anchor_per_center个anchor-box，
                      因此共有feature_h * feature_w * num_anchor_per_center个。
                      num_anchor_per_center个anchor的计算：以anchor_scale=[128,256,512]，anchor_ratio=[0.5, 1., 2.0]为例：
                      anchor高度(heights)的计算：
                          每个anchor_scale 除以 每个anchor_ratio的开方，即：
                          128有3个: 128 / sqrt(0.5), 128 / sqrt(1), 128 / sqrt(2) ----> h1, h2, h3
                          256有3个: 256 / sqrt(0.5), 256 / sqrt(1), 256 / sqrt(2) ----> h4, h5, h6
                          512有3个: 512 / sqrt(0.5), 512 / sqrt(1), 512 / sqrt(2) ----> h7, h8, h9
                          共9个：[h1, h2, h3, h4, h5, h6, h7, h8, h9]

                      anchor宽度(widths)的计算：
                          每个anchor_scale 乘以 每个anchor_ratio的开方，即：
                          128有3个: 128 * sqrt(0.5), 128 * sqrt(1), 128 * sqrt(2) ----> w1, w2, w3
                          256有3个: 256 * sqrt(0.5), 256 * sqrt(1), 256 * sqrt(2) ----> w4, w5, w6
                          512有3个: 512 * sqrt(0.5), 512 * sqrt(1), 512 * sqrt(2) ----> w7, w8, w9
                          共9个：[w1, w2, w3, w4, w5, w6, w7, w8, w9]
                      因此，共有9个anchor，分别为[[h1, w1], [h2, w2], [h3, w3], [h4, w4], [h5, w5], [h6, w6], [h7, w7], [h8, w8], [h9, w9]]

        :param feature_shape: 特征层的大小，tensor, [h, w]， eg. feature_shape=[32, 16]
        :param anchor_scale: list，eg. anchor_scale=[128,256,512]
        :param anchor_ratio: list, eg. anchor_ratio=[0.5, 1., 2.0]
        :param image_size:  输入网络图像的大小，tensor, eg. 设图像大小为 image_size=[256, 256], [h, w]
        :return: anchor_boxes
                shape = [num_anchors, 4] = [feature_h * feature_w * num_anchor_per_center, 4]， 注意：坐标为原图上的坐标
                [y_min, x_min, y_max, x_max]
        '''
        # 若原图像大小为[128, 128]，映射的feature_map为[32, 16]，则feature_strides= [4,8]
        feature_stride = image_size / feature_shape

        anchor_scale = tf.constant(anchor_scale, dtype=tf.float32)
        anchor_ratio = tf.constant(anchor_ratio, dtype=tf.float32)
        anchor_ratio = tf.reshape(anchor_ratio, [-1, 1])

        heights = anchor_scale / tf.sqrt(anchor_ratio)  # [[h1, h2, h3],
        #  [h4, h5, h6],
        #  [h7, h8, h9]]
        widths = anchor_scale * tf.sqrt(anchor_ratio)  # [[w1, w2, w3],
        #  [w4, w5, w6],
        #  [w7, w8, w9]]

        # Enumerate shifts in feature space
        shifts_y = (tf.range(tf.cast(feature_shape[0], tf.float32)) + 0.5) * tf.cast(feature_stride[0],
                                                                                     tf.float32)  # [2, 6, ... , 128-2]
        shifts_x = (tf.range(tf.cast(feature_shape[1], tf.float32)) + 0.5) * tf.cast(feature_stride[1],
                                                                                     tf.float32)  # [4, 12, ... , 128-4]

        # [A,B]=Meshgrid(a,b), meshgrid用于从数组a和b产生网格。生成的网格矩阵A和B大小是相同的, 它相当于a从一行重复增加到size(b)行，把b转置成一列再重复增加到size(a)列
        # 如果a，b为2维以上的，则先拍扁，例：
        # a:                b:
        #   [[1,2],           [[5,6],
        #    [3,4]]            [7,8]]
        # A:                             B:
        #   [[1,2,3,4],                     [[5,5,5,5],
        #    [1,2,3,4],                      [6,6,6,6],
        #    [1,2,3,4],                      [7,7,7,7],
        #    [1,2,3,4]]                      [8,8,8,8]]
        shifts_x, shifts_y = tf.cast(shifts_x, tf.float32), tf.cast(shifts_y, tf.float32)
        shifts_x, shifts_y = tf.meshgrid(shifts_x, shifts_y)

        # mesh A: [num_anchor_per_center] B:[feature_shape[0]*feature_shape[1]]=> [feature_shape[0]*feature_shape[1], num_anchor_per_center]
        box_widths, box_centers_x = tf.meshgrid(widths, shifts_x)
        box_heights, box_centers_y = tf.meshgrid(heights, shifts_y)

        box_centers_y = tf.expand_dims(box_centers_y, axis=2)
        box_centers_x = tf.expand_dims(box_centers_x, axis=2)
        box_heights = tf.expand_dims(box_heights, axis=2)
        box_widths = tf.expand_dims(box_widths, axis=2)

        box_centers = tf.reshape(tf.concat([box_centers_y, box_centers_x], axis=-1), (-1, 2))
        box_sizes = tf.reshape(tf.concat([box_heights, box_widths], axis=-1), (-1, 2))

        # Convert to corner coordinates (y1, x1, y2, x2)
        anchor_boxes = tf.concat([box_centers - 0.5 * box_sizes,
                                  box_centers + 0.5 * box_sizes], axis=1)
        anchor_boxes = ops.bbox_clip(anchor_boxes,
                                     tf.concat([tf.constant([0, 0], dtype=tf.float32), tf.cast(image_size, tf.float32)],
                                               axis=0))
        anchor_boxes = tf.stop_gradient(anchor_boxes)
        return anchor_boxes

    def build_target(self, anchors, gt_boxes, neg_iou_thr, pos_iou_thr, batch_size, positive_fraction):
        '''

        :param anchors: [num_anchors, 4] ----> [y_min, x_min, y_max, x_max]
        :param gt_boxes: [-1, num_gt, 4] ----> [y_min, x_min, y_max, x_max]
        :param neg_iou_thr: float
        :param pos_iou_thr: float
        :param batch_size: int 取出样本的个数
        :param positive_fraction：float 正样本所占的比例
        :return:
                target_matchs: shape=[num_anchors] 1=positive, -1=negative, 0=neutral anchor.
                target_deltas: num_rpn_deltas 正样本---->[dy, dx, dh, dw]   负样本---->[0,0,0,0]
        '''
        gt_boxes = tf.reshape(gt_boxes, [-1, 4])  # [num_gt, 4]

        # 初始都为0， 一系列操作后， -1：负样本， 0：不关心的样本， 1：正样本
        target_matchs = tf.zeros(anchors.shape[0], dtype=tf.int32)

        y_min_A, x_min_A, y_max_A, x_max_A = tf.split(anchors, [1, 1, 1, 1], axis=-1)
        anchors_convert = tf.concat([x_min_A, y_min_A, x_max_A, y_max_A], axis=-1)

        y_min_G, x_min_G, y_max_G, x_max_G = tf.split(gt_boxes, [1, 1, 1, 1], axis=-1)
        gt_boxes_convert = tf.concat([x_min_G, y_min_G, x_max_G, y_max_G], axis=-1)

        # Compute overlaps [num_anchors, num_gt_boxes]
        overlaps = ops.mx_compute_overlap(anchors_convert, gt_boxes_convert)
        # Match anchors to GT Boxes
        # If an anchor overlaps ANY GT box with IoU >= 0.7 then it's positive.
        # If an anchor overlaps ALL GT box with IoU < 0.3 then it's negative.
        # Neutral anchors are those that don't match the conditions above,
        # and they don't influence the loss function.
        # However, don't keep any GT box unmatched (rare, but happens). Instead,
        # match it to the closest anchor (even if its max IoU is < 0.3).

        # 1. Set negative anchors first. They get overwritten below if a GT box is
        # matched to them. [N_anchors, N_gt_boxes]
        anchor_iou_argmax = tf.argmax(overlaps, axis=1)  # [326396] get clost gt boxes for each anchors
        anchor_iou_max = tf.reduce_max(overlaps, axis=[1])  # [326396] get closet gt boxes's overlap scores
        # if an anchor box overlap all GT box with IoU < 0.3, marked as -1 background
        target_matchs = tf.where(anchor_iou_max < neg_iou_thr,
                                 -tf.ones(anchors.shape[0], dtype=tf.int32), target_matchs)

        # 2. Set anchors with high overlap as positive.
        # if an anchor overlap with any GT box with IoU > 0.7, marked as foreground
        target_matchs = tf.where(anchor_iou_max >= pos_iou_thr,
                                 tf.ones(anchors.shape[0], dtype=tf.int32), target_matchs)

        # 3. Set an anchor for each GT box (regardless of IoU value).
        # update corresponding value=>1 for GT boxes' closest boxes
        # 解释：将每个gt_box与所以anchor_boxes中iou最大的那个anchor设为正样本，这样就确保每个gt_box都对应一个anchor，即保证至少有num_gt个正样本
        gt_iou_argmax = tf.argmax(overlaps, axis=0)  # [N_gt_boxes]
        target_matchs = tf.compat.v1.scatter_update(tf.Variable(target_matchs), gt_iou_argmax, 1)

        # Subsample to balance positive and negative anchors
        # Don't let positives be more than half the anchors
        ids = tf.where(tf.equal(target_matchs, 1))  # [N_pos_anchors, 1], [15, 1]
        ids = tf.squeeze(ids, 1)  # [15]
        # 目前正样本的数量 - 应有正样本的数量
        extra = ids.shape.as_list()[0] - int(batch_size * positive_fraction)
        # 如果目前正样本的数量 > 应有正样本的数量, 则随机去掉 (目前正样本的数量 - 应有正样本的数量)正样本
        if extra > 0:  # extra means the redundant pos_anchors
            # Reset the extra random ones to neutral
            ids = tf.random.shuffle(ids)[:extra]
            target_matchs = tf.compat.v1.scatter_update(target_matchs, ids, 0)

        # Same for negative proposals
        ids = tf.where(tf.equal(target_matchs, -1))  # [213748, 1]
        ids = tf.squeeze(ids, 1)
        # 目前负样本的数量 - 应有负样本的数量
        extra = ids.shape.as_list()[0] - (batch_size - tf.reduce_sum(tf.cast(tf.equal(target_matchs, 1), tf.int32)))
        # 如果目前负样本的数量 > 应有负样本的数量, 则随机去掉 (目前负样本的数量 - 应有负样本的数量)正样本
        if extra > 0:  # 213507, so many negative anchors!
            # Rest the extra ones to neutral
            ids = tf.random.shuffle(ids)[:extra]
            target_matchs = tf.compat.v1.scatter_update(target_matchs, ids, 0)

        # since we only need 256 anchors, and it had better contains half positive anchors, and harlf neg .
        # For positive anchors, compute shift and scale needed to transform them
        # to match the corresponding GT boxes.
        ids = tf.where(tf.equal(target_matchs, 1))  # [15]

        # 取出正样本anchor
        a = tf.gather_nd(anchors, ids)
        # 取出对应的gt
        anchor_idx = tf.gather_nd(anchor_iou_argmax, ids)
        gt = tf.gather(gt_boxes, anchor_idx)
        target_deltas = ops.mx_encode_bbox(a, gt)

        padding = tf.maximum(batch_size - tf.shape(target_deltas)[0], 0)
        target_deltas = tf.pad(target_deltas, [(0, padding), (0, 0)])
        target_matchs = tf.stop_gradient(target_matchs)
        target_deltas = tf.stop_gradient(target_deltas)
        return target_matchs, target_deltas

    def get_proposals(self, rpn_probs, rpn_bbox, anchors, image_size):
        '''
        Introduction: 根据rpn预测的分数和box的大小：
                      1. 限定anchor数量的上限为12000（训练时）；
                      2. 去掉重叠率高的anchor， 取出前2000个分数高的anchor,就是proposals
        :param rpn_probs: [num_anchors, 2]
        :param rpn_bbox: [num_anchors, 4] [dy, dx, dh, dw]
        :param anchors: [num_anchors, 4], [y0, x0, y1, x1]
        :param image_size: tensor [h, w]
        :return:
                proposal_bbox: [2000, 4] [y0,x0,y1,x1], Normalize:0~1
                proposal_probs:[2000]
        '''
        # 1. decode boxes
        rpn_probs = rpn_probs[:, 1]
        decode_rpn_bbox = ops.mx_decode_bbox(anchors, rpn_bbox)
        # 2. clip to img boundaries
        window = tf.concat([tf.constant([0, 0], dtype=tf.float32), tf.cast(image_size, tf.float32)], axis=0)
        decode_rpn_bbox = ops.bbox_clip(decode_rpn_bbox, window)

        # 3. get top N to NMS
        pre_nms_topN = min(12000, anchors.shape[0])
        ix = tf.nn.top_k(rpn_probs, pre_nms_topN, sorted=True).indices  # 先从大到小排序，取出前pre_nms_topN值，并返回索引
        # 获取前pre_nms_topN个rpn_probs，
        proposal_probs = tf.gather(rpn_probs, ix)
        proposal_bbox = tf.gather(decode_rpn_bbox, ix)
        # anchors = tf.gather(anchors, ix)

        # Normalize, (y1, x1, y2, x2)
        proposal_bbox = proposal_bbox / tf.cast(tf.concat([image_size, image_size], axis=0), dtype=tf.float32)

        # NMS, indices: [2000]
        indices = tf.image.non_max_suppression(
            proposal_bbox, proposal_probs, self.proposal_count, self.nms_threshold)

        proposal_bbox = tf.gather(proposal_bbox, indices)  # [2000, 4]
        proposal_probs = tf.gather(proposal_probs, indices)
        # if True:
        #     proposal_probs = tf.expand_dims(tf.gather(rpn_probs, indices), axis=1)
        #     proposals = tf.concat([proposals, proposal_probs], axis=1)
        # anchor_boxes = tf.stop_gradient(anchor_boxes)
        proposal_bbox = tf.stop_gradient(proposal_bbox)
        proposal_probs = tf.stop_gradient(proposal_probs)
        return proposal_bbox, proposal_probs

    def rpn_acc(self, rpn_class_probs, rpn_classes_target):
        '''

        :param rpn_class_probs: shape=[-1,2]
        :param rpn_classes_target: shape=[-1]  1=positive, -1=negative, 0=neutral anchor.
        :return:
        '''
        # 1=positive, -1=negative, 0=neutral anchor.
        # pos_neg_idx： 正负样本的索引
        pos_neg_idx = tf.where(tf.not_equal(rpn_classes_target, 0))[:, 0]
        # class_target： 取出正负样本
        class_target = tf.gather(rpn_classes_target, pos_neg_idx)
        # 将负样本的label变成0
        class_target_mask = tf.zeros_like(class_target)
        class_target = tf.where(class_target == -1, class_target_mask, class_target)
        class_target = tf.cast(class_target, tf.float32)

        # 获取预测结果
        class_pred = tf.gather(rpn_class_probs, pos_neg_idx)
        class_pred = tf.cast(tf.argmax(class_pred, axis=1), tf.float32)

        rpn_class_acc = tf.reduce_mean(tf.cast(tf.equal(class_target, class_pred), tf.float32))

        return rpn_class_acc

    def rpn_class_loss(self, rpn_class_score, rpn_classes_target):
        '''

        :param rpn_class_score: rpn_probs: [num_anchors, 2], batch_size=1
        :param rpn_classes_target: [num_anchors]  1=positive, -1=negative, 0=neutral anchor.
        :return:
        '''

        # Get anchor classes. Convert the -1/+1 match to 0/1 values.
        # 将 -1=negative, 0=neutral 全部变成0
        anchor_class = tf.cast(tf.equal(rpn_classes_target, 1), tf.int32)

        # Positive and Negative anchors contribute to the loss,
        # but neutral anchors (match value = 0) don't.
        # 除去0=neutral anchor，获取正负样本的索引
        indices = tf.where(tf.not_equal(rpn_classes_target, 0))
        # 通过所以取出正负样本
        anchor_class = tf.gather(anchor_class, tf.reshape(indices, [-1]))
        # 通过所以取出正负样本
        rpn_class_score = tf.reshape(rpn_class_score, [-1, 2])
        rpn_class_score = tf.gather(rpn_class_score, tf.reshape(indices, [-1]))
        num_classes = rpn_class_score.shape[-1]
        loss = keras.losses.categorical_crossentropy(tf.one_hot(anchor_class, depth=num_classes),
                                                     rpn_class_score, from_logits=True)
        return tf.reduce_mean(loss)

    def rpn_location_loss(self, rpn_bbox, target_bbox, target_class):
        '''

        :param rpn_bbox: [num_anchors, 4]， 网络的输出
        :param target_bbox: [样本数,4]
        :param target_class: [num_anchors]  1=positive, -1=negative, 0=neutral anchor.
        :return:
        '''
        # Positive anchors contribute to the loss, but negative and
        # neutral anchors (match value of 0 or -1) don't.
        indices = tf.where(tf.equal(target_class, 1))

        # Pick bbox deltas that contribute to the loss
        rpn_bbox = tf.reshape(rpn_bbox, [-1, 4])
        rpn_bbox = tf.gather_nd(rpn_bbox, indices)

        num_postive = rpn_bbox.shape.as_list()[0]
        target_bbox = target_bbox[:num_postive, :]
        loss = ops.smooth_l1_loss(target_bbox, rpn_bbox)
        return tf.reduce_mean(loss)


class RCNNHead(keras.Model):
    def __init__(self, num_classes,
                 batch_size,
                 pos_frac,
                 pos_iou_thr,
                 neg_iou_thr,
                 min_confidence,
                 nms_threshold,
                 max_instances,
                 res5):
        super(RCNNHead, self).__init__()

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.pos_frac = pos_frac
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_confidence = min_confidence
        self.nms_threshold = nms_threshold
        self.max_instances = max_instances
        self.res5 = res5

        self.max_pool = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')

        self.rcnn_class_score = layers.Dense(num_classes, name='rcnn_class_logits')

        self.rcnn_bbox_pred = layers.Dense(num_classes * 4, name='rcnn_bbox_fc')

    def call(self, inputs):
        '''

        :param inputs: [num_rois, h, w, dim]
        :return:
        '''
        x = self.res5(inputs)
        fc_flatten = tf.reduce_mean(x, axis=[1, 2])
        rcnn_class_score = self.rcnn_class_score(fc_flatten)
        rcnn_class_score = tf.reshape(rcnn_class_score, [-1, self.num_classes])
        rcnn_class_prob = tf.nn.softmax(rcnn_class_score)

        rcnn_bbox_pred = self.rcnn_bbox_pred(fc_flatten)
        rcnn_bbox_pred = tf.reshape(rcnn_bbox_pred, [-1, 4 * self.num_classes])

        return rcnn_class_score, rcnn_class_prob, rcnn_bbox_pred

    def roi_pooling(self, proposals, feature_maps):
        '''

        :param proposals:
        :return:
        '''

        N = proposals.shape.as_list()[0]
        cropped_roi_features = tf.image.crop_and_resize(image=feature_maps, boxes=proposals,
                                                        box_indices=tf.zeros(shape=[N, ], dtype=tf.int32),
                                                        crop_size=[14, 14])
        roi_features = self.max_pool(cropped_roi_features)
        return roi_features

    def build_target(self, proposals, gt_boxes, gt_classes, image_size):
        '''

        :param proposals: [-1, 4]
        :param gt_boxes: [-1, num_gt, 4]
        :param gt_classes: [-1, num_gt]
        :param image_size: [h, w]
        :return:
        '''

        gt_boxes = tf.reshape(gt_boxes, [-1, 4])
        gt_classes = tf.reshape(gt_classes, [-1])
        # normalize (y1, x1, y2, x2) => 0~1
        gt_boxes = gt_boxes / tf.cast(tf.concat([image_size, image_size], axis=0), tf.float32)
        # [2k, 4] with [7, 4] => [2k, 7] overlop scores
        overlaps = ops.mx_compute_overlap(proposals, gt_boxes)

        gt_assignment = tf.argmax(overlaps, axis=1)
        roi_iou_max = tf.reduce_max(overlaps, axis=1)  # [2000]get clost gt boxes overlop score for each anchor boxes
        target_classes = tf.gather(gt_classes, gt_assignment)

        fg_idx = tf.where(roi_iou_max >= self.pos_iou_thr)[:, 0]
        bg_idx = tf.where(roi_iou_max < self.neg_iou_thr)[:, 0]

        num_fg = tf.cast(tf.constant(self.batch_size * self.pos_frac), dtype=tf.int32)
        num_fg_fact = fg_idx.shape[0]
        num_fg_rois = tf.cond(tf.less(num_fg, num_fg_fact), lambda: num_fg, lambda: num_fg_fact)

        if num_fg_fact > 0:
            fg_idx = tf.random.shuffle(fg_idx)[:num_fg_rois]

        num_bg = tf.cast(tf.constant(self.batch_size), dtype=tf.int32) - num_fg_rois
        num_bg_fact = bg_idx.shape[0]
        num_bg_rois = tf.cond(tf.less(num_bg, num_bg_fact), lambda: num_bg, lambda: num_bg_fact)

        if num_bg_fact > 0:
            bg_idx = tf.random.shuffle(bg_idx)[:num_bg_rois]

        keep_idx = tf.concat([fg_idx, bg_idx], axis=0)
        target_classes = tf.gather(target_classes, keep_idx)  # 获取选中的正负样本

        bg_rois_idx = tf.range(fg_idx.shape[0], tf.constant(keep_idx.shape[0], dtype=tf.int32))
        target_classes = tf.compat.v1.scatter_update(tf.Variable(target_classes), bg_rois_idx, 0)

        rois = tf.gather(proposals, keep_idx)

        target_boxes = tf.gather(gt_boxes, gt_assignment)
        target_rois = tf.gather(target_boxes, keep_idx)

        target_bbox = ops.mx_encode_bbox(rois, target_rois)

        target_classes = tf.stop_gradient(target_classes)  # [34+102]
        target_bbox = tf.stop_gradient(target_bbox)
        rois = tf.stop_gradient(rois)
        return target_classes, target_bbox, rois

    def rcnn_acc(self, rcnn_class_prob, target_classes):
        '''

        :param rcnn_class_prob:
        :param target_classes:
        :return:
        '''
        target_classes = tf.cast(target_classes, tf.float32)
        # 获取预测结果
        class_pred = tf.cast(tf.argmax(rcnn_class_prob, axis=1), tf.float32)

        rcnn_class_acc = tf.reduce_mean(tf.cast(tf.equal(target_classes, class_pred), tf.float32))

        return rcnn_class_acc

    def rcnn_class_loss(self, rcnn_class_score, rcnn_classes_target):
        '''

        :param rcnn_class_score:
        :param rcnn_classes_target:
        :return:
        '''

        loss = keras.losses.categorical_crossentropy(tf.one_hot(rcnn_classes_target, depth=self.num_classes),
                                                     rcnn_class_score, from_logits=True)
        return tf.reduce_mean(loss)

    def rcnn_location_loss(self, rcnn_bbox_pred, target_bbox, target_classes):
        '''

        :param rcnn_bbox_pred:
        :param target_bbox:
        :return:
        '''
        rcnn_bbox_pred = tf.reshape(rcnn_bbox_pred, [-1, self.num_classes, 4])

        # 取出正样本角标
        pos_idx = tf.where(target_classes > 0)

        # 取出正样本标签
        pos_rois_label = tf.gather(target_classes, pos_idx[:, 0])
        pos_rois_label = tf.cast(pos_rois_label, tf.int64)
        #
        rcnn_bbox_pred_pos_idx = tf.concat([pos_idx, tf.reshape(pos_rois_label, [-1, 1])], axis=1)

        pos_rois_box = tf.gather(target_bbox, pos_idx[:, 0])
        rcnn_bbox_pred_pos = tf.gather_nd(rcnn_bbox_pred, rcnn_bbox_pred_pos_idx)

        loss = ops.smooth_l1_loss(pos_rois_box, rcnn_bbox_pred_pos)
        if pos_idx.shape[0]:
            loss = tf.reduce_mean(loss)
        else:
            loss = tf.constant(0, dtype=tf.float32)
        return loss

    def detect_image(self, rcnn_class_prob, rcnn_bbox_pred, rois):
        '''

        :param rcnn_class_prob: [-1, num_classes]
        :param rcnn_bbox_pred: [-1, num_classes * 4]
        :param rois: [-1, 4]
        :return:
        '''
        rois = tf.stop_gradient(rois)
        rcnn_class_prob = tf.stop_gradient(rcnn_class_prob)
        rcnn_bbox_pred = tf.reshape(rcnn_bbox_pred, [-1, self.num_classes, 4])
        rcnn_bbox_pred = tf.stop_gradient(rcnn_bbox_pred)


        best_classes = tf.argmax(rcnn_class_prob, axis=1)
        best_boxes_idx = tf.concat([tf.reshape(tf.range(0, rcnn_bbox_pred.shape[0]), [-1, 1]) ,
                                    tf.cast(tf.reshape(best_classes, [-1,1]), tf.int32)],
                                   axis=1)
        best_boxes = tf.gather_nd(rcnn_bbox_pred, best_boxes_idx)

        decode_rcnn_bbox = ops.mx_decode_bbox(rois, best_boxes)
        window = tf.constant([0, 0, 1, 1], dtype=tf.float32)
        decode_rcnn_bbox = ops.bbox_clip(decode_rcnn_bbox, window) # [num_rois, 4]



        rcnn_class_prob = rcnn_class_prob[:, 1:]

        best_prob = tf.reduce_max(rcnn_class_prob, axis=1)
        # print('best_prob:', best_prob)
        get_value_idx = tf.where(best_prob > self.min_confidence)


        get_value_prob = tf.gather(rcnn_class_prob, get_value_idx[:, 0])
        get_value_box = tf.gather(decode_rcnn_bbox, get_value_idx[:, 0])
        rcnn_probs = tf.reduce_max(get_value_prob, axis=1)


        # 3. get top N to NMS
        pre_nms_topN = min(self.max_instances, get_value_box.shape[0])
        ix = tf.nn.top_k(rcnn_probs, pre_nms_topN, sorted=True).indices  # 先从大到小排序，取出前pre_nms_topN值，并返回索引
        # 获取前pre_nms_topN个rpn_probs，
        get_value_prob = tf.gather(get_value_prob, ix)
        rcnn_probs = tf.gather(rcnn_probs, ix)
        get_value_box = tf.gather(get_value_box, ix)
        # anchors = tf.gather(anchors, ix)

        # NMS, indices: [2000]
        indices = tf.image.non_max_suppression(
            get_value_box, rcnn_probs, self.max_instances, self.nms_threshold)

        final_box = tf.gather(get_value_box, indices)
        get_value_prob = tf.gather(get_value_prob, indices)
        final_class = tf.argmax(get_value_prob, axis=1)
        final_score = tf.reduce_max(get_value_prob,axis=1)

        return final_class, final_box, final_score
