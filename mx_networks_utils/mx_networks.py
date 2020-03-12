import numpy as np
import tensorflow as tf
# 导入 keras 模型，不能使用 import keras，它导入的是标准的 Keras 库
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import argparse, os, time, sys

sys.path.append('../')
import models.resnet as resnet
import mx_networks_utils.mx_ops as mx_ops


class mx_BE_Generator(keras.Model):
    def __init__(self, filter_num):
        super(mx_BE_Generator, self).__init__()
        self.filter_num = filter_num
        self.fc = layers.Dense(filter_num * 8 * 8)

        self.module1 = mx_BE_Encoder_Module(filter_num)

        self.module2 = mx_BE_Encoder_Module(filter_num)

        self.module3 = mx_BE_Encoder_Module(filter_num)

        self.module4 = mx_BE_Encoder_Module(filter_num)

        self.module5 = mx_BE_Encoder_Module(filter_num)

        self.module6 = mx_BE_Encoder_Module(filter_num)

        self.to_image = layers.Conv2D(3, (3, 3), strides=1, padding='same', activation='tanh')

    def call(self, z):
        x = self.fc(z)
        x = tf.reshape(x, [-1, 8, 8, self.filter_num])
        x = self.module1(x)
        x = self.mx_scale(x)
        x = self.module2(x)
        x = self.mx_scale(x)
        x = self.module3(x)
        x = self.mx_scale(x)
        x = self.module4(x)
        x = self.mx_scale(x)
        x = self.module5(x)
        x = self.mx_scale(x)
        x = self.module6(x)
        x = self.to_image(x)
        return x

    def mx_scale(self, x):
        h = x.shape[1] * 2
        w = x.shape[2] * 2
        x = tf.image.resize(x, [h, w])
        return x


class mx_BE_Discriminator(keras.Model):
    def __init__(self, filter_num):
        super(mx_BE_Discriminator, self).__init__()
        self.filter_num = filter_num
        self.from_image = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same', activation='elu')

        self.module1 = mx_BE_Encoder_Module(filter_num * 1)
        self.conv1 = layers.Conv2D(filter_num * 1, (3, 3), strides=2, padding='same', activation='elu')

        self.module2 = mx_BE_Encoder_Module(filter_num * 2)
        self.conv2 = layers.Conv2D(filter_num * 2, (3, 3), strides=2, padding='same', activation='elu')

        self.module3 = mx_BE_Encoder_Module(filter_num * 3)
        self.conv3 = layers.Conv2D(filter_num * 3, (3, 3), strides=2, padding='same', activation='elu')

        self.module4 = mx_BE_Encoder_Module(filter_num * 4)
        self.conv4 = layers.Conv2D(filter_num * 4, (3, 3), strides=2, padding='same', activation='elu')

        self.module5 = mx_BE_Encoder_Module(filter_num * 5)
        self.conv5 = layers.Conv2D(filter_num * 5, (3, 3), strides=2, padding='same', activation='elu')

        self.module6 = mx_BE_Encoder_Module(filter_num * 6)

        self.fc1 = layers.Dense(8 * 8 * filter_num * 6)

        self.Emodule1 = mx_BE_Encoder_Module(filter_num)

        self.Emodule2 = mx_BE_Encoder_Module(filter_num)

        self.Emodule3 = mx_BE_Encoder_Module(filter_num)

        self.Emodule4 = mx_BE_Encoder_Module(filter_num)

        self.Emodule5 = mx_BE_Encoder_Module(filter_num)

        self.Emodule6 = mx_BE_Encoder_Module(filter_num)

        self.to_image = layers.Conv2D(3, (3, 3), strides=1, padding='same', activation='tanh')

    def call(self, image):
        x = self.from_image(image)
        x = self.module1(x)
        x = self.conv1(x)
        x = self.module2(x)
        x = self.conv2(x)
        x = self.module3(x)
        x = self.conv3(x)
        x = self.module4(x)
        x = self.conv4(x)
        x = self.module5(x)
        x = self.conv5(x)
        x = self.module6(x)

        x = tf.reshape(x, [-1, 8, 8, self.filter_num * 6])
        x = self.fc1(x)

        x = self.Emodule1(x)
        x = self.mx_scale(x)
        x = self.Emodule2(x)
        x = self.mx_scale(x)
        x = self.Emodule3(x)
        x = self.mx_scale(x)
        x = self.Emodule4(x)
        x = self.mx_scale(x)
        x = self.Emodule5(x)
        x = self.mx_scale(x)
        x = self.Emodule6(x)
        x = self.to_image(x)

        return x

    def mx_scale(self, x):
        h = x.shape[1] * 2
        w = x.shape[2] * 2
        x = tf.image.resize(x, [h, w])
        return x


class mx_BE_Encoder_Module(layers.Layer):
    def __init__(self, filter_num):
        super(mx_BE_Encoder_Module, self).__init__()

        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same', activation='elu')
        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same', activation='elu')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x


class Faster_RCNN(keras.Model):
    def __init__(self, cfg,
                 NUM_CLASSES,
                 ANCHOR_SCALES,
                 ANCHOR_RATIOS,
                 PRN_BATCH_SIZE,
                 RPN_POS_FRAC,
                 RPN_POS_IOU_THR,
                 RPN_NEG_IOU_THR,
                 PRN_PROPOSAL_COUNT,
                 PRN_NMS_THRESHOLD,
                 POOL_SIZE,
                 RCNN_BATCH_SIZE,
                 RCNN_POS_FRAC,
                 RCNN_POS_IOU_THR,
                 RCNN_NEG_IOU_THR,
                 RCNN_MIN_CONFIDENCE,
                 RCNN_NME_THRESHOLD,
                 RCNN_MAX_INSTANCES):
        super(Faster_RCNN, self).__init__()
        self.cfg = cfg
        self.TRAINING = cfg.is_train
        self.NUM_CLASSES = NUM_CLASSES
        self.ANCHOR_SCALES = ANCHOR_SCALES
        self.ANCHOR_RATIOS = ANCHOR_RATIOS
        self.PRN_BATCH_SIZE = PRN_BATCH_SIZE
        self.RPN_POS_FRAC = RPN_POS_FRAC
        self.RPN_POS_IOU_THR = RPN_POS_IOU_THR
        self.RPN_NEG_IOU_THR = RPN_NEG_IOU_THR
        self.PRN_PROPOSAL_COUNT = PRN_PROPOSAL_COUNT
        self.PRN_NMS_THRESHOLD = PRN_NMS_THRESHOLD
        self.POOL_SIZE = POOL_SIZE
        self.RCNN_BATCH_SIZE = RCNN_BATCH_SIZE
        self.RCNN_POS_FRAC = RCNN_POS_FRAC
        self.RCNN_POS_IOU_THR = RCNN_POS_IOU_THR
        self.RCNN_NEG_IOU_THR = RCNN_NEG_IOU_THR
        self.RCNN_MIN_CONFIDENCE = RCNN_MIN_CONFIDENCE
        self.RCNN_NME_THRESHOLD = RCNN_NME_THRESHOLD
        self.RCNN_MAX_INSTANCES = RCNN_MAX_INSTANCES

        self.backbone = resnet.resnet101(training=self.TRAINING)
        self.neck = FPN(256)
        self.rpn_head = RPNHead(anchor_scales=self.ANCHOR_SCALES,
                                anchor_ratios=self.ANCHOR_RATIOS,
                                proposal_count=self.PRN_PROPOSAL_COUNT,
                                nms_threshold=self.PRN_NMS_THRESHOLD,
                                num_rpn_deltas=self.PRN_BATCH_SIZE,
                                positive_fraction=self.RPN_POS_FRAC,
                                pos_iou_thr=self.RPN_POS_IOU_THR,
                                neg_iou_thr=self.RPN_NEG_IOU_THR)
        self.rcnn_head = RCNNHead(num_classes=self.NUM_CLASSES,
                                  pool_size=self.POOL_SIZE,
                                  min_confidence=self.RCNN_MIN_CONFIDENCE,
                                  nms_threshold=self.RCNN_NME_THRESHOLD,
                                  max_instances=self.RCNN_MAX_INSTANCES,
                                  training=self.TRAINING)

    def call(self, inputs):
        num_gt, batch_image_real, boxes, labels, img_width, img_height, batch_z = inputs
        x = self.backbone(batch_image_real)
        feature_map = self.neck(x)

        # rpn
        rpn_probs, rpn_deltas = self.rpn_head(feature_map)
        self.feature_shape = self.rpn_head.feature_shape
        # rcnn
        proposals = self.get_proposal(rpn_probs, rpn_deltas)  # 坐标归一化了
        self.rois, self.target_matchs, self.target_deltas = self.build_fast_rcnn_targer(proposals, boxes, labels,
                                                                         [self.cfg.img_size[0],
                                                                          self.cfg.img_size[1]])
        cropped_roi_features = self.roi_pooling(self.rois, feature_map)
        rcnn_logits, rcnn_deltas = self.rcnn_head(cropped_roi_features)


        return rpn_probs, rpn_deltas, rcnn_logits, rcnn_deltas

    def loss_function(self, rpn_probs, rpn_deltas, gt_boxes, gt_classes, rcnn_logits, rcnn_deltas):
        rpn_class_loss, rpn_location_loss = self.rpn_head.rpn_loss(rpn_probs=rpn_probs, rpn_deltas=rpn_deltas,
                                                                   gt_boxes=gt_boxes,
                                                                   gt_classes=gt_classes,
                                                                   feature_shape=self.feature_shape,
                                                                   image_size=tf.constant(
                                                                       [self.cfg.img_size[0], self.cfg.img_size[1]]))
        rcnn_class_loss, rcnn_location_loss = self.rcnn_head.rcnn_loss(rcnn_logits, rcnn_deltas, self.target_matchs, self.target_deltas)
        return rpn_class_loss, rpn_location_loss, rcnn_class_loss, rcnn_location_loss

    def get_proposal(self, rpn_probs, rpn_deltas):
        proposal = self.rpn_head.get_proposals(rpn_probs, rpn_deltas, self.feature_shape,
                                               [self.cfg.img_size[0], self.cfg.img_size[1]])
        return proposal

    def build_fast_rcnn_targer(self, proposals, gt_boxes, gt_classes, image_size):
        '''

        :param proposals: [num_proposals, (y1, x1, y2, x2)] in normalized coordinates.
        :param gt_boxes: [-1， num_gt_boxes, (y1, x1, y2, x2)]
        :param gt_classes: [-1，num_gt_boxes]
        :param image_size: list [h,w]
        :return:
            rois: [浮动的数, (y1, x1, y2, x2)]
            target_matchs: [浮动的数], 不够填0
            target_deltas: [num_positive_rois, (dy, dx, log(dh), log(dw))]
        '''
        H, W = image_size
        gt_boxes = tf.reshape(gt_boxes, [-1, 4])
        gt_classes = tf.reshape(gt_classes, [-1])
        # normalize (y1, x1, y2, x2) => 0~1
        gt_boxes = gt_boxes / tf.constant([H, W, H, W], dtype=tf.float32)
        # [2k, 4] with [7, 4] => [2k, 7] overlop scores
        overlaps = mx_ops.mx_compute_overlap(proposals, gt_boxes)

        roi_iou_max = tf.reduce_max(overlaps, axis=1)  # [2000]get clost gt boxes overlop score for each anchor boxes

        # roi_iou_max: [2000],
        positive_roi_bool = (roi_iou_max >= self.RCNN_POS_IOU_THR)  # [2000]
        positive_indices = tf.where(positive_roi_bool)[:, 0]  # [48, 1] =>[48]
        # get all positive indices, namely get all pos_anchor indices
        negative_indices = tf.where(roi_iou_max < self.RCNN_NEG_IOU_THR)[:, 0]

        # get all negative anchor indices
        # Subsample ROIs. Aim for 33% positive
        # Positive ROIs
        positive_count = int(self.RCNN_BATCH_SIZE * self.RCNN_POS_FRAC)
        positive_indices = tf.random.shuffle(positive_indices)[:positive_count]  # [256*0.25]=64, at most get 64
        positive_count = tf.shape(positive_indices)[0]

        # Negative ROIs. Add enough to maintain positive:negative ratio.
        r = 1.0 / self.RCNN_POS_FRAC
        negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count  # 102
        # negative_count = self.RCNN_BATCH_SIZE - positive_count
        negative_indices = tf.random.shuffle(negative_indices)[:negative_count]  # [102]

        # Gather selected ROIs, based on remove redundant pos/neg indices
        positive_rois = tf.gather(proposals, positive_indices)  # [34, 4]
        negative_rois = tf.gather(proposals, negative_indices)  # [102, 4]

        # Assign positive ROIs to GT boxes.
        positive_overlaps = tf.gather(overlaps, positive_indices)  # [34, 7]
        roi_gt_box_assignment = tf.argmax(positive_overlaps, axis=1)  # [34]for each anchor, get its clost gt boxes
        roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)  # [34, 4]
        target_matchs = tf.gather(gt_classes, roi_gt_box_assignment)  # [34]
        # target_matchs, target_deltas all get!!
        # proposal: [34, 4], target: [34, 4]
        target_deltas = mx_ops.mx_encode_bbox2delta(positive_rois, roi_gt_boxes)
        # [34, 4] [102, 4]
        rois = tf.concat([positive_rois, negative_rois], axis=0)

        N = tf.shape(negative_rois)[0]  # 102
        target_matchs = tf.pad(target_matchs, [(0, N)])  # [34] padding after with [N]

        target_matchs = tf.stop_gradient(target_matchs)  # [34+102]
        target_deltas = tf.stop_gradient(target_deltas)  # [34, 4]
        # rois: [34+102, 4]
        return rois, target_matchs, target_deltas

    def roi_pooling(self, rois, feature_maps):
        '''

        :param rois: [浮动的数, (y1, x1, y2, x2)]
        :param feature_maps: [bs, 26,26,256]
        :return:
        '''
        N = rois.shape.as_list()[0]
        cropped_roi_features = tf.image.crop_and_resize(image=feature_maps, boxes=rois,
                                                        box_indices=tf.zeros(shape=[N, ], dtype=tf.int32),
                                                        crop_size=[14, 14])
        return cropped_roi_features

    def get_detection_boxes(self, rcnn_logits, rcnn_deltas, image_size):
        '''

        :param rcnn_logits:
        :param rcnn_deltas:
        :param image_size:
        :return:
        '''
        H, W = image_size
        # Class IDs per ROI
        class_ids = tf.argmax(rcnn_logits, axis=1, output_type=tf.int32)
        # Class probability of the top class of each ROI
        indices = tf.stack([tf.range(rcnn_logits.shape[0]), class_ids], axis=1)
        class_scores = tf.gather_nd(rcnn_logits, indices)

        # Class-specific bounding box deltas
        deltas_specific = tf.gather_nd(rcnn_deltas, indices)

        # Apply bounding box deltas
        # Shape: [num_rois, (y1, x1, y2, x2)] in normalized coordinates
        refined_rois = mx_ops.mx_decode_bbox2delta(self.rois, deltas_specific)

        # Clip boxes to image window
        refined_rois *= tf.constant([H, W, H, W], dtype=tf.float32)
        window = tf.constant([0., 0., H * 1., W * 1.], dtype=tf.float32)
        refined_rois = mx_ops.bbox_clip(refined_rois, window)

        # Filter out background boxes
        keep = tf.where(class_ids > 0)[:, 0]
        # Filter out low confidence boxes
        if self.RCNN_MIN_CONFIDENCE:
            conf_keep = tf.where(class_scores >= self.RCNN_MIN_CONFIDENCE)[:, 0]
            keep = tf.compat.v2.sets.intersection(tf.expand_dims(keep, 0),
                                                  tf.expand_dims(conf_keep, 0))
            keep = tf.sparse.to_dense(keep)[0]

        # Apply per-class NMS
        # 1. Prepare variables
        pre_nms_class_ids = tf.gather(class_ids, keep)
        pre_nms_scores = tf.gather(class_scores, keep)
        pre_nms_rois = tf.gather(refined_rois, keep)
        unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

        def nms_keep_map(class_id):
            '''Apply Non-Maximum Suppression on ROIs of the given class.'''
            # Indices of ROIs of the given class
            ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
            # Apply NMS
            class_keep = tf.image.non_max_suppression(
                    tf.gather(pre_nms_rois, ixs),
                    tf.gather(pre_nms_scores, ixs),
                    max_output_size=self.RCNN_MAX_INSTANCES,
                    iou_threshold=self.RCNN_NME_THRESHOLD)
            # Map indices
            class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
            return class_keep

        # 2. Map over class IDs
        nms_keep = tf.constant([], dtype=tf.float32)
        for i in range(unique_pre_nms_class_ids.shape[0]):
            nms_keep = tf.concat([nms_keep, tf.cast(nms_keep_map(unique_pre_nms_class_ids[i]), tf.float32)], axis=0)

        # 3. Compute intersection between keep and nms_keep
        keep = tf.cast(keep, tf.int64)
        nms_keep = tf.cast(nms_keep, tf.int64)
        keep = tf.compat.v2.sets.intersection(tf.expand_dims(keep, 0),
                                              tf.expand_dims(nms_keep, 0))
        keep = tf.sparse.to_dense(keep)[0]
        # Keep top detections
        roi_count = self.RCNN_MAX_INSTANCES
        class_scores_keep = tf.gather(class_scores, keep)
        num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
        top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
        keep = tf.gather(keep, top_ids)

        detections = tf.concat([
            tf.gather(refined_rois, keep),
            tf.cast(tf.gather(class_ids, keep), tf.float32)[..., tf.newaxis],
            tf.gather(class_scores, keep)[..., tf.newaxis]
        ], axis=1)

        return detections



class FPN(keras.Model):
    '''
        Feature Pyramid Networks
    '''

    def __init__(self, filter_num):
        super(FPN, self).__init__()

        self.filter_num = filter_num

        self.fpn_1 = layers.Conv2D(filter_num, (1, 1), strides=(1, 1), kernel_initializer='he_normal',
                                   padding='valid')

        self.fpn_2 = layers.Conv2D(filter_num, (3, 3), strides=(1, 1), kernel_initializer='he_normal', padding='same')

    def call(self, inputs):
        x = self.fpn_1(inputs)
        x = self.fpn_2(x)  # [bs, 26,26,256]
        return x


class RPNHead(keras.Model):

    def __init__(self,
                 anchor_scales,
                 anchor_ratios,
                 proposal_count,
                 nms_threshold,
                 num_rpn_deltas,
                 positive_fraction,
                 pos_iou_thr,
                 neg_iou_thr):
        super(RPNHead, self).__init__()
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold
        self.num_rpn_deltas = num_rpn_deltas
        self.positive_fraction = positive_fraction
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        # Shared convolutional base of the RPN
        self.rpn_conv_shared = layers.Conv2D(512, (3, 3), padding='same',
                                             kernel_initializer='he_normal',
                                             name='rpn_conv_shared')

        self.rpn_class_raw = layers.Conv2D(len(anchor_ratios) * len(anchor_scales) * 2, (1, 1),
                                           kernel_initializer='he_normal',
                                           name='rpn_class_raw')

        self.rpn_delta_pred = layers.Conv2D(len(anchor_ratios) * len(anchor_scales) * 4, (1, 1),
                                            kernel_initializer='he_normal',
                                            name='rpn_bbox_pred')

    def call(self, inputs):
        '''

        :param inputs:
        :return:
        '''

        shared = self.rpn_conv_shared(inputs)
        self.feature_shape = shared.shape[1:3]
        shared = tf.nn.relu(shared)

        # class output
        # rpn_probs: shape----> [batch, num_anchors, 2]
        x = self.rpn_class_raw(shared)
        rpn_class_logits = tf.reshape(x, [tf.shape(x)[0], -1, 2])
        rpn_probs = tf.nn.softmax(rpn_class_logits)

        # center, size output
        # rpn_deltas: shape----> [batch, num_anchors, 4]
        x = self.rpn_delta_pred(shared)
        rpn_deltas = tf.reshape(x, [tf.shape(x)[0], -1, 4])

        return rpn_probs, rpn_deltas

    def rpn_loss(self, rpn_probs, rpn_deltas, gt_boxes, gt_classes, feature_shape, image_size):
        '''

        :param rpn_probs: [-1, num_anchors, 2]
        :param rpn_deltas: [-1, num_anchors, 4]
        :param gt_boxes: [-1, num_gt, 4]
        :param gt_classes: [-1, num_gt]
        :param feature_shape: 特征层的大小，tensor, [h, w]
        :param image_size: 输入网络图像的大小，tensor,  [h, w]
        :return:
        '''
        anchor_boxes = mx_ops.mx_generate_anchors(feature_shape, self.anchor_scales, self.anchor_ratios, image_size)
        target_matchs, target_deltas = mx_ops.build_target(anchor_boxes, gt_boxes,
                                                           gt_classes, self.neg_iou_thr,
                                                           self.pos_iou_thr, self.num_rpn_deltas,
                                                           self.positive_fraction)
        class_loss = self.rpn_class_loss(rpn_probs, target_matchs)
        location_loss = self.rpn_location_loss(rpn_deltas, target_deltas, target_matchs)

        return class_loss, location_loss

    def rpn_class_loss(self, rpn_probs, target_matchs):
        '''

        :param rpn_probs: rpn_probs: [batch_size, num_anchors, 2], batch_size=1
        :param target_matchs: [num_anchors]  1=positive, -1=negative, 0=neutral anchor.
        :return:
        '''

        # Get anchor classes. Convert the -1/+1 match to 0/1 values.
        # 将 -1=negative, 0=neutral 全部变成0
        anchor_class = tf.cast(tf.equal(target_matchs, 1), tf.int32)

        # Positive and Negative anchors contribute to the loss,
        # but neutral anchors (match value = 0) don't.
        # 除去0=neutral anchor，获取正负样本的索引
        indices = tf.where(tf.not_equal(target_matchs, 0))
        # 通过所以取出正负样本
        anchor_class = tf.gather(anchor_class, tf.reshape(indices, [-1]))
        # 通过所以取出正负样本
        rpn_probs = tf.reshape(rpn_probs, [-1, 2])
        rpn_class_logits = tf.gather(rpn_probs, tf.reshape(indices, [-1]))
        num_classes = rpn_class_logits.shape[-1]
        loss = keras.losses.categorical_crossentropy(tf.one_hot(anchor_class, depth=num_classes),
                                                     rpn_class_logits, from_logits=True)
        return tf.reduce_sum(loss)

    def rpn_location_loss(self, rpn_deltas, target_deltas, target_matchs):
        '''

        :param rpn_deltas: [batch_size, num_anchors, 4]， 网络的输出
        :param target_deltas: [样本数,4]
        :param target_matchs: [num_anchors]  1=positive, -1=negative, 0=neutral anchor.
        :return:
        '''
        # Positive anchors contribute to the loss, but negative and
        # neutral anchors (match value of 0 or -1) don't.
        indices = tf.where(tf.equal(target_matchs, 1))

        # Pick bbox deltas that contribute to the loss
        rpn_deltas = tf.reshape(rpn_deltas, [-1, 4])
        rpn_deltas = tf.gather_nd(rpn_deltas, indices)

        num_postive = rpn_deltas.shape.as_list()[0]
        target_deltas = target_deltas[:num_postive, :]
        loss = mx_ops.smooth_l1_loss(target_deltas, rpn_deltas)
        return tf.reduce_sum(loss)

    def get_proposals(self, rpn_probs, rpn_deltas, feature_shape, image_size):
        '''

        :param rpn_probs: [-1, num_anchors, 2]
        :param rpn_deltas: [-1, num_anchors, 4]
        :param feature_shape: 特征层的大小，tensor, [h, w]， eg. feature_shape=[32, 16]
        :param image_size: list [h, w]
        :return:
        '''
        # [num_anchors, 4]
        anchors = mx_ops.mx_generate_anchors(feature_shape, self.anchor_scales, self.anchor_ratios,
                                             tf.constant(image_size))
        # [369303, 4], [b, 11]
        # [b, N, (background prob, foreground prob)], get anchor's foreground prob, [1, 369303]
        rpn_probs = tf.reshape(rpn_probs, [-1, 2])
        rpn_probs = rpn_probs[:, 1]
        rpn_deltas = tf.reshape(rpn_probs, [-1, 4])

        H, W = image_size

        # Improve performance
        pre_nms_limit = min(6000, anchors.shape[0])  # min(6000, 215169) => 6000
        ix = tf.nn.top_k(rpn_probs, pre_nms_limit, sorted=True).indices
        # [215169] => [6000], respectively
        rpn_probs = tf.gather(rpn_probs, ix)
        rpn_deltas = tf.gather(rpn_deltas, ix)
        anchors = tf.gather(anchors, ix)

        # Get refined anchors, => [6000, 4]
        proposals = mx_ops.mx_decode_bbox2delta(anchors, rpn_deltas)
        # clipping to valid area, [6000, 4]
        window = tf.constant([0., 0., H, W], dtype=tf.float32)
        proposals = mx_ops.bbox_clip(proposals, window)

        # Normalize, (y1, x1, y2, x2)
        proposals = proposals / tf.constant([H, W, H, W], dtype=tf.float32)

        # NMS, indices: [2000]
        indices = tf.image.non_max_suppression(
            proposals, rpn_probs, self.proposal_count, self.nms_threshold)
        proposals = tf.gather(proposals, indices)  # [2000, 4]

        # if True:
        #     proposal_probs = tf.expand_dims(tf.gather(rpn_probs, indices), axis=1)
        #     proposals = tf.concat([proposals, proposal_probs], axis=1)

        return proposals  # [num_proposals, (y1, x1, y2, x2)]


class RCNNHead(keras.Model):
    def __init__(self, num_classes,
                 pool_size,
                 min_confidence,
                 nms_threshold,
                 max_instances,
                 training):
        super(RCNNHead, self).__init__()

        self.num_classes = num_classes
        self.pool_size = tuple(pool_size)
        self.min_confidence = min_confidence
        self.nms_threshold = nms_threshold
        self.max_instances = max_instances
        self.training = training
        self.rcnn_class_conv1 = layers.Conv2D(1024, self.pool_size,
                                              padding='valid', name='rcnn_class_conv1')

        self.rcnn_class_bn1 = layers.BatchNormalization(name='rcnn_class_bn1')

        self.rcnn_class_conv2 = layers.Conv2D(1024, (1, 1),
                                              name='rcnn_class_conv2')

        self.rcnn_class_bn2 = layers.BatchNormalization(name='rcnn_class_bn2')

        self.rcnn_class_logits = layers.Dense(num_classes, name='rcnn_class_logits')

        self.rcnn_delta_fc = layers.Dense(num_classes * 4, name='rcnn_bbox_fc')

    def call(self, inputs):
        '''

        :param inputs: [num_rois, 14, 14, 256]
        :return:
        '''

        x = self.rcnn_class_conv1(inputs)
        x = self.rcnn_class_bn1(x, training=self.training)
        x = tf.nn.relu(x)

        x = self.rcnn_class_conv2(x)
        x = self.rcnn_class_bn2(x, training=self.training)
        x = tf.nn.relu(x)

        x = tf.reshape(x, [x.shape[0], -1])

        logits = self.rcnn_class_logits(x)
        probs = tf.nn.softmax(logits)

        deltas = self.rcnn_delta_fc(x)
        deltas = tf.reshape(deltas, (-1, self.num_classes, 4))

        return probs, deltas
    def rcnn_loss(self, logits, deltas, target_matchs, target_deltas):
        '''

        :param logits: [num_rois, num_class] 0: bg
        :param deltas: [num_rois, num_rois, 4]
        :param rois:
        :param target_matchs:
        :param target_deltas:
        :return:
        '''
        # class loss
        target_matchs = tf.cast(target_matchs, dtype=tf.int64)
        num_classes = logits.shape[-1]
        class_loss = keras.losses.categorical_crossentropy(tf.one_hot(target_matchs, depth=num_classes),
                                                     logits, from_logits=True)
        class_loss = tf.reduce_mean(class_loss) if tf.size(class_loss) > 0 else tf.constant(0.0)

        # location loss
        positive_roi_ix = tf.where(target_matchs > 0)[:, 0]
        positive_roi_class_ids = tf.gather(target_matchs, positive_roi_ix)

        indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

        # Gather the deltas (predicted and true) that contribute to loss
        rcnn_deltas = tf.gather_nd(deltas, indices)

        # Smooth-L1 Loss
        location_loss = mx_ops.smooth_l1_loss(target_deltas, rcnn_deltas)
        location_loss = tf.reduce_mean(location_loss) if tf.size(location_loss) > 0 else tf.constant(0.0)
        return class_loss, location_loss

def main():
    filter_num = 128
    z = tf.random.normal([4, 128])

    G = mx_BE_Generator(filter_num)
    G.build(input_shape=(None, 128))

    D = mx_BE_Discriminator(filter_num)
    D.build(input_shape=(None, 64, 64, 3))

    print(G.summary())
    print(D.summary())
    img = G(z)
    eimg = D(img)
    print(img.shape)
    print(eimg.shape)


if __name__ == '__main__':
    main()
