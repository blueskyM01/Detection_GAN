import numpy as np
import tensorflow as tf
# 导入 keras 模型，不能使用 import keras，它导入的是标准的 Keras 库
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import argparse, os, time, sys, datetime, cv2

sys.path.append('../')
import mx_Dataset.mx_load_dataset as mx_data_loader
import mx_networks_utils.mx_ops as mx_ops
import mx_networks_utils.mx_utils as mx_utils
import mx_networks_utils.mx_networks as mx_net


class FasterRCNN:
    def __init__(self, cfg):
        self.cfg = cfg

        data_loader = mx_data_loader.mx_DatasetLoader(self.cfg.img_size)
        self.dataset, self.dataset_len = data_loader.mx_dataset_load(self.cfg.dataset_dir, self.cfg.dataset_name,
                                                                     label_dir=self.cfg.label_dir,
                                                                     label_name=self.cfg.label_name,
                                                                     shuffle=True, shuffle_size=1000,
                                                                     batch_size=self.cfg.batch_size,
                                                                     epoch=self.cfg.epoch)
        self.db_train = iter(self.dataset)
        self.classes = mx_utils.get_classes(self.cfg.class_path)
        self.num_classes = len(self.classes) + 1  # '0': backgroud

        # RPN configuration
        # Anchor attributes
        self.ANCHOR_SCALES = (128, 256, 512)
        self.ANCHOR_RATIOS = (0.5, 1, 2)

        # RPN training configuration
        self.PRN_BATCH_SIZE = 256
        self.RPN_POS_FRAC = 0.5
        self.RPN_POS_IOU_THR = 0.7
        self.RPN_NEG_IOU_THR = 0.3

        # ROIs kept configuration
        self.PRN_PROPOSAL_COUNT = 2000
        self.PRN_NMS_THRESHOLD = 0.7

        # RCNN configuration
        # Bounding box refinement mean and standard deviation
        self.RCNN_TARGET_MEANS = (0., 0., 0., 0.)
        self.RCNN_TARGET_STDS = (0.1, 0.1, 0.2, 0.2)

        # ROI Feat Size
        self.POOL_SIZE = (7, 7)

        # RCNN training configuration
        self.RCNN_BATCH_SIZE = 256
        self.RCNN_POS_FRAC = 0.25
        self.RCNN_POS_IOU_THR = 0.5
        self.RCNN_NEG_IOU_THR = 0.5

        # Boxes kept configuration
        self.RCNN_MIN_CONFIDENCE = 0.7
        self.RCNN_NME_THRESHOLD = 0.3
        self.RCNN_MAX_INSTANCES = 100

    def build_model(self):
        # 创建log文件
        summary_writer = tf.summary.create_file_writer(os.path.join(self.cfg.results_dir,
                                                                    self.cfg.log_dir,
                                                                    self.cfg.tmp_result_name))

        with summary_writer.as_default():

            self.faster_rcnn = mx_net.Faster_RCNN(cfg=self.cfg,
                                                  NUM_CLASSES=self.num_classes,
                                                  ANCHOR_SCALES=self.ANCHOR_SCALES,
                                                  ANCHOR_RATIOS=self.ANCHOR_RATIOS,
                                                  PRN_BATCH_SIZE=self.PRN_BATCH_SIZE,
                                                  RPN_POS_FRAC=self.RPN_POS_FRAC,
                                                  RPN_POS_IOU_THR=self.RPN_POS_IOU_THR,
                                                  RPN_NEG_IOU_THR=self.RPN_NEG_IOU_THR,
                                                  PRN_PROPOSAL_COUNT=self.PRN_PROPOSAL_COUNT,
                                                  PRN_NMS_THRESHOLD=self.PRN_NMS_THRESHOLD,
                                                  POOL_SIZE=self.POOL_SIZE,
                                                  RCNN_BATCH_SIZE=self.RCNN_BATCH_SIZE,
                                                  RCNN_POS_FRAC=self.RCNN_POS_FRAC,
                                                  RCNN_POS_IOU_THR=self.RCNN_POS_IOU_THR,
                                                  RCNN_NEG_IOU_THR=self.RCNN_NEG_IOU_THR,
                                                  RCNN_MIN_CONFIDENCE=self.RCNN_MIN_CONFIDENCE,
                                                  RCNN_NME_THRESHOLD=self.RCNN_NME_THRESHOLD,
                                                  RCNN_MAX_INSTANCES=self.RCNN_MAX_INSTANCES)
            optimizer = keras.optimizers.Adam(learning_rate=self.cfg.lr, beta_1=0.5)
            # self.faster_rcnn.build(input_shape=(None, 416, 416, 3))
            epoch_size = self.dataset_len // self.cfg.batch_size
            counter = 0
            for epoch in range(self.cfg.epoch):
                for idx in range(epoch_size):
                    starttime = datetime.datetime.now()
                    inputs = next(self.db_train)
                    num_gt, batch_image_real, boxes, labels, img_width, img_height, batch_z = inputs

                    boxes_list = []
                    boxes_norm_list = []
                    labels_list = []
                    for i in range(self.cfg.batch_size):
                        num_gt_idx = num_gt[i]
                        boxes = boxes[i]
                        labels = labels[i]
                        # 取出有效的boxes
                        boxes_value = boxes[:num_gt_idx, :]

                        x_min, y_min, x_max, y_max = tf.split(boxes_value, [1, 1, 1, 1], axis=-1)
                        boxes_value = tf.concat([y_min, x_min, y_max, x_max], axis=-1)
                        boxes_list.append(boxes_value)
                        # 坐标归一化0~1

                        x_min_norm = x_min / tf.constant(self.cfg.img_size[1], dtype=tf.float32)
                        y_min_norm = y_min / tf.constant(self.cfg.img_size[0], dtype=tf.float32)
                        x_max_norm = x_max / tf.constant(self.cfg.img_size[1], dtype=tf.float32)
                        y_max_norm = y_max / tf.constant(self.cfg.img_size[0], dtype=tf.float32)
                        boxes_value_norm = tf.concat([y_min_norm, x_min_norm, y_max_norm, x_max_norm], axis=-1)
                        boxes_norm_list.append(boxes_value_norm)
                        # 取出有效的labels
                        labels_value = labels[:num_gt_idx]
                        labels_list.append(labels_value)
                    # 这的batch=1,因此可以直接转tensor
                    boxes = tf.convert_to_tensor(boxes_list)
                    boxes_norm = tf.convert_to_tensor(boxes_norm_list)
                    labels = tf.convert_to_tensor(labels_list) + 1  # 类别从0开始， 但新增背景类0，因此‘+1’，依次往后挪

                    # num_gt: shape=[-1,]
                    # batch_image_real: shape=[-1, h, w, 3]
                    # boxes: shape=[-1, num_gt, 4], shape-->[y_min, x_min, y_max, x_max]
                    # boxes_norm: shape=[-1, num_gt, 4], shape-->[y_min, x_min, y_max, x_max], The bounding box coordinates are floats in [0.0, 1.0]
                    # labels: shape=[-1, num_gt]
                    # img_width: shape=[-1,]
                    # img_height: shape=[-1,]
                    # batch_z: shape=[-1, 128]
                    inputs = (num_gt, batch_image_real, boxes, labels, img_width, img_height, batch_z)
                    with tf.GradientTape(persistent=True) as tape:
                        rpn_probs, rpn_deltas, rcnn_logits, rcnn_deltas = self.faster_rcnn(inputs)

                        rpn_class_loss, rpn_location_loss, \
                        rcnn_class_loss, rcnn_location_loss = self.faster_rcnn.loss_function(rpn_probs, rpn_deltas,
                                                                                             boxes, labels, rcnn_logits,
                                                                                             rcnn_deltas)

                        total_loss = rpn_class_loss + rpn_location_loss + rcnn_class_loss + rcnn_location_loss
                    grads = tape.gradient(total_loss, self.faster_rcnn.trainable_variables)
                    optimizer.apply_gradients(zip(grads, self.faster_rcnn.trainable_variables))

                    if counter % 100 == 0:
                        # image_drow_boxes = tf.image.draw_bounding_boxes(batch_image_real, boxes_norm, colors=None)
                        #
                        # tf.summary.image("images:", image_drow_boxes, max_outputs=9, step=counter)
                        tf.summary.scalar('total_loss', float(total_loss), step=counter)
                        tf.summary.scalar('rpn_class_loss', float(rpn_class_loss), step=counter)
                        tf.summary.scalar('rpn_location_loss', float(rpn_location_loss), step=counter)
                        tf.summary.scalar('rcnn_class_loss', float(rcnn_class_loss), step=counter)
                        tf.summary.scalar('rcnn_location_loss', float(rcnn_location_loss), step=counter)

                        # print(counter)
                        # cv2.imwrite('./tmp/' + str(counter) + '.jpg', image_drow_boxes[0].numpy() * 127.5 + 127.5)



                    endtime = datetime.datetime.now()
                    timediff = (endtime - starttime).total_seconds()
                    print(
                        'epoch:[%3d/%3d] step:[%5d/%5d] time:%2.4f total_loss: %3.5f rpn_class_loss:%3.5f rpn_location_loss:%3.5f rcnn_class_loss:%3.5f rcnn_loaction_loss:%3.5f' % \
                        (epoch, self.cfg.epoch, idx, epoch_size, timediff, float(total_loss), float(rpn_class_loss),
                         float(rpn_location_loss), float(rcnn_class_loss), float(rcnn_location_loss)))
                    counter += 1
