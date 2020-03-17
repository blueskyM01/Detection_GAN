import numpy as np
import tensorflow as tf
# 导入 keras 模型，不能使用 import keras，它导入的是标准的 Keras 库
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import argparse, os, time, sys, datetime, cv2, logging

sys.path.append('../')
import mx_Dataset.mx_load_dataset as mx_data_loader
import mx_networks_utils.mx_ops as mx_ops
import mx_networks_utils.mx_utils as mx_utils
import mx_networks_utils.mx_networks as mx_net
import models.mx_faster_rcnn as mx_faster_rcnn


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

        self.anchor_scales = [128, 256, 512]
        self.anchor_ratios = [0.5, 1., 2.]
        self.rpn_batch_size = 256
        self.rpn_pos_frac = 0.5
        self.rpn_pos_iou_thr = 0.7
        self.rpn_neg_iou_thr = 0.3
        self.rpn_proposal_count = 2000
        self.rpn_nms_thr = 0.7
        self.pooling_size = (7, 7)
        self.rcnn_batch_size = 256
        self.rcnn_pos_frac = 0.25
        self.rcnn_pos_iou_thr = 0.5
        self.rcnn_neg_iou_thr = 0.5
        self.rcnn_min_confidence = 0.4
        self.rcnn_nms_thr = 0.3
        self.rcnn_max_instance = 100
        self.rpn_cls_loss_weight = 1.
        self.rpn_loc_loss_weight = 1.
        self.rcnn_cls_loss_weight = 1.
        self.rcnn_loc_loss_weight = 1.

    def build_model(self):
        # 创建log文件
        summary_writer = tf.summary.create_file_writer(os.path.join(self.cfg.results_dir,
                                                                    self.cfg.log_dir,
                                                                    self.cfg.tmp_result_name))

        with summary_writer.as_default():
            self.faster_rcnn = mx_faster_rcnn.Faster_RCNN(cfg=self.cfg,
                                                          num_classes=self.num_classes,
                                                          anchor_scales=self.anchor_scales,
                                                          anchor_ratios=self.anchor_ratios,
                                                          rpn_batch_size=self.rpn_batch_size,
                                                          rpn_pos_frac=self.rpn_pos_frac,
                                                          rpn_pos_iou_thr=self.rpn_pos_iou_thr,
                                                          rpn_neg_iou_thr=self.rpn_neg_iou_thr,
                                                          rpn_proposal_count=self.rpn_proposal_count,
                                                          rpn_nms_thr=self.rpn_nms_thr,
                                                          pooling_size=self.pooling_size,
                                                          rcnn_batch_size=self.rcnn_batch_size,
                                                          rcnn_pos_frac=self.rcnn_pos_frac,
                                                          rcnn_pos_iou_thr=self.rcnn_pos_iou_thr,
                                                          rcnn_neg_iou_thr=self.rcnn_neg_iou_thr,
                                                          rcnn_min_confidence=self.rcnn_min_confidence,
                                                          rcnn_nms_thr=self.rcnn_nms_thr,
                                                          rcnn_max_instance=self.rcnn_max_instance)

            optimizer = keras.optimizers.Adam(learning_rate=self.cfg.lr, beta_1=0.5)
            # self.faster_rcnn.build(input_shape=(None, 416, 416, 3))
            epoch_size = self.dataset_len // self.cfg.batch_size
            counter = 0
            log, file, stream, final_log_file = mx_utils.log_creater(os.path.join(self.cfg.results_dir,
                                                                                  self.cfg.log_dir,
                                                                                  self.cfg.tmp_result_name), 'log_file')
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
                        boxes_ = boxes[i]
                        labels = labels[i]
                        # 取出有效的boxes
                        boxes_value = boxes_[:num_gt_idx, :]

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
                        rpn_class_loss, rpn_location_loss, rcnn_class_loss, rcnn_location_loss = self.faster_rcnn(
                            inputs)
                        total_loss = rpn_class_loss * self.rpn_cls_loss_weight + rpn_location_loss * self.rpn_loc_loss_weight + \
                                     rcnn_class_loss * self.rcnn_cls_loss_weight + rcnn_location_loss * self.rcnn_loc_loss_weight

                    grads = tape.gradient(total_loss, self.faster_rcnn.trainable_variables)
                    optimizer.apply_gradients(zip(grads, self.faster_rcnn.trainable_variables))

                    if counter % 100 == 0:
                        # image_drow_boxes = tf.image.draw_bounding_boxes(batch_image_real, boxes_norm, colors=None)
                        #
                        # tf.summary.image("images:", image_drow_boxes, max_outputs=9, step=counter)
                        tf.summary.scalar('total_loss', float(total_loss), step=counter)
                        tf.summary.scalar('rpn_class_loss', float(rpn_class_loss), step=counter)
                        tf.summary.scalar('rpn_location_loss', float(rpn_location_loss), step=counter)
                        tf.summary.scalar('rpn_class_acc', float(self.faster_rcnn.rpn_class_acc), step=counter)
                        tf.summary.scalar('rcnn_class_loss', float(rcnn_class_loss), step=counter)
                        tf.summary.scalar('rcnn_location_loss', float(rcnn_location_loss), step=counter)

                        # print(counter)
                        # cv2.imwrite('./tmp/' + str(counter) + '.jpg', image_drow_boxes[0].numpy() * 127.5 + 127.5)
                    if counter % 40 == 0:
                        # proposal result
                        roi_box = self.faster_rcnn.proposal_bbox
                        roi_prob = self.faster_rcnn.proposal_probs

                        roi_box = roi_box.numpy()
                        roi_prob = roi_prob.numpy().tolist()

                        img = batch_image_real[0].numpy() * 127.5 + 127.5
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        for box, score in zip(roi_box, roi_prob):
                            y0 = int(box[0] * self.cfg.img_size[0])
                            x0 = int(box[1] * self.cfg.img_size[1])
                            y1 = int(box[2] * self.cfg.img_size[0])
                            x1 = int(box[3] * self.cfg.img_size[1])

                            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
                            cv2.putText(img, str(score), (x1, y0), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        img_path = os.path.join(
                            os.path.join(self.cfg.results_dir, self.cfg.generate_image_dir, self.cfg.tmp_result_name))
                        cv2.imwrite(img_path + '/roi' + str(counter) + '.jpg', img)
                        # detection result

                    if counter % 2000 == 0:
                        self.faster_rcnn.save_weights(os.path.join(
                            os.path.join(self.cfg.results_dir, self.cfg.checkpoint_dir, self.cfg.tmp_result_name),
                            'faster_rcnn_%04d.ckpt' % (epoch)))

                        print('save checkpoint....')

                    endtime = datetime.datetime.now()
                    timediff = (endtime - starttime).total_seconds()
                    print(
                        'epoch:[%3d/%3d] step:[%5d/%5d] time:%2.4f total_loss: %3.5f rpn_cls_loss:%3.5f rpn_loc_loss:%3.5f rpn_acc:%1.5f \
                         rcnn_cls_loss:%3.5f rcnn_loc_loss:%3.5f rcnn_acc:%1.5f' % \
                        (epoch, self.cfg.epoch, idx, epoch_size, timediff, float(total_loss),
                         float(rpn_class_loss), float(rpn_location_loss), float(self.faster_rcnn.rpn_class_acc),
                         float(rcnn_class_loss), float(rcnn_location_loss), float(self.faster_rcnn.rcnn_class_acc)))

                    formatter = logging.Formatter(
                        'epoch:{:3d}/{:3d} step:{:6d}/{:6d} time:{:2.4f} total_loss:{:3.5f} rpn_cls_loss:{:3.5f} rpn_loc_loss:{:3.5f} rpn_class_acc:{:1.5f} rcnn_cls_loss:{:3.5f} rcnn_loc_loss:{:3.5f} rcnn_class_acc:{:1.5f}'.format(
                            epoch, self.cfg.epoch, idx, epoch_size,
                            timediff, float(total_loss),
                            float(rpn_class_loss),
                            float(rpn_location_loss),
                            float(self.faster_rcnn.rpn_class_acc),
                            float(rcnn_class_loss),
                            float(rcnn_location_loss),
                            float(self.faster_rcnn.rcnn_class_acc)))
                    mx_utils.log_write(log, file, stream, final_log_file, formatter)

                    counter += 1
