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

class FasterRCNN:
    def __init__(self, cfg, strategy):
        self.cfg = cfg
        self.strategy = strategy

        self.totoal_batch_size = self.strategy.num_replicas_in_sync * self.cfg.batch_size

        print('totoal_batch_size:', self.totoal_batch_size)
        print('batchmmmm:', self.strategy.num_replicas_in_sync)

        data_loader = mx_data_loader.mx_DatasetLoader(self.cfg.img_size)
        self.dataset, self.dataset_len = data_loader.mx_dataset_load(self.cfg.dataset_dir, self.cfg.dataset_name,
                                                                     label_dir=self.cfg.label_dir,
                                                                     label_name=self.cfg.label_name,
                                                                     shuffle=True, shuffle_size=1000,
                                                                     batch_size=self.totoal_batch_size,
                                                                     epoch=self.cfg.epoch)

    def build_model(self):
        # 创建log文件
        summary_writer = tf.summary.create_file_writer(os.path.join(self.cfg.results_dir,
                                                                    self.cfg.log_dir,
                                                                    self.cfg.tmp_result_name))

        with summary_writer.as_default():
            with self.strategy.scope():
                dataset_distribute = self.strategy.experimental_distribute_dataset(self.dataset)
                self.db_train = iter(dataset_distribute)





                epoch_size = self.dataset_len // self.totoal_batch_size
                for epoch in range(self.cfg.epoch):
                    for i in range(epoch_size):
                        starttime = datetime.datetime.now()

                        inputs = next(self.db_train)
                        num_gt, img, boxes, labels, img_width, img_height = inputs



                        print(num_gt.shape, img.shape, boxes.shape, labels.shape, img_width.shape, img_height.shape)

                        endtime = datetime.datetime.now()
                        timediff = (endtime - starttime).total_seconds()
                        print('time:', timediff)
