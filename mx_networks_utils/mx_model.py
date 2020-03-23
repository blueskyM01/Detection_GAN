import numpy as np
import tensorflow as tf
# 导入 keras 模型，不能使用 import keras，它导入的是标准的 Keras 库
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import argparse, os, time, sys, datetime, cv2, logging
sys.path.append('../')
import mx_networks_utils.mx_networks as mx_net
import mx_Dataset.mx_load_dataset as mx_data_loader
import mx_networks_utils.mx_ops as mx_ops
import mx_networks_utils.mx_utils as mx_utils

class DetectionGAN:
    def __init__(self, cfg, strategy):
        self.cfg = cfg
        self.strategy = strategy
        self.totoal_batch_size = self.strategy.num_replicas_in_sync * self.cfg.batch_size
        data_loader = mx_data_loader.mx_DatasetLoader(self.cfg.img_size)
        self.dataset, self.dataset_len = data_loader.mx_dataset_load(self.cfg.dataset_dir, self.cfg.dataset_name,
                                                                label_dir=self.cfg.label_dir,
                                                                label_name=self.cfg.label_name,
                                                                shuffle=True, shuffle_size=1000,
                                                                batch_size=self.totoal_batch_size,
                                                                epoch=self.cfg.epoch)
        self.k_t = tf.Variable(0., trainable=False)


    def build_model(self):

        # 创建log文件
        summary_writer = tf.summary.create_file_writer(os.path.join(self.cfg.results_dir,
                                                                    self.cfg.log_dir,
                                                                    self.cfg.tmp_result_name))
        with summary_writer.as_default():
            with self.strategy.scope():
                dataset_distribute = self.strategy.experimental_distribute_dataset(self.dataset)
                self.db_train = iter(dataset_distribute)

                G = mx_net.mx_BE_Generator(self.cfg.filter_num)
                G.build(input_shape=(None, 128+2048))
                D = mx_net.mx_BE_Discriminator(self.cfg.filter_num)
                D.build(input_shape=(None, self.cfg.img_size[0], self.cfg.img_size[1], 3))
                # 分别为生成器和判别器创建优化器
                g_optimizer = keras.optimizers.Adam(learning_rate=self.cfg.g_lr, beta_1=0.5)
                d_optimizer = keras.optimizers.Adam(learning_rate=self.cfg.d_lr, beta_1=0.5)

                resnet101 = keras.applications.ResNet101(weights='imagenet', include_top=False)

                num_layers = len(resnet101.layers)
                print("Number of layers in the resnet101: ", num_layers)
                for layer in resnet101.layers[:num_layers]:
                    layer.trainable = False

                # G.load_weights('/gs/home/yangjb/My_Job/AI_CODE/Detection_GAN/results/checkpoint/roi_image-c2/generator_00000198.ckpt')
                # D.load_weights('/gs/home/yangjb/My_Job/AI_CODE/Detection_GAN/results/checkpoint/roi_image-c2/discriminator_00000198.ckpt')
                # print('Loaded chpt!!')

                def compute_loss(AE_real, AE_fake, real, fake, k_t):
                    d_loss, g_loss, AE_real_loss = mx_ops.w_AE_loss_fn(AE_real, AE_fake, real, fake, k_t)

                    # 计算的是每块卡上的损失，即total_batch_size/num_gpu个样本
                    # 需要注意的是，这里的loss的shape应为 [-1, 1]，即每个样本的损失。
                    # tf.nn.compute_average_loss：会在即total_batch_size个样本loss的sum上求均值。如下，假设我们有三块卡，total_batch_size=64：
                    # GPU0:total_loss = 20,GPU1:total_loss = 30,GPU2:total_loss = 25 ----> (20 + 30 + 25) / 64 -> 20/64 + 30/64 + 25/64 + 26/64
                    d_average_loss = tf.nn.compute_average_loss(d_loss, global_batch_size=self.totoal_batch_size)
                    g_average_loss = tf.nn.compute_average_loss(g_loss, global_batch_size=self.totoal_batch_size)
                    return d_average_loss, g_average_loss, AE_real_loss

                # 使用distributed_train_step封装，因此不需要再用tf.function
                # train_step在每一个GPU上运行，其输入是(total_batch_size / num_gpu)个样本，
                # 因此计算的也是(total_batch_size / num_gpu)个样本的loss值
                def train_step(inputs):
                    num_gt, batch_image_real, boxes, labels, img_width, img_height, batch_z = inputs

                    # roi_images = tf.constant([], shape=[0, self.cfg.img_size[0], self.cfg.img_size[1], 3], dtype=tf.float32)
                    # for i in range(batch_image_real.shape[0]):
                    #     image = batch_image_real[i]
                    #     image_mask = image * 0.5
                    #     num_gt_idx = num_gt[i]
                    #     boxes_ = boxes[i]
                    #
                    #     # 取出有效的boxes
                    #     boxes_value = boxes_[:num_gt_idx, :]
                    #
                    #     x_min, y_min, x_max, y_max = tf.split(boxes_value, [1, 1, 1, 1], axis=-1)
                    #     boxes_value = tf.concat([y_min, x_min, y_max, x_max], axis=-1)
                    #
                    #     x_min_norm = x_min / tf.constant(self.cfg.img_size[1], dtype=tf.float32)
                    #     y_min_norm = y_min / tf.constant(self.cfg.img_size[0], dtype=tf.float32)
                    #     x_max_norm = x_max / tf.constant(self.cfg.img_size[1], dtype=tf.float32)
                    #     y_max_norm = y_max / tf.constant(self.cfg.img_size[0], dtype=tf.float32)
                    #     boxes_value_norm = tf.concat([y_min_norm, x_min_norm, y_max_norm, x_max_norm], axis=-1)
                    #     boxes_value_norm = tf.expand_dims(boxes_value_norm, axis=0)
                    #     image_mask = tf.expand_dims(image_mask, axis=0)
                    #
                    #     roi_image = tf.image.draw_bounding_boxes(image_mask, boxes_value_norm, colors=None)
                    #     roi_images = tf.concat([roi_images, roi_image], axis=0)
                    #
                    # tf.summary.image("org_images:", batch_image_real, max_outputs=9, step=counter)
                    # tf.summary.image("roi_images:", roi_images, max_outputs=9, step=counter)

                    with tf.GradientTape(persistent=True) as tape:
                        feature_map = resnet101(batch_image_real)
                        feature_map = tf.nn.l2_normalize(tf.reduce_mean(feature_map, axis=[1, 2]), axis=1)
                        z_feat = tf.concat([batch_z, feature_map], axis=-1)
                        batch_image_fake = G(z_feat)
                        d_fake = D(batch_image_fake)
                        d_real = D(batch_image_real)
                        d_average_loss, g_average_loss, AE_real_loss = compute_loss(d_real, d_fake, batch_image_real, batch_image_fake, self.k_t)
                        # gp = mx_ops.gradient_penalty(D, batch_image_real, batch_image_fake, d_fake.shape[0],
                        #                              is_train=self.cfg.is_train)
                        # d_average_loss = d_average_loss + self.cfg.grad_penalty_weight * gp

                    d_grads = tape.gradient(d_average_loss, D.trainable_variables)
                    g_grads = tape.gradient(g_average_loss, G.trainable_variables)

                    d_optimizer.apply_gradients(zip(d_grads, D.trainable_variables))
                    g_optimizer.apply_gradients(zip(g_grads, G.trainable_variables))

                    balance = 0.5 * tf.reduce_mean(AE_real_loss) - g_average_loss

                    self.k_t.assign(tf.clip_by_value(self.k_t + 0.01 * balance, 0, 1))

                    return d_average_loss, g_average_loss

                @tf.function
                def distributed_train_step(inputs):
                    # 根据GPU数对batch进行均分，每个GPU使用batch/len(gpu)个样本独自运行train_step
                    d_average_loss, g_average_loss = self.strategy.experimental_run_v2(train_step, args=(inputs,))

                    d_final_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, d_average_loss,
                                                   axis=None)  # 将不同GPU上的loss进行聚合
                    g_final_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, g_average_loss,
                                                   axis=None)  # 将不同GPU上的loss进行聚合

                    return d_final_loss, g_final_loss


                epoch_size = self.dataset_len // self.totoal_batch_size
                batch_z_val = tf.random.normal([self.cfg.batch_size, 128])
                counter = 0
                log, file, stream, final_log_file = mx_utils.log_creater(os.path.join(self.cfg.results_dir,
                                                                                      self.cfg.log_dir,
                                                                                      self.cfg.tmp_result_name),
                                                                         'log_file')

                for epoch in range(self.cfg.epoch):
                    for step_idx in range(epoch_size):
                        starttime = datetime.datetime.now()
                        inputs = next(self.db_train)
                        num_gt, batch_image_real, boxes, labels, img_width, img_height, batch_z = inputs
                        #
                        # rois_imgs = self.conver_to_roi_image(num_gt.numpy(), batch_image_real.numpy(), boxes.numpy())
                        # batch_image_real = tf.convert_to_tensor(rois_imgs)
                        # batch_image_real = tf.cast(batch_image_real, tf.float32)
                        # inputs = (num_gt, batch_image_real, boxes, labels, img_width, img_height, batch_z)


                        d_loss, g_loss = distributed_train_step(inputs)

                        if counter % 200 == 0:
                            tf.summary.scalar('d_loss', float(d_loss), step=counter)
                            tf.summary.scalar('g_loss', float(g_loss), step=counter)

                            feature_map = resnet101(batch_image_real)
                            feature_map = tf.nn.l2_normalize(tf.reduce_mean(feature_map, axis=[1, 2]), axis=1)
                            z_feat = tf.concat([batch_z, feature_map], axis=-1)
                            val_images = G(z_feat)

                            tf.summary.image("val_images:", val_images, max_outputs=9, step=counter)

                            img = mx_utils.m4_image_save_cv(val_images.numpy(), rows=8, zero_mean=True)

                            cv2.imwrite(os.path.join(self.cfg.results_dir, self.cfg.generate_image_dir, self.cfg.tmp_result_name) + '/' + '%08d' % (counter) + '.jpg', img)
                            print('add summary once....')

                        if counter % 2000 == 0:
                            G.save_weights(os.path.join(os.path.join(self.cfg.results_dir, self.cfg.checkpoint_dir, self.cfg.tmp_result_name),
                                                        'generator_%08d.ckpt' % (epoch)))
                            D.save_weights(os.path.join(os.path.join(self.cfg.results_dir, self.cfg.checkpoint_dir, self.cfg.tmp_result_name),
                                                        'discriminator_%08d.ckpt' % (epoch)))
                            print('save checkpoint....')

                        endtime = datetime.datetime.now()
                        timediff = (endtime - starttime).total_seconds()
                        print('epoch:[%3d/%3d] step:[%5d/%5d] time:%2.4f d_loss: %3.5f g_loss:%3.5f k_t:%3.5f' % \
                              (epoch, self.cfg.epoch, step_idx, epoch_size, timediff, float(d_loss), float(g_loss), float(self.k_t.numpy())))

                        formatter = logging.Formatter(
                            'epoch:{:3d}/{:3d} step:{:6d}/{:6d} time:{:2.4f} d_loss:{:3.5f} g_loss:{:3.5f} k_t:{:1.5f}'.format(
                                epoch, self.cfg.epoch, step_idx, epoch_size,
                                timediff, float(d_loss),
                                float(g_loss),
                                float(self.k_t.numpy())),
                                )
                        mx_utils.log_write(log, file, stream, final_log_file, formatter)

                        counter += 1

    def conver_to_roi_image(self, num_gt, imgs, boxes):
        '''
        Introduction: 生成roi图像
        :param num_gt: shape=(batch_size,), int
        :param imgs: shape=(batch_size, h, w, 3) -1~1
        :param boxes: [batch_size, -1, 4] [x0, y0, x1, y1]
        :return:
        '''
        roi_imgs = []
        for idx in range(imgs.shape[0]):
            box = boxes[idx]
            img = np.zeros([imgs[idx].shape[0], imgs[idx].shape[1], 1])
            num = num_gt[idx]
            box_val = box[:num, :]

            for b in box_val:
                x0 = int(b[0])
                y0 = int(b[1])
                x1 = int(b[2])
                y1 = int(b[3])
                cv2.rectangle(img, (x0, y0), (x1, y1), (1, 1, 1), 4)
            roi_imgs.append(img)
        return np.array(roi_imgs)
