import numpy as np
import tensorflow as tf
# 导入 keras 模型，不能使用 import keras，它导入的是标准的 Keras 库
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import argparse, os, time, sys, datetime, cv2
sys.path.append('../')
import mx_networks_utils.mx_networks as mx_net
import mx_Dataset.mx_load_anime_dataset as mx_data_loader
import mx_networks_utils.mx_ops as mx_ops
import mx_networks_utils.mx_utils as mx_utils

class DetectionGAN:
    def __init__(self, cfg):
        self.cfg = cfg
        data_loader = mx_data_loader.mx_DatasetLoader(self.cfg.img_size)
        dataset, self.dataset_len = data_loader.mx_dataset_load(self.cfg.dataset_dir, self.cfg.dataset_name,
                                                                label_dir=self.cfg.label_dir,
                                                                label_name=self.cfg.label_name,
                                                                shuffle=True, shuffle_size=1000,
                                                                batch_size=self.cfg.batch_size,
                                                                epoch=self.cfg.epoch)
        self.db_train = iter(dataset)

    def build_model(self):

        G = mx_net.Generator()
        G.build(input_shape=(self.cfg.batch_size, 100))
        D = mx_net.Discriminator()
        D.build(input_shape=(self.cfg.batch_size, self.cfg.img_size[0], self.cfg.img_size[1], self.cfg.img_size[2]))
        # 分别为生成器和判别器创建优化器
        g_optimizer = keras.optimizers.Adam(learning_rate=self.cfg.g_lr, beta_1=0.5)
        d_optimizer = keras.optimizers.Adam(learning_rate=self.cfg.d_lr, beta_1=0.5)

        # 创建log文件
        summary_writer = tf.summary.create_file_writer(os.path.join(self.cfg.log_dir, self.cfg.dataset_name))

        # G.load_weights('generator.ckpt')
        # D.load_weights('discriminator.ckpt')
        # print('Loaded chpt!!')

        epoch_size = self.dataset_len // self.cfg.batch_size

        batch_z_val = tf.random.normal([self.cfg.batch_size, 100])
        counter = 0
        with summary_writer.as_default():
            for epoch in range(self.cfg.epoch):
                for i in range(epoch_size):
                    starttime = datetime.datetime.now()
                    batch_image_real = next(self.db_train)
                    batch_z = tf.random.normal([self.cfg.batch_size, 100])

                    # 判别器前向计算
                    with tf.GradientTape(persistent=True) as tape:
                        batch_image_fake = G(batch_z, self.cfg.is_train)
                        d_fake = D(batch_image_fake, self.cfg.is_train)
                        d_real = D(batch_image_real, self.cfg.is_train)

                        d_loss = mx_ops.d_loss_fn(d_fake, d_real)
                        g_loss = mx_ops.g_loss_fn(d_fake)

                    d_grads = tape.gradient(d_loss, D.trainable_variables)
                    d_optimizer.apply_gradients(zip(d_grads, D.trainable_variables))

                    g_grads = tape.gradient(g_loss, G.trainable_variables)
                    g_optimizer.apply_gradients(zip(g_grads, G.trainable_variables))

                    if counter % 100 == 0:
                        tf.summary.scalar('d_loss', float(d_loss), step=counter)
                        tf.summary.scalar('g_loss', float(g_loss), step=counter)

                        val_images = G(batch_z_val, False)

                        tf.summary.image("val_images:", val_images, max_outputs=9, step=counter)

                        img = mx_utils.m4_image_save_cv(val_images.numpy(), rows=8, zero_mean=True)

                        cv2.imwrite(os.path.join(self.cfg.g_image_dir, self.cfg.dataset_name) + '/' + '%08d' % (counter) + '.jpg', img)
                        print('add summary once....')

                    if counter % 1000 == 0:
                        G.save_weights(os.path.join(os.path.join(self.cfg.checkpoint_dir, self.cfg.dataset_name),
                                                    'generator.ckpt'))
                        D.save_weights(os.path.join(os.path.join(self.cfg.checkpoint_dir, self.cfg.dataset_name),
                                                    'discriminator.ckpt'))
                        print('save checkpoint....')

                    endtime = datetime.datetime.now()
                    timediff = (endtime - starttime).total_seconds()
                    print('epoch:[%3d/%3d] step:[%5d/%5d] time:%2.4f d_loss: %2.5f g_loss:%2.5f' % \
                          (epoch, self.cfg.epoch, i, epoch_size, timediff, float(d_loss), float(g_loss)))

                    counter += 1
