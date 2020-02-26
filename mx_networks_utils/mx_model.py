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
    def __init__(self, cfg, strategy, num_gpu):
        self.cfg = cfg
        self.strategy = strategy
        self.num_gpu = num_gpu
        self.totoal_batch_size = self.cfg.batch_size * self.num_gpu
        data_loader = mx_data_loader.mx_DatasetLoader(self.cfg.img_size)
        self.dataset, self.dataset_len = data_loader.mx_dataset_load(self.cfg.dataset_dir, self.cfg.dataset_name,
                                                                label_dir=self.cfg.label_dir,
                                                                label_name=self.cfg.label_name,
                                                                shuffle=True, shuffle_size=1000,
                                                                batch_size=self.totoal_batch_size,
                                                                epoch=self.cfg.epoch)
        # self.db_train = iter(dataset)


    def build_model(self):
        # 创建log文件
        summary_writer = tf.summary.create_file_writer(os.path.join(self.cfg.log_dir, self.cfg.dataset_name))
        epoch_size = self.dataset_len // self.totoal_batch_size

        batch_z_val = tf.random.normal([self.cfg.batch_size, 100])
        counter = 0
        with summary_writer.as_default():

            with self.strategy.scope():
                self.dataset = self.strategy.experimental_distribute_dataset(self.dataset)
                G = mx_net.Generator()
                G.build(input_shape=(self.cfg.batch_size, 100))
                D = mx_net.Discriminator()
                D.build(input_shape=(self.cfg.batch_size, self.cfg.img_size[0], self.cfg.img_size[1], self.cfg.img_size[2]))
                # 分别为生成器和判别器创建优化器
                g_optimizer = keras.optimizers.Adam(learning_rate=self.cfg.g_lr, beta_1=0.5)
                d_optimizer = keras.optimizers.Adam(learning_rate=self.cfg.d_lr, beta_1=0.5)

                def compute_loss(per_replica_loss, totoal_batch_size):
                    return tf.nn.compute_average_loss(per_replica_loss, global_batch_size = totoal_batch_size)

                # 使用distributed_train_step封装，因此不需要再用tf.function
                # train_step在每一个GPU上运行，其输入是16个样本，因此计算的也是16个样本的loss值
                def train_step(inputs):
                    batch_image_real, batch_z = inputs
                    with tf.GradientTape(persistent=True) as tape:
                        batch_image_fake = G(batch_z, self.cfg.is_train)
                        d_fake = D(batch_image_fake, self.cfg.is_train)
                        d_real = D(batch_image_real, self.cfg.is_train)
                        d_loss, g_loss = mx_ops.w_loss_fn(d_fake_logits=d_fake, d_real_logits=d_real)

                        d_loss = compute_loss(d_loss, self.totoal_batch_size)
                        g_loss = compute_loss(g_loss, self.totoal_batch_size)

                    d_grads = tape.gradient(d_loss, D.trainable_variables)
                    g_grads = tape.gradient(g_loss, G.trainable_variables)

                    d_optimizer.apply_gradients(zip(d_grads, D.trainable_variables))
                    # for d_v in D.trainable_variables:
                    #     d_v.assign_sub(tf.clip_by_value(d_v, -0.000001, 0.000001))
                    g_optimizer.apply_gradients(zip(g_grads, G.trainable_variables))

                    return d_loss

                @tf.function
                def distributed_train_step(inputs):
                    # 根据GPU数对batch进行均分，每个GPU使用batch/len(gpu)个样本独自运行train_step
                    per_replica_d_loss = self.strategy.experimental_run_v2(train_step, args=(inputs,))
                    return self.strategy.reduce(tf.distribute.ReduceOp.SUM,
                                           per_replica_d_loss,
                                           axis=None)  # 将不同GPU上的loss进行聚合

                # G.load_weights('generator.ckpt')
                # D.load_weights('discriminator.ckpt')
                # print('Loaded chpt!!')

            for epoch in range(self.cfg.epoch):
                for i in range(epoch_size):
                    starttime = datetime.datetime.now()
                    data_input = next(iter(self.dataset))
                    d_loss = distributed_train_step(data_input)
                    g_loss = 0
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
                    print('epoch:[%3d/%3d] step:[%5d/%5d] time:%2.4f d_loss: %3.5f g_loss:%3.5f' % \
                          (epoch, self.cfg.epoch, i, epoch_size, timediff, float(d_loss), float(g_loss)))

                    counter += 1
