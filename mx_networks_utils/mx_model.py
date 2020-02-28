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


    def build_model(self):

        # 创建log文件
        summary_writer = tf.summary.create_file_writer(os.path.join(self.cfg.log_dir, self.cfg.tmp_result_name))
        with summary_writer.as_default():
            with self.strategy.scope():
                dataset_distribute = self.strategy.experimental_distribute_dataset(self.dataset)
                self.db_train = iter(dataset_distribute)



                G = mx_net.Generator()
                G.build(input_shape=(self.cfg.batch_size, 100))
                D = mx_net.Discriminator()
                D.build(input_shape=(self.cfg.batch_size, self.cfg.img_size[0], self.cfg.img_size[1], self.cfg.img_size[2]))
                # 分别为生成器和判别器创建优化器
                g_optimizer = keras.optimizers.Adam(learning_rate=self.cfg.g_lr, beta_1=0.5)
                d_optimizer = keras.optimizers.Adam(learning_rate=self.cfg.d_lr, beta_1=0.5)

                def compute_loss(d_fake, d_real):
                    d_loss, g_loss = mx_ops.w_loss_fn(d_fake_logits=d_fake, d_real_logits=d_real)
                    # 计算的是每块卡上的损失，即total_batch_size/num_gpu个样本
                    # 需要注意的是，这里的loss的shape应为 [-1, 1]，即每个样本的损失。
                    # tf.nn.compute_average_loss：会在即total_batch_size个样本loss的sum上求均值。如下，假设我们有三块卡，total_batch_size=64：
                    # GPU0:total_loss = 20,GPU1:total_loss = 30,GPU2:total_loss = 25 ----> (20 + 30 + 25) / 64 -> 20/64 + 30/64 + 25/64 + 26/64
                    d_average_loss = tf.nn.compute_average_loss(d_loss, global_batch_size=self.totoal_batch_size)
                    g_average_loss = tf.nn.compute_average_loss(g_loss, global_batch_size=self.totoal_batch_size)
                    return d_average_loss, g_average_loss

                # 使用distributed_train_step封装，因此不需要再用tf.function
                # train_step在每一个GPU上运行，其输入是(total_batch_size / num_gpu)个样本，
                # 因此计算的也是(total_batch_size / num_gpu)个样本的loss值
                def train_step(inputs):
                    batch_image_real, batch_z = inputs
                    with tf.GradientTape(persistent=True) as tape:
                        batch_image_fake = G(batch_z, self.cfg.is_train)
                        d_fake = D(batch_image_fake, self.cfg.is_train)
                        d_real = D(batch_image_real, self.cfg.is_train)

                        d_average_loss, g_average_loss = compute_loss(d_fake, d_real)
                        gp = mx_ops.gradient_penalty(D, batch_image_real, batch_image_fake, d_fake.shape[0],
                                                     is_train=self.cfg.is_train)
                        d_average_loss = d_average_loss + self.cfg.grad_penalty_weight * gp

                    d_grads = tape.gradient(d_average_loss, D.trainable_variables)
                    g_grads = tape.gradient(g_average_loss, G.trainable_variables)

                    d_optimizer.apply_gradients(zip(d_grads, D.trainable_variables))
                    g_optimizer.apply_gradients(zip(g_grads, G.trainable_variables))

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
                batch_z_val = tf.random.normal([self.cfg.batch_size, 100])
                counter = 0

                for epoch in range(self.cfg.epoch):
                    for i in range(epoch_size):
                        starttime = datetime.datetime.now()
                        inputs = next(self.db_train)

                        d_loss, g_loss = distributed_train_step(inputs)


                        if counter % 40 == 0:
                            tf.summary.scalar('d_loss', float(d_loss), step=counter)
                            tf.summary.scalar('g_loss', float(g_loss), step=counter)

                            val_images = G(batch_z_val, False)

                            tf.summary.image("val_images:", val_images, max_outputs=9, step=counter)

                            img = mx_utils.m4_image_save_cv(val_images.numpy(), rows=8, zero_mean=True)

                            cv2.imwrite(os.path.join(self.cfg.g_image_dir, self.cfg.tmp_result_name) + '/' + '%08d' % (counter) + '.jpg', img)
                            print('add summary once....')

                        if counter % 1000 == 0:
                            G.save_weights(os.path.join(os.path.join(self.cfg.checkpoint_dir, self.cfg.tmp_result_name),
                                                        'generator.ckpt'))
                            D.save_weights(os.path.join(os.path.join(self.cfg.checkpoint_dir, self.cfg.tmp_result_name),
                                                        'discriminator.ckpt'))
                            print('save checkpoint....')

                        endtime = datetime.datetime.now()
                        timediff = (endtime - starttime).total_seconds()
                        print('epoch:[%3d/%3d] step:[%5d/%5d] time:%2.4f d_loss: %3.5f g_loss:%3.5f' % \
                              (epoch, self.cfg.epoch, i, epoch_size, timediff, float(d_loss), float(g_loss)))

                        counter += 1
