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