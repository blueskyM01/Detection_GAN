import numpy as np
import tensorflow as tf
# 导入 keras 模型，不能使用 import keras，它导入的是标准的 Keras 库
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import argparse, os, time

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

        self.module1 = mx_BE_Encoder_Module(filter_num*1)
        self.conv1 = layers.Conv2D(filter_num*1, (3, 3), strides=2, padding='same', activation='elu')

        self.module2 = mx_BE_Encoder_Module(filter_num * 2)
        self.conv2 = layers.Conv2D(filter_num*2, (3, 3), strides=2, padding='same', activation='elu')

        self.module3 = mx_BE_Encoder_Module(filter_num * 3)
        self.conv3 = layers.Conv2D(filter_num*3, (3, 3), strides=2, padding='same', activation='elu')

        self.module4 = mx_BE_Encoder_Module(filter_num * 4)
        self.conv4 = layers.Conv2D(filter_num * 4, (3, 3), strides=2, padding='same', activation='elu')

        self.module5 = mx_BE_Encoder_Module(filter_num * 5)
        self.conv5 = layers.Conv2D(filter_num * 5, (3, 3), strides=2, padding='same', activation='elu')

        self.module6 = mx_BE_Encoder_Module(filter_num * 6)

        self.fc1 = layers.Dense(8*8*filter_num*6)

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
