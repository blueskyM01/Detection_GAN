import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers, Sequential
import sys
sys.path.append('../')

class Bottleneck(layers.Layer):
    # 残差模块
    def __init__(self, filters=[64, 64, 256], stride=1, training=True):
        super(Bottleneck, self).__init__()
        filters1, filters2, filters3 = filters

        self.training = training
        self.stride = stride

        self.conv2a = layers.Conv2D(filters1, (1, 1), strides=(stride, stride), kernel_initializer='he_normal',
                                    padding='valid')
        self.bn2a = layers.BatchNormalization()
        self.relua = layers.Activation('relu')
        self.conv2b = layers.Conv2D(filters2, (3, 3), strides=(1, 1), kernel_initializer='he_normal', padding='same')
        self.bn2b = layers.BatchNormalization()
        self.relub = layers.Activation('relu')
        self.conv2c = layers.Conv2D(filters3, (1, 1), strides=(1, 1), kernel_initializer='he_normal', padding='same')
        self.bn2c = layers.BatchNormalization()

        if self.stride !=1:
            self.conv_shortcut = layers.Conv2D(filters3, (1, 1), strides=(stride, stride),
                                               kernel_initializer='he_normal', padding='valid')
            self.bn_shortcut = layers.BatchNormalization()
        else:
            self.conv_shortcut = layers.Conv2D(filters3, (1, 1), strides=(1, 1),
                                               kernel_initializer='he_normal', padding='same')
            self.bn_shortcut = layers.BatchNormalization()

        self.reluc = layers.Activation('relu')


    def call(self, inputs):
        x = self.conv2a(inputs)
        x = self.bn2a(x, training=self.training)
        x = self.relua(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=self.training)
        x = self.relub(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=self.training)


        shortcut = self.conv_shortcut(inputs)
        shortcut = self.bn_shortcut(shortcut, training=self.training)

        x = x + shortcut
        x = self.reluc(x)

        return x

class ResNet(keras.Model):
    # 通用的ResNet实现类
    def __init__(self, layer_dims, training=True): # [2, 2, 2, 2]
        super(ResNet, self).__init__()
        self.training = training
        self.padding = layers.ZeroPadding2D((3, 3))
        self.conv1 = layers.Conv2D(64, (7, 7),
                                   strides=(1, 1),
                                   kernel_initializer='he_normal')
        self.bn_conv1 = layers.BatchNormalization()
        self.max_pool = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')

        self.res2 = self.build_block(filters=[64, 64, 256], num_blocks=layer_dims[0], stride=2, training=training)
        self.res3 = self.build_block(filters=[256, 256, 512], num_blocks=layer_dims[1], stride=2, training=training)
        self.res4 = self.build_block(filters=[512, 512, 1024], num_blocks=layer_dims[2], stride=2, training=training)
        self.res5 = self.build_block(filters=[1024, 1024, 2048], num_blocks=layer_dims[3], stride=2, training=training)

    def call(self, inputs):
        x = self.padding(inputs)
        x = self.conv1(x)
        x = self.bn_conv1(x, training=self.training)
        x = tf.nn.relu(x)
        x = self.max_pool(x)

        out2 = self.res2(x)
        out3 = self.res3(out2)
        out4 = self.res4(out3)
        out5 = self.res5(out4)
        return out2, out3, out4, out5

    def build_block(self, filters=[64, 64, 256], num_blocks=3, stride=1, training=True):
        blocks = Sequential()
        blocks.add(Bottleneck(filters, stride=stride, training=training))

        for i in range(num_blocks-1):
            blocks.add(Bottleneck(filters, stride=1, training=training))
        return blocks

def resnet50(training=True):
    res = ResNet(layer_dims=[3,4,6,3], training=training)
    return res

# [3,4,23,3]
def resnet101(training=True):
    res = ResNet(layer_dims=[3,4,6,3], training=training)
    return res

def resnet152(training=True):
    res = ResNet(layer_dims=[3,8,36,3], training=training)
    return res

if __name__ == '__main__':
    aaa = tf.zeros([1,416,416,3])
    model = resnet101()
    model.build(input_shape=(None, 416, 416, 3))
    out = model(aaa)
    print(model.summary())
    print(out)
    for var in model.non_trainable_variables:
        print(var.name)





