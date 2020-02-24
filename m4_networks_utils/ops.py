import tensorflow as tf
import numpy as np

def m4_leak_relu(x, leak=0.2):
    '''
    Introdction: leak_relu active function
    :param x: a tensor
    :param leak:
    :return: a tensor
    '''
    return tf.maximum(x, leak * x, name='leak_relu')

def m4_batch_norm(input_, is_trainable):
    '''
    Introduction: batch normalizaiton function
    :param input_: a tensor
    :param is_trainable: bool: true or false
    :return: a tensor
    '''
    # try:
    output = tf.contrib.layers.batch_norm(input_, decay=0.99,
                                          updates_collections=None,
                                          epsilon=1e-5,
                                          scale=True,
                                          is_training=is_trainable)
    # except:
    #     mean, variance = tf.nn.moments(input_, axes=[0, 1, 2])
    #     _, _, _, nc = input_.get_shape().as_list()
    #     beta = tf.get_variable('beta', [nc], tf.float32,
    #                            initializer=tf.constant_initializer(0.0, tf.float32))  # [out_channels]
    #     gamma = tf.get_variable('gamma', [nc], tf.float32,
    #                             initializer=tf.constant_initializer(1.0, tf.float32))
    #     output = tf.nn.batch_normalization(input_, mean, variance, beta, gamma, 1e-5)
    return output

def m4_norm_func(input_, is_trainable, name):
    '''
    Introduction:可以实现多种norm防过拟合函数，
    注意：不用norm时，设置成None
    :param input_: a tensor
    :param is_trainable: bool: true or false
    :param name: norm函数的名称，目前只能是"batch_norm", 因为目前只有这个防过拟合函数
    :return:
    '''
    if name==None:
        output_ = input_
    elif name=='batch_norm':
        output_ = m4_batch_norm(input_, is_trainable)
    return output_

def m4_active_function(input_, active_function='relu'):
    '''
    Introduction:可以实现多种激活函数，目前只有'relu'和'leak_relu'，
    注意：不用激活函数时设置成None
    :param input_: a tensor
    :param active_function: str: 目前只能是'relu'和'leak_relu'
    :return: a tensor
    '''
    if active_function==None:
        active = input_
    elif active_function == 'relu':
        active = tf.nn.relu(input_)
    elif active_function == 'leak_relu':
        active = m4_leak_relu(input_)
    return active

def m4_conv_layers(input_, fiters, k_h = 3, k_w = 3, s_h = 1, s_w = 1,
                   padding = "SAME", get_vars_name=False, active_func=None,norm=None,
                   is_trainable=True, stddev = 0.02, weight_decay=0.0005, name = 'm4_conv'):
    '''
    Introduction:实现一个卷积模块，w * x + bias, 同时后面可以选择是否norm和active
                注意： norm在active之前
    :param input_: a tensor
    :param fiters: 卷积核的个数
    :param k_h: 卷积核的高度
    :param k_w: 卷积核的宽度
    :param s_h: 高度方向的步长
    :param s_w: 宽度方向的步长
    :param padding: 'SANE': 卷积后图像的长宽w=h= w/s
                    'VALID':卷积后图像的长宽w=h= (w-f+1)/s
    :param get_vars_name: bool:True of False
    :param active_func: str: 激活函数的名称
                        注意：不用激活函数时设置成None
    :param norm: str: norm函数的名称
                 注意：不用norm时，设置成None
    :param is_trainable: bool: True or False, norm防过拟合时需要
    :param stddev: 初始化权重的标准差
    :param name: str: 卷积模块的名称
    :return: 当get_vars_name为False， 返回一个tensor；
                            为True， 返回一个tensor和该层中变量列表
    '''
    with tf.variable_scope(name) as scope:
        l2_reg = tf.contrib.layers.l2_regularizer(weight_decay)
        batch, heigt, width, nc = input_.get_shape().as_list()
        w = tf.get_variable(name='filter', shape=[k_h, k_w, nc, fiters], regularizer=l2_reg,
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        bias = tf.get_variable(name='biases', shape=[fiters], regularizer=l2_reg, initializer=tf.constant_initializer(0.0))
        output_ = tf.nn.conv2d(input_, w, strides=[1, s_h, s_w, 1], padding=padding) + bias
        output_ = m4_norm_func(output_, is_trainable, name=norm)
        output_ = m4_active_function(output_, active_function=active_func)

        if get_vars_name:
            vars = tf.contrib.framework.get_variables(scope)
            return output_, vars
        else:
            return output_


def m4_max_pool(input_, ks=2, stride=2, padding='SAME', name='max_pool'):
    '''
    Introduction:实现 最大池化层
    :param input_:a tensor
    :param ks:池化核大小，默认2x2
    :param stride:池化，步长
    :param padding: "SAME"
    :param name: 池化层名称
    :return: a tensor
    '''

    return tf.nn.max_pool(input_, ksize=[1, ks, ks, 1], strides=[1, stride, stride, 1], padding=padding, name=name)

def m4_average_pool(input_, ks=2, stride=2, padding='SAME', name='average_pool'):
    '''
    Introduction:实现 平均池化层
    :param input_:a tensor
    :param ks:池化核大小，默认2x2
    :param stride:池化，步长
    :param padding: "SAME"
    :param name: 池化层名称
    :return: a tensor
    '''
    return tf.nn.avg_pool(input_, ksize=[1, ks, ks, 1], strides=[1, stride, stride, 1], padding=padding, name=name)

def m4_linear(input_, output, active_function=None, norm=None, get_vars_name=False, is_trainable=True,
              stddev=0.02, weight_decay=0.0005, name='fc'):
    '''
    Introduction:实现一个全连接模块， w * x + bias, 同时后面可以选择是否norm和active
                注意： norm在active之前
    :param input_: a tensor
    :param output: 期望输出全连接层的节点数
    :param active_function: str: 激活函数的名称
                            注意：不用激活函数时设置成None
    :param norm: norm函数的名称
                 注意：不用norm时，设置成None
    :param get_vars_name: bool:True of False
    :param is_trainable: bool: True or False, norm防过拟合时需要
    :param stddev: 初始化权重的标准差
    :param name: str: 该模块的名称
    :return: 当get_vars_name为False， 返回一个tensor；
                            为True， 返回一个tensor和该层中变量列表
    '''
    with tf.variable_scope(name) as scope:
        l2_reg = tf.contrib.layers.l2_regularizer(weight_decay)
        input_shape = input_.get_shape().as_list()
        w = tf.get_variable('w', [input_shape[-1], output], regularizer=l2_reg, initializer=tf.random_normal_initializer(stddev=stddev))
        biases = tf.get_variable('biases', [output], regularizer=l2_reg, initializer=tf.constant_initializer(0.0))
        conn = tf.matmul(input_, w) + biases
        output_ = m4_norm_func(conn, is_trainable, name=norm)
        output_ = m4_active_function(output_, active_function=active_function)
        if get_vars_name:
            vars = tf.contrib.framework.get_variables(scope)
            return output_, vars
        else:
            return output_

def m4_resblock(input_, fiters, k_h = 3, k_w = 3, s_h = 1, s_w = 1, is_downsample=False,
                   padding = "SAME", get_vars_name=False, active_func=None,norm=None,
                   is_trainable=True, stddev = 0.02, weight_decay=0.0005, name = 'resblock'):

    '''
    Introduction: 用于 “ResNet18” 和 “ResNet34” 的卷积模块. 具体的看resnet网络的介绍
    :param input_: a tensor
    :param fiters: 卷积核的个数
    :param k_h: 卷积核的高度
                注意：根据resnet卷积模块的要求，当is_downsample为true时，有的卷积层为起作用
                对应：resnet的定义看
    :param k_w: 卷积核的宽度
                注意：根据resnet卷积模块的要求，当is_downsample为true时，有的卷积层为起作用
                对应：resnet的定义看
    :param s_h: 高度方向的步长
                注意：根据resnet卷积模块的要求，当is_downsample为true时，有的卷积层为起作用
                对应：resnet的定义看
    :param s_w: 宽度方向的步长
                注意：根据resnet卷积模块的要求，当is_downsample为true时，有的卷积层为起作用
                对应：resnet的定义看
    :param is_downsample: bool: True of False, 图像尺寸减小
    :param padding: 'SANE': 卷积后图像的长宽w=h= w/s
                    'VALID':卷积后图像的长宽w=h= (w-f+1)/s
    :param get_vars_name: bool:True of False
    :param active_func: str: 激活函数的名称
                        注意：不用激活函数时设置成None
    :param norm: str: norm函数的名称
                 注意：不用norm时，设置成None
    :param is_trainable: bool: True or False, norm防过拟合时需要
    :param stddev: 初始化权重的标准差
    :param name: str: 卷积模块的名称
    :return: 当get_vars_name为False， 返回一个tensor；
                            为True， 返回一个tensor和该层中变量列表
    '''
    with tf.variable_scope(name) as scope:
        if is_downsample:
            x = m4_conv_layers(input_, fiters, k_h=k_h, k_w=k_w, s_h=2, s_w=2,
                               padding=padding, get_vars_name=get_vars_name, active_func=active_func, norm=norm,
                               is_trainable=is_trainable, stddev=0.02, weight_decay=weight_decay, name='conv_1')
            x = m4_conv_layers(x, fiters, k_h=k_h, k_w=k_w, s_h=s_h, s_w=s_w,
                               padding=padding, get_vars_name=get_vars_name, active_func=None, norm=norm,
                               is_trainable=is_trainable, stddev=0.02, weight_decay=weight_decay, name='conv_2')
            input_ = m4_conv_layers(input_, fiters, k_h=1, k_w=1, s_h=2, s_w=2,
                               padding=padding, get_vars_name=get_vars_name, active_func=None, norm=norm,
                               is_trainable=is_trainable, stddev=0.02, weight_decay=weight_decay, name='main_branch_downsample')

        else:
            x = m4_conv_layers(input_, fiters, k_h=k_h, k_w=k_w, s_h=s_h, s_w=s_w,
                       padding=padding, get_vars_name=get_vars_name, active_func=active_func, norm=norm,
                       is_trainable=is_trainable, stddev=0.02, weight_decay=weight_decay, name='conv_1')
            x = m4_conv_layers(x, fiters, k_h=k_h, k_w=k_w, s_h=s_h, s_w=s_w,
                               padding=padding, get_vars_name=get_vars_name, active_func=None, norm=norm,
                               is_trainable=is_trainable, stddev=0.02, weight_decay=weight_decay, name='conv_2')
        x = x + input_

        x = m4_active_function(x, active_function=active_func)
        if get_vars_name:
            vars = tf.contrib.framework.get_variables(scope)
            return x, vars
        else:
            return x

def m4_bottle_resblock(input_, fiters, k_h = 3, k_w = 3, s_h = 1, s_w = 1, is_downsample=False,
                   padding = "SAME", get_vars_name=False, active_func=None,norm=None,
                   is_trainable=True, stddev = 0.02, weight_decay=0.0005, name='m4_bottle_resblock'):
    '''
    Introduction: 用于 “ResNet50”,， “ResNet101” 和 “ResNet152” 的卷积模块
    :param input_: a tensor
    :param fiters: 卷积核的个数
    :param k_h: 卷积核的高度
                注意：根据resnet卷积模块的要求，当is_downsample为true时，有的卷积层为起作用
                对应：resnet的定义看
    :param k_w: 卷积核的宽度
                注意：根据resnet卷积模块的要求，当is_downsample为true时，有的卷积层为起作用
                对应：resnet的定义看
    :param s_h: 高度方向的步长
                注意：根据resnet卷积模块的要求，当is_downsample为true时，有的卷积层为起作用
                对应：resnet的定义看
    :param s_w: 宽度方向的步长
                注意：根据resnet卷积模块的要求，当is_downsample为true时，有的卷积层为起作用
                对应：resnet的定义看
    :param is_downsample: bool: True of False, 图像尺寸减小
    :param padding: 'SANE': 卷积后图像的长宽w=h= w/s
                    'VALID':卷积后图像的长宽w=h= (w-f+1)/s
    :param get_vars_name: bool:True of False
    :param active_func: str: 激活函数的名称
                        注意：不用激活函数时设置成None
    :param norm: str: norm函数的名称
                 注意：不用norm时，设置成None
    :param is_trainable: bool: True or False, norm防过拟合时需要
    :param stddev: 初始化权重的标准差
    :param name: str: 卷积模块的名称
    :return: 当get_vars_name为False， 返回一个tensor；
                            为True， 返回一个tensor和该层中变量列表
    '''
    with tf.variable_scope(name) as scope:

        if is_downsample:
            x = m4_conv_layers(input_, fiters, k_h=1, k_w=1, s_h=2, s_w=2,
                               padding=padding, get_vars_name=get_vars_name, active_func=active_func, norm=norm,
                               is_trainable=is_trainable, stddev=0.02, weight_decay=weight_decay, name='conv_1')
            x = m4_conv_layers(x, fiters, k_h=k_h, k_w=k_w, s_h=s_h, s_w=s_w,
                               padding=padding, get_vars_name=get_vars_name, active_func=active_func, norm=norm,
                               is_trainable=is_trainable, stddev=0.02, weight_decay=weight_decay, name='conv_2')
            x = m4_conv_layers(x, fiters * 4, k_h=1, k_w=1, s_h=s_h, s_w=s_w,
                               padding=padding, get_vars_name=get_vars_name, active_func=None, norm=norm,
                               is_trainable=is_trainable, stddev=0.02, weight_decay=weight_decay, name='conv_3')
            input_ = m4_conv_layers(input_, fiters * 4, k_h=1, k_w=1, s_h=2, s_w=2,
                               padding=padding, get_vars_name=get_vars_name, active_func=None, norm=norm,
                               is_trainable=is_trainable, stddev=0.02, weight_decay=weight_decay, name='main_branch_downsample')

        else:
            x = m4_conv_layers(input_, fiters, k_h=1, k_w=1, s_h=s_h, s_w=s_w,
                       padding=padding, get_vars_name=get_vars_name, active_func=active_func, norm=norm,
                       is_trainable=is_trainable, stddev=0.02, weight_decay=weight_decay, name='conv_1')
            x = m4_conv_layers(x, fiters, k_h=k_h, k_w=k_w, s_h=s_h, s_w=s_w,
                               padding=padding, get_vars_name=get_vars_name, active_func=active_func, norm=norm,
                               is_trainable=is_trainable, stddev=0.02, weight_decay=weight_decay, name='conv_2')
            x = m4_conv_layers(x, fiters * 4, k_h=1, k_w=1, s_h=s_h, s_w=s_w,
                               padding=padding, get_vars_name=get_vars_name, active_func=None, norm=norm,
                               is_trainable=is_trainable, stddev=0.02, weight_decay=weight_decay, name='conv_3')
            input_ = m4_conv_layers(input_, fiters * 4, k_h=1, k_w=1, s_h=1, s_w=1,
                                    padding=padding, get_vars_name=get_vars_name, active_func=None, norm=norm,
                                    is_trainable=is_trainable, stddev=0.02, weight_decay=weight_decay, name='main_branch_keepdim')
        x = x + input_

        x = m4_active_function(x, active_function=active_func)
        if get_vars_name:
            vars = tf.contrib.framework.get_variables(scope)
            return x, vars
        else:
            return x