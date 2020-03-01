import tensorflow as tf
import numpy as np
import time, os, cv2, glob
import sys
sys.path.append('../')
from mx_networks_utils import mx_utils
import multiprocessing, datetime, random

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

class mx_DatasetLoader:
    def __init__(self, image_shape=[64,64,3]):
        '''

        :param image_shape:
        '''
        self.image_h = image_shape[0]
        self.image_w = image_shape[1]
        self.image_nc = image_shape[2]

    def mx_dataset_load(self, dataset_dir, dataset_name, label_dir=None, label_name=None, shuffle=True,
                        shuffle_size=1000, batch_size=64, epoch=100):
        '''

        :param dataset_dir:
        :param dataset_name:
        :param label_dir:
        :param label_name:
        :param shuffle:
        :param shuffle_size:
        :param batch_size:
        :param repeat:
        :return:
        '''
        dataset_path = os.path.join(dataset_dir, dataset_name)
        img_path_list = glob.glob(dataset_path + '/*.jpg') + \
                   glob.glob(dataset_path + '/*.png') + \
                   glob.glob(dataset_path + '/*.jpeg') + \
                   glob.glob(dataset_path + '/*.bmp')

        random.shuffle(img_path_list) # 打乱数据集顺序

        dataset_len = len(img_path_list)

        if label_dir==None or label_name==None:
            print('None label....')
            memory_data = img_path_list
        else:
            label_path = os.path.join(label_dir, label_name)
            label_list = []
            memory_data = (img_path_list, label_list)

        dataset = tf.data.Dataset.from_tensor_slices(memory_data)
        n_map_threads = multiprocessing.cpu_count()

        if shuffle:
            dataset = dataset.shuffle(shuffle_size)

        if label_dir == None or label_name == None:
            dataset = dataset.map(self.preprocess_without_label, num_parallel_calls=n_map_threads)
        else:
            dataset = dataset.map(self.preprocess_with_label, num_parallel_calls=n_map_threads)

        # 参数drop_remaindar，用于标示是否对于最后一个batch如果数据量达不到batch_size时保留还是抛弃
        # buffer_size仅仅影响生成下一个元素的时间,通过将数据预处理与下游计算重叠。
        # 典型地，在管道末尾增加一个prefetch buffer（也许仅仅是单个样本），
        # 但更复杂的管道能够从额外的prefetching获益，尤其是当生成单个元素的时间变化时。
        dataset = dataset.batch(batch_size, drop_remainder=True).repeat(epoch).prefetch(buffer_size=10)

        return dataset, dataset_len

    def preprocess_with_label(self, x,y):
        # x: 图片的路径 List， y：图片的数字编码 List
        x = tf.io.read_file(x) # 根据路径读取图片
        x = tf.image.decode_jpeg(x, channels=3) # 图片解码
        x = tf.image.resize(x, [self.image_h, self.image_w]) # 图片缩放
        # 数据增强
        # x = tf.image.random_flip_up_down(x)
        x= tf.image.random_flip_left_right(x) # 左右镜像
        x = tf.image.random_crop(x, [self.image_h, self.image_w, self.image_nc]) # 随机裁剪
        # 转换成张量
        # x: [0,255]=> -1~1
        x = tf.cast(x, dtype=tf.float32) / 255. * 2 - 1

        y = tf.convert_to_tensor(y) # 转换成张量
        return x, y

    def preprocess_without_label(self, x):
        # x: 图片的路径 List
        x = tf.io.read_file(x) # 根据路径读取图片
        x = tf.image.decode_jpeg(x, channels=3) # 图片解码
        x = tf.image.resize(x, [self.image_h, self.image_w]) # 图片缩放
        # 数据增强
        # x = tf.image.random_flip_up_down(x)
        x= tf.image.random_flip_left_right(x) # 左右镜像
        x = tf.image.random_crop(x, [self.image_h, self.image_w, self.image_nc]) # 随机裁剪
        # 转换成张量
        # x: [0,255]=> -1~1
        x = tf.cast(x, dtype=tf.float32) / 255. * 2 - 1
        z = tf.random.normal([128])

        return x, z



if __name__ == '__main__':
    dataset_dir = '/gs/home/yangjb/My_Job/dataset/face/cartoon'
    dataset_name = 'faces'
    image_shape = [64,64,3]
    data_loader = mx_DatasetLoader(image_shape)
    dataset, dataset_len = data_loader.mx_dataset_load(dataset_dir, dataset_name, label_dir=None, label_name=None,
                                                       shuffle=True, shuffle_size=1000, batch_size=64, epoch=100)
    print('dataset_size:', dataset_len)
    db_train = iter(dataset)
    counter = 0
    for epoch in range(100):
        for step in range(10):
            starttime = datetime.datetime.now()
            batch_x = next(db_train)
            endtime = datetime.datetime.now()
            timediff = (endtime - starttime).total_seconds()
            print('time:', timediff)
            batch_x = batch_x.numpy()

            img = mx_utils.m4_image_save_cv(batch_x, rows=8, zero_mean=True)

            cv2.imwrite('../mx_image_save/' + '%06d' % (counter) + '.jpg', img)

            counter += 1

