import tensorflow as tf
import numpy as np
import time, os, cv2, glob
import sys
sys.path.append('../')
from mx_networks_utils import mx_utils
import multiprocessing, datetime

os.environ["CUDA_VISIBLE_DEVICES"] = '4'

def mx_dataset_load(dataset_dir, dataset_name, label_dir=None, label_name=None, shuffle=True,
                    shuffle_size=10000, batch_size=64, repeat=100):

    dataset_path = os.path.join(dataset_dir, dataset_name)
    img_path_list = glob.glob(dataset_path + '/*.jpg') + \
               glob.glob(dataset_path + '/*.png') + \
               glob.glob(dataset_path + '/*.jpeg') + \
               glob.glob(dataset_path + '/*.bmp')

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
        dataset = dataset.map(preprocess_without_label, num_parallel_calls=n_map_threads)
    else:
        dataset = dataset.map(preprocess_with_label, num_parallel_calls=n_map_threads)

    dataset = dataset.batch(batch_size).repeat(repeat)

    return dataset, dataset_len



def preprocess_with_label(x,y):
    # x: 图片的路径 List， y：图片的数字编码 List
    x = tf.io.read_file(x) # 根据路径读取图片
    x = tf.image.decode_jpeg(x, channels=3) # 图片解码
    x = tf.image.resize(x, [244, 244]) # 图片缩放
    # 数据增强
    # x = tf.image.random_flip_up_down(x)
    x= tf.image.random_flip_left_right(x) # 左右镜像
    x = tf.image.random_crop(x, [224, 224, 3]) # 随机裁剪
    # 转换成张量
    # x: [0,255]=> 0~1
    x = tf.cast(x, dtype=tf.float32) / 255.
    # 0~1 => D(0,1)
    x = normalize(x) # 标准化
    y = tf.convert_to_tensor(y) # 转换成张量
    return x, y

def preprocess_without_label(x):
    # x: 图片的路径 List， y：图片的数字编码 List
    x = tf.io.read_file(x) # 根据路径读取图片
    x = tf.image.decode_jpeg(x, channels=3) # 图片解码
    x = tf.image.resize(x, [64, 64]) # 图片缩放
    # 数据增强
    # x = tf.image.random_flip_up_down(x)
    x= tf.image.random_flip_left_right(x) # 左右镜像
    x = tf.image.random_crop(x, [64, 64, 3]) # 随机裁剪
    # 转换成张量
    # x: [0,255]=> -1~1
    x = tf.cast(x, dtype=tf.float32) / 255. * 2 - 1

    return x



if __name__ == '__main__':
    dataset_dir = '/gs/home/yangjb/My_Job/dataset/face/cartoon'
    dataset_name = 'faces'
    dataset, dataset_len = mx_dataset_load(dataset_dir, dataset_name, label_dir=None, label_name=None, shuffle=True,
                    shuffle_size=1000, batch_size=64, repeat=100)
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

