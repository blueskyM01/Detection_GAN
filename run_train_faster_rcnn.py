import numpy as np
import tensorflow as tf
# 导入 keras 模型，不能使用 import keras，它导入的是标准的 Keras 库
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import argparse, os, time

import mx_networks_utils.mx_faster_rcnn_model as mx_faster_rcnn_model

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default='0,1,7', type=str, help="assign gpu")
parser.add_argument("--is_train", default=True, type=bool, help="train or test")
parser.add_argument("--dataset_dir", default='', type=str, help="dir of dataset")
parser.add_argument("--dataset_name", default='coco', type=str, help="name of dataset")
parser.add_argument("--label_dir", default='./Train_labels', type=str, help="dir of label file")
parser.add_argument("--label_name", default='trainn_coco1.txt', type=str, help="name of label file")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--epoch", default=200, type=int, help="num of epoch")
parser.add_argument('--img_size', nargs=3, default=[416, 416, 3], type=int, action='store',
                    help='with, height, channel of input image')
parser.add_argument("--lr", default=0.00002, type=float, help="learning rate of G")
parser.add_argument("--log_dir", default='./logs', type=str, help="dir to save log file")
parser.add_argument("--checkpoint_dir", default='./checkpoint', type=str, help="dir to save train reslut")
parser.add_argument("--results_dir", default='./results', type=str, help="results dir")
parser.add_argument("--generate_image_dir", default='./generate_image', type=str, help="dir to save generated image")
parser.add_argument("--tmp_result_name", default='faster-rcnn', type=str, help="tmp file save dir")
cfg = parser.parse_args()

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu  # 指定第  块GPU可用
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # or any {'0', '1', '2'}
    # TF_CPP_MIN_LOG_LEVEL 取值 0 ： 0也是默认值，输出所有信息
    # TF_CPP_MIN_LOG_LEVEL 取值 1 ： 屏蔽通知信息
    # TF_CPP_MIN_LOG_LEVEL 取值 2 ： 屏蔽通知信息和警告信息
    # TF_CPP_MIN_LOG_LEVEL 取值 3 ： 屏蔽通知信息、警告信息和报错信息

    # 把模型的变量分布在哪个GPU上给打印出来
    tf.debugging.set_log_device_placement(True)

    # -------------------------------获取GPU列表---------------------------
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print('-*-*-' * 10, 'GPUS of device:', '-*-*-' * 10, '\n', gpus, '\n', '-*-*-' * 24)
    # -------------------------------获取GPU列表---------------------------

    # ----------------设置不占用整块显卡， 用多少显存分配多少显存------------------------
    if gpus:
        try:
            for gpu in gpus:
                # 设置 GPU 显存占用为按需分配
                tf.config.experimental.set_memory_growth(gpu, True)
                # 设置GPU可见,一般一个物理GPU对应一个逻辑GPU
                # tf.config.experimental.set_visible_devices(gpu, 'GPU')

            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            num_gpu = len(logical_gpus)
            print('-*-*-' * 10, 'GPUS of device:', '-*-*-' * 10)
            print(len(gpus), "Physical GPUs,", num_gpu, "Logical GPUs")
            print('-*-*-' * 10, 'GPUS of device:', '-*-*-' * 10)
        except RuntimeError as e:
            # 异常处理
            print(e)
    # ----------------设置不占用整块显卡， 用多少显存分配多少显存------------------------

    # 构建模型
    if not os.path.exists(os.path.join(cfg.results_dir, cfg.log_dir, cfg.tmp_result_name)):
        os.makedirs(os.path.join(cfg.results_dir, cfg.log_dir, cfg.tmp_result_name))

    if not os.path.exists(os.path.join(cfg.results_dir, cfg.checkpoint_dir, cfg.tmp_result_name)):
        os.makedirs(os.path.join(cfg.results_dir, cfg.checkpoint_dir, cfg.tmp_result_name))

    if not os.path.exists(os.path.join(cfg.results_dir, cfg.generate_image_dir, cfg.tmp_result_name)):
        os.makedirs(os.path.join(cfg.results_dir, cfg.generate_image_dir, cfg.tmp_result_name))

    # 创建一个MirroredStrategy分发数据和计算图
    # This will create a MirroredStrategy instance which will use all the GPUs that are visible to TensorFlow
    strategy = tf.distribute.MirroredStrategy()
    # only some of the GPUs on your machine, you can do so like this:
    # strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

    faster_rcnn_model = mx_faster_rcnn_model.FasterRCNN(cfg, strategy)
    faster_rcnn_model.build_model()