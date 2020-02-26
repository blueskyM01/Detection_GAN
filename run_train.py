import numpy as np
import tensorflow as tf
# 导入 keras 模型，不能使用 import keras，它导入的是标准的 Keras 库
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import argparse, os, time

import mx_networks_utils.mx_model as mx_model

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default='7', type=str, help="assign gpu")
parser.add_argument("--is_train", default=True, type=bool, help="train or test")
parser.add_argument("--dataset_dir", default='/gs/home/yangjb/My_Job/dataset/face/cartoon', type=str, help="dir of dataset")
parser.add_argument("--dataset_name", default='faces', type=str, help="name of dataset")
parser.add_argument("--label_dir", default=None, type=str, help="dir of label file")
parser.add_argument("--label_name", default=None, type=str, help="name of label file")
parser.add_argument("--batch_size", default=64, type=int, help="batch size")
parser.add_argument("--epoch", default=100, type=int, help="num of epoch")
parser.add_argument('--img_size', nargs=3, default=[64, 64, 3], type=int, action='store',
                    help='with, height, channel of input image')
parser.add_argument("--g_lr", default=0.00002, type=float, help="learning rate of G")
parser.add_argument("--d_lr", default=0.00002, type=float, help="learning rate of D")
parser.add_argument("--log_dir", default='./results/logs', type=str, help="dir to save log file")
parser.add_argument("--checkpoint_dir", default='./results/checkpoint', type=str, help="dir to save train reslut")
parser.add_argument("--g_image_dir", default='./g_image_save', type=str, help="dir to save generated image")
cfg = parser.parse_args()

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu  # 指定第  块GPU可用

    # -------------------------------获取GPU列表---------------------------
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print('-*-*-' * 10, 'GPUS of device:', '-*-*-' * 10, '\n', gpus, '\n', '-*-*-' * 24)
    # -------------------------------获取GPU列表---------------------------

    # ----------------设置不占用整块显卡， 用多少显存分配多少显存------------------------
    if gpus:
        try:
            # 设置 GPU 显存占用为按需分配
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print('-*-*-' * 10, 'GPUS of device:', '-*-*-' * 10)
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            print('-*-*-' * 10, 'GPUS of device:', '-*-*-' * 10)
        except RuntimeError as e:
            # 异常处理
            print(e)
    # ----------------设置不占用整块显卡， 用多少显存分配多少显存------------------------

    # 构建模型
    if not os.path.exists(os.path.join(cfg.log_dir, cfg.dataset_name)):
        os.makedirs(os.path.join(cfg.log_dir, cfg.dataset_name))

    if not os.path.exists(os.path.join(cfg.checkpoint_dir, cfg.dataset_name)):
        os.makedirs(os.path.join(cfg.checkpoint_dir, cfg.dataset_name))

    if not os.path.exists(os.path.join(cfg.g_image_dir, cfg.dataset_name)):
        os.makedirs(os.path.join(cfg.g_image_dir, cfg.dataset_name))

    DG_model = mx_model.DetectionGAN(cfg)
    DG_model.build_model()