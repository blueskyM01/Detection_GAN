import numpy as np
import tensorflow as tf
# 导入 keras 模型，不能使用 import keras，它导入的是标准的 Keras 库
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import argparse, os, time, cv2

import mx_networks_utils.mx_ops as mx_ops
import mx_networks_utils.mx_utils as mx_utils
import models.mx_faster_rcnn as mx_faster_rcnn

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default='0', type=str, help="assign gpu")
parser.add_argument("--is_train", default=False, type=bool, help="train or test")
parser.add_argument("--class_path", default='./Train_labels/voc2007.names', type=str, help="path of class file")
parser.add_argument("--pre_model", default='H:/demo/NewFolder/faster_rcnn_0050.ckpt',
                    type=str, help="file of pre trained model")
parser.add_argument('--img_size', nargs=3, default=[256, 256, 3], type=int, action='store',
                    help='with, height, channel of input image')
parser.add_argument("--results_dir", default='./results', type=str, help="results dir")
parser.add_argument("--test_result_name", default='test', type=str, help="path of class file")
parser.add_argument("--image_path", default='test_image/2.jpg', type=str, help="path of test image")

cfg = parser.parse_args()

print('*******************************input parser*******************************')
print(' --gpu:{} \n --is_train:{} \
      \n --class_path:{} \n --img_size:{} \
      \n --results_dir:{} \n --pre_model:{}  --test_result_name:{} \n'.format(
    cfg.gpu, cfg.is_train, cfg.class_path, cfg.img_size, cfg.results_dir, cfg.pre_model, cfg.test_result_name))

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu  # 指定第  块GPU可用
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
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
    if not os.path.exists(os.path.join(cfg.results_dir, cfg.test_result_name)):
        os.makedirs(os.path.join(cfg.results_dir, cfg.test_result_name))

    image_name_file = 'E:/dataset/PASCAL_VOC/VOCtrainval_06-Nov-2007/VOCdevkit/voc_train_val.txt'
    image_name_list = []
    with open(image_name_file, 'r') as f:
        line = f.readline()
        while line:
            info = line.rstrip('\n').split(' ')
            image_name_list.append(info[1])
            line = f.readline()

    classes = mx_utils.get_classes(cfg.class_path)
    num_classes = len(classes) + 1
    anchor_scales = [128, 256, 512]
    anchor_ratios = [0.5, 1., 2.]
    rpn_batch_size = 256
    rpn_pos_frac = 0.5
    rpn_pos_iou_thr = 0.7
    rpn_neg_iou_thr = 0.3
    rpn_proposal_count = 2000
    rpn_nms_thr = 0.7
    pooling_size = (7, 7)
    rcnn_batch_size = 256
    rcnn_pos_frac = 0.25
    rcnn_pos_iou_thr = 0.5
    rcnn_neg_iou_thr = 0.5
    rcnn_min_confidence = 0.4
    rcnn_nms_thr = 0.3
    rcnn_max_instance = 100


    faster_rcnn = mx_faster_rcnn.Faster_RCNN(cfg,
                                             num_classes=num_classes,
                                             anchor_scales=anchor_scales,
                                             anchor_ratios=anchor_ratios,
                                             rpn_batch_size=rpn_batch_size,
                                             rpn_pos_frac=rpn_pos_frac,
                                             rpn_pos_iou_thr=rpn_pos_iou_thr,
                                             rpn_neg_iou_thr=rpn_neg_iou_thr,
                                             rpn_proposal_count=rpn_proposal_count,
                                             rpn_nms_thr=rpn_nms_thr,
                                             pooling_size=pooling_size,
                                             rcnn_batch_size=rcnn_batch_size,
                                             rcnn_pos_frac=rcnn_pos_frac,
                                             rcnn_pos_iou_thr=rcnn_pos_iou_thr,
                                             rcnn_neg_iou_thr=rcnn_neg_iou_thr,
                                             rcnn_min_confidence=rcnn_min_confidence,
                                             rcnn_nms_thr=rcnn_nms_thr,
                                             rcnn_max_instance=rcnn_max_instance)



    if os.path.exists(cfg.pre_model + '.index') == True:
        faster_rcnn.load_weights(cfg.pre_model)
        print('*' * 10, 'load pre-model {} successful!'.format(cfg.pre_model), '*' * 10)
    else:
        print('*' * 10, 'no pre-model..................', '*' * 10)
    time.sleep(1)

    for image_path in (image_name_list):


        img = cv2.imread(image_path)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (cfg.img_size[1], cfg.img_size[0]))
        image = image.astype(np.float) / 127.5 - 1.0
        image = np.array([image])
        image = tf.convert_to_tensor(image)
        image = tf.cast(image, tf.float32)
        final_classes, final_boxes, final_scores = faster_rcnn.detect_single_image(image)


        final_boxes = final_boxes.numpy()
        final_classes = final_classes.numpy().tolist()
        final_scores = final_scores.numpy().tolist()

        or_h = img.shape[0]
        or_w = img.shape[1]

        ratio_h = or_h / float(cfg.img_size[0])
        ratio_w = or_w / float(cfg.img_size[1])

        for box, cls, score in zip(final_boxes, final_classes, final_scores):
            y0 = int(box[0] * cfg.img_size[0] * ratio_h)
            x0 = int(box[1] * cfg.img_size[1] * ratio_w)
            y1 = int(box[2] * cfg.img_size[0] * ratio_h)
            x1 = int(box[3] * cfg.img_size[1] * ratio_w)
            score = round(score, 2)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
            cv2.putText(img, classes[cls] + ':' + str(score), (x0+2, y0+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.imshow('result', img)
        cv2.waitKey(1000)

    # img_path = os.path.join(
    #     os.path.join(cfg.results_dir, cfg.test_result_name))
    # cv2.imwrite(img_path + '/'  + 'detect.jpg', img)