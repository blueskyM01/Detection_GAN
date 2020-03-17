import cv2, time, os, random
import numpy as np
import tensorflow as tf
import logging

def get_label(label_dir, label_name):
    '''
    读取的".txt"文件的每行存储格式为： [image_index, image_absolute_path, img_width, img_height, label_index, box_1, label_index, box_2, ..., label_index, box_n]
                              Box_x format: label_index x_min y_min x_max y_max. (The origin of coordinates is at the left top corner, left top => (xmin, ymin), right bottom => (xmax, ymax).)
                              image_index： is the line index which starts from zero.
                              label_index： is in range [0, class_num - 1].
                              For example:
                              0 xxx/xxx/a.jpg 1920 1080 0 453 369 473 391 1 588 245 608 268
                              1 xxx/xxx/b.jpg 1920 1080 1 466 403 485 422 2 793 300 809 320

    :param label_dir:
    :param label_name:
    :return: lines： 将".txt"文件的每行变成列表， 存储到lines这个大列表中
    '''
    label_path = os.path.join(label_dir, label_name)
    lines = []
    with open(label_path, 'r') as f:
        line = f.readline()
        while line:
            lines.append(line.rstrip('\n').split(' '))
            line = f.readline()
    random.shuffle(lines)
    return lines


def get_classes(class_file):
    classes = []
    with open(class_file, 'r') as f:
        line = f.readline()
        while line:
            classes.append(line.rstrip('\n'))
            line = f.readline()
    return classes


def parse_line(line):
    '''
    功能： 获取每张图像上所有的标注矩形框和标注类别
    :param line: ".txt"文件的每行变成列表（每个元素都是字符串）
    :return: img_idx： 数据集中的第几场图像编号
             img_path： 图像的存储路径（绝对路径）
             boxes： numpy格式： [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax], ....]
             labels：numpy格式： [label1, label2, ....],与上面的box一一对应
             img_width： 图像的宽度
             img_height： 图像的高度
    '''
    img_idx = int(line[0])
    img_path = line[1]
    img_width = int(line[2])
    img_height = int(line[3])
    boxes = []
    labels = []
    s = line[4:]
    for i in range(len(s) // 5):
        label, xmin, ymin, xmax, ymax = int(s[i * 5]), float(s[i * 5 + 1]), float(s[i * 5 + 2]), float(
            s[i * 5 + 3]), float(s[i * 5 + 4])
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)
    boxes = np.asarray(boxes, np.float32)
    labels = np.asarray(labels, np.int64)
    return img_idx, img_path, boxes, labels, img_width, img_height


def mx_get_roi_boxes(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(img_gray, 127, 255, 0)
    image, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    roi_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        xmin, ymin, xmax, ymax = x, y, x + w, y + h
        roi_boxes.append([xmin, ymin, xmax, ymax])
    roi_boxes_np = np.array(roi_boxes)
    return roi_boxes_np


def m4_image_save_cv(images, rows=4, zero_mean=True):
    # introduction: a series of images save as a picture
    # image: 4 dims
    # rows: how many images in a row
    # cols: how many images in a col
    # zero_mean:

    if zero_mean:
        images = images * 127.5 + 127.5
    if images.dtype != np.uint8:
        images = images.astype(np.uint8)
    img_num, img_height, img_width, nc = images.shape
    h_nums = rows
    w_nums = img_num // h_nums
    merge_image_height = h_nums * img_height
    merge_image_width = w_nums * img_width
    merge_image = np.ones([merge_image_height, merge_image_width, nc], dtype=np.uint8)
    for i in range(h_nums):
        for j in range(w_nums):
            merge_image[i * img_height:(i + 1) * img_height, j * img_width:(j + 1) * img_width] = images[
                i * w_nums + j]

    merge_image = cv2.cvtColor(merge_image, cv2.COLOR_BGR2RGB)  # cv2默认为bgr顺序
    return merge_image


def mx_draw_boundingboxes(img, boxes, labels, filename):
    '''

    :param img: -1~1
    :param boxes: np.array, shape=[-1,4] (xmin, ymin, xmax, ymax)
    :param labels: np.array, shape=(-1,)
    :param filename:
    :return:
    '''
    # -1~1 ---> 0~255
    img = img * 127.5 + 127.5
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2默认为bgr顺序
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    for box, label in zip(boxes, labels):
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(img, str(label), (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imwrite(filename, img)


def conver_to_roi_image(num_gt, imgs, boxes):
    '''
    Introduction: 生成roi图像
    :param num_gt: shape=(batch_size,), int
    :param imgs: shape=(batch_size, h, w, 3) -1~1
    :param boxes: [batch_size, -1, 4] [x0, y0, x1, y1]
    :return:
    '''
    roi_imgs = []
    for idx in range(imgs.shape[0]):
        box = boxes[idx]
        img = imgs[idx] * 0.5
        num = num_gt[idx]

        box_val = box[:num, :]

        for b in box_val:
            x0 = int(b[0])
            y0 = int(b[1])
            x1 = int(b[2])
            y1 = int(b[3])
            cv2.rectangle(img, (x0, y0), (x1, y1), (1, 1, 1), 4)
        roi_imgs.append(img)
    return np.array(roi_imgs)


def log_creater(output_dir, filenale):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log_name = '{}.log'.format(time.strftime(filenale))
    final_log_file = os.path.join(output_dir, log_name)
    # creat a log
    log = logging.getLogger('train_log')
    log.setLevel(logging.DEBUG)
    # FileHandler
    file = logging.FileHandler(final_log_file)
    file.setLevel(logging.DEBUG)

    # StreamHandler
    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)
    return log, file, stream, final_log_file

def log_write(log, file, stream, final_log_file, formatter):
    # Formatter

    # setFormatter
    file.setFormatter(formatter)
    stream.setFormatter(formatter)
    # addHandler
    log.addHandler(file)
    log.addHandler(stream)

    log.info('creating {}'.format(final_log_file))