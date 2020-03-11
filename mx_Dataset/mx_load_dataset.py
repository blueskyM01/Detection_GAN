import tensorflow as tf
import numpy as np
import time, os, cv2, glob
import multiprocessing, datetime, random
import sys
sys.path.append('../')

from mx_networks_utils import mx_utils

class mx_DatasetLoader:
    def __init__(self, image_shape=[416,416,3]):
        '''

        :param image_shape:
        '''
        self.image_h = image_shape[0]
        self.image_w = image_shape[1]
        self.image_nc = image_shape[2]

    def mx_dataset_load(self, dataset_dir, dataset_name, label_dir=None, label_name=None, shuffle=True,
                        shuffle_size=1000, batch_size=1, epoch=100):
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

        lines = self.get_label(dataset_name, label_dir, label_name)
        dataset_len = len(lines)
        num_gt, img_path, boxes, labels, img_width, img_height = self.split_to_list(lines)
        dataset = tf.data.Dataset.from_tensor_slices((num_gt, img_path, boxes,
                                                      labels, img_width, img_height))
        n_map_threads = multiprocessing.cpu_count()

        if shuffle:
            dataset = dataset.shuffle(shuffle_size)

        dataset = dataset.map(self.preprocess_with_label, num_parallel_calls=n_map_threads)

        # 参数drop_remaindar，用于标示是否对于最后一个batch如果数据量达不到batch_size时保留还是抛弃
        # buffer_size仅仅影响生成下一个元素的时间,通过将数据预处理与下游计算重叠。
        # 典型地，在管道末尾增加一个prefetch buffer（也许仅仅是单个样本），
        # 但更复杂的管道能够从额外的prefetching获益，尤其是当生成单个元素的时间变化时。
        dataset = dataset.batch(batch_size, drop_remainder=True).repeat(epoch).prefetch(buffer_size=10)

        return dataset, dataset_len

    def preprocess_with_label(self, num_gt, img_path, boxes, labels, img_width, img_height):

        # x: 图片的路径 List， y：图片的数字编码 List
        img = tf.io.read_file(img_path) # 根据路径读取图片
        img = tf.image.decode_jpeg(img, channels=3) # 图片解码
        img = tf.image.resize(img, [self.image_h, self.image_w]) # 图片缩放

        w_scale = float(img_width / self.image_w)
        h_scale = float(img_height / self.image_h)

        x_min = boxes[:, 0:1] / w_scale
        y_min = boxes[:, 1:2] / h_scale
        x_max = boxes[:, 2:3] / w_scale
        y_max = boxes[:, 3:4] / h_scale
        boxes = tf.concat([x_min, y_min, x_max, y_max], axis=-1)
        # img: [0,255]=> -1~1
        img = tf.cast(img, dtype=tf.float32) / 255. * 2 - 1

        # 转换成张量
        num_gt = tf.convert_to_tensor(num_gt)
        boxes = tf.convert_to_tensor(boxes) # 转换成张量
        labels = tf.convert_to_tensor(labels) # 转换成张量
        img_width = tf.convert_to_tensor(img_width)
        img_height = tf.convert_to_tensor(img_height)
        z = tf.random.normal([128])
        return num_gt, img, boxes, labels, img_width, img_height, z

    def get_label(self, dataset_name, label_dir, label_name):
        '''
        读取的".txt"文件的每行存储格式为： [num_gt, image_absolute_path, img_width, img_height, box_1, box_2, ..., box_n]
                                  Box_x format: label_index x_min y_min x_max y_max. (The origin of coordinates is at the left top corner, left top => (xmin, ymin), right bottom => (xmax, ymax).)
                                  num_gt： iis the number of ground true boxes.
                                  label_index： is in range [0, class_num - 1].
                                  For example:
                                  0 xxx/xxx/a.jpg 1920 1080 0 453 369 473 391 1 588 245 608 268
                                  1 xxx/xxx/b.jpg 1920 1080 1 466 403 485 422 2 793 300 809 320
        :param dataset_name:
        :param label_dir:
        :param label_name:
        :return: lines： 将".txt"文件的每行变成列表， 存储到lines这个大列表中
        '''
        label_path = os.path.join(label_dir, label_name)
        lines = []
        num_gt_list = []
        with open(label_path, 'r') as f:
            line = f.readline()
            while line:
                info = line.rstrip('\n').split(' ')
                num_gt_list.append(int(info[0]))
                lines.append(info)
                line = f.readline()
            num_gt_max = max(num_gt_list)
            print('max ground true boxes in {} dataset is {}!'.format(dataset_name, num_gt_max))

            # align
            for anno in lines:
                length = len(anno)
                align_num = 4 + 5 * num_gt_max
                if length < align_num:
                    for _ in range(align_num - length):
                        anno.append(str(0))
        random.shuffle(lines)
        return lines

    def parse_line(self, line):
        '''
        功能： 获取每张图像上所有的标注矩形框和标注类别
        :param line: ".txt"文件的每行变成列表（每个元素都是字符串）
        :return: num_gt： bounding boxes 的数量
                 img_path： 图像的存储路径（绝对路径）
                 boxes： numpy格式： [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax], ....]
                 labels：numpy格式： [label1, label2, ....],与上面的box一一对应
                 img_width： 图像的宽度
                 img_height： 图像的高度
        '''
        num_gt = int(line[0])
        img_path = line[1]
        img_width = int(line[2])
        img_height = int(line[3])
        boxes = []
        labels = []
        s = line[4:]
        for i in range(len(s) // 5):
            label, xmin, ymin, xmax, ymax = int(s[i*5]), float(s[i*5+1]), float(s[i*5+2]), float(s[i*5+3]), float(s[i*5+4])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
        boxes = np.asarray(boxes, np.float32)
        labels = np.asarray(labels, np.int32)
        return num_gt, img_path, boxes, labels, img_width, img_height

    def split_to_list(self, lines):
        num_gt_list = []
        img_path_list = []
        boxes_list = []
        labels_list = []
        img_width_list = []
        img_height_list = []
        for line in lines:
            num_gt, img_path, boxes, labels, img_width, img_height = self.parse_line(line)
            num_gt_list.append(num_gt)
            img_path_list.append(img_path)
            boxes_list.append(boxes)
            labels_list.append(labels)
            img_width_list.append(img_width)
            img_height_list.append(img_height)
        num_gt_np = np.asarray(num_gt_list, dtype=np.int32)
        img_path_np = np.asarray(img_path_list, dtype=np.str)
        boxes_np = np.asarray(boxes_list, dtype=np.float32)
        labels_np = np.asarray(labels_list, dtype=np.int32)
        img_width_np = np.asarray(img_width_list, dtype=np.int32)
        img_height_np = np.asarray(img_height_list, dtype=np.int32)
        return num_gt_np, img_path_np, boxes_np, labels_np, img_width_np, img_height_np

if __name__ == '__main__':
    dataset_dir = ''
    dataset_name = 'coco'
    image_shape = [416,416,3]
    label_dir = '../Train_labels'
    label_name = 'trainn_coco1.txt'
    classes_file = '../Train_labels/coco.names'
    data_loader = mx_DatasetLoader(image_shape)
    dataset, dataset_len = data_loader.mx_dataset_load(dataset_dir, dataset_name, label_dir=label_dir, label_name=label_name,
                                                       shuffle=True, shuffle_size=1000, batch_size=1, epoch=100)
    print('dataset_size:', dataset_len)
    db_train = iter(dataset)
    counter = 0
    classes = mx_utils.get_classes(classes_file)
    for epoch in range(100):
        for step in range(10):
            starttime = datetime.datetime.now()
            num_gt, img, boxes, labels, img_width, img_height, z = next(db_train)

            img = img.numpy()
            boxes = boxes.numpy()
            num_gt = num_gt.numpy()
            labels = labels.numpy()
            img_width = img_width.numpy()
            img_height = img_height.numpy()

            endtime = datetime.datetime.now()
            timediff = (endtime - starttime).total_seconds()
            print('time:', timediff)

            for i in range(1):
                img = ((img[i] + 1.0) * 127.5).astype(np.uint8)
                boxes = boxes[i]
                num_gt = num_gt[i]
                labels = labels[i]
                img_width = img_width[i]
                img_height = img_height[i]

                boxes = boxes[:num_gt, :]
                labels = labels[:num_gt]

                for box, label in zip(boxes, labels):
                    x0 = int(box[0])
                    y0 = int(box[1])
                    x1 = int(box[2])
                    y1 = int(box[3])
                    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
                    cv2.putText(img, classes[label], (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                cv2.imwrite('../tmp/'+str(counter) + '.jpg', img)




            counter += 1