import os, json, cv2
import numpy as np
from collections import defaultdict
import os
import argparse
def read_annotation(dataset_dir, dataset_name, label_dir, label_name, save_path):
    '''
    功能： Generate train.txt/val.txt/test.txt files One line for one image, in the format like：
          image_index, image_absolute_path, img_width, img_height, box_1, box_2, ... box_n.
          Box_x format: label_index x_min y_min x_max y_max.
                        (The origin of coordinates is at the left top corner, left top => (xmin, ymin), right bottom => (xmax, ymax).)
          image_index: is the line index which starts from zero.
          label_index: is in range [0, class_num - 1].
          For example:
          0 xxx/xxx/a.jpg 1920 1080 0 453 369 473 391 1 588 245 608 268
          1 xxx/xxx/b.jpg 1920 1080 1 466 403 485 422 2 793 300 809 320
    :param dataset_dir: 具体的图像文件夹存储的目录
    :param dataset_name: 图像文件夹的名称
    :param label_dir: “.json”所在的目录
    :param label_name: “.json”的名称
    :param save_path: 生成的“.txt"存储的路径
    :return:
    '''
    file_path = os.path.join(label_dir, label_name)
    image_data = []
    boxes_data = []
    name_box_id = defaultdict(list) # 创建一个字典，值的type是list
    with open(file_path, encoding='utf-8') as file:
        data = json.load(file)
        annotations = data['annotations']
        for ant in annotations:
            id = ant['image_id']
            name = os.path.join(dataset_dir, dataset_name, '%012d.jpg' % id)
            cat = ant['category_id']

            # 80个类
            if cat >= 1 and cat <= 11:
                cat = cat - 1
            elif cat >= 13 and cat <= 25:
                cat = cat - 2
            elif cat >= 27 and cat <= 28:
                cat = cat - 3
            elif cat >= 31 and cat <= 44:
                cat = cat - 5
            elif cat >= 46 and cat <= 65:
                cat = cat - 6
            elif cat == 67:
                cat = cat - 7
            elif cat == 70:
                cat = cat - 9
            elif cat >= 72 and cat <= 82:
                cat = cat - 10
            elif cat >= 84 and cat <= 90:
                cat = cat - 11

            '''
            ant['bbox']: ant中只有一个box: [x,y,width,height]
            [ant['bbox'], cat]: [[x,y,width,height],cat]
            name_box_id[name]:
                            [[[x,y,width,height],cat],
                                        .
                                        .
                                        .
                             [[x,y,width,height],cat],
                             [[x,y,width,height],cat]]
            '''
            name_box_id[name].append([ant['bbox'], cat])


        # name_box_label_list = []
        f = open(save_path, 'w')
        counter = 0
        for key in name_box_id.keys():
            elem = []
            elem.append(counter) # image_index
            counter += 1
            elem.append(key) # image_absolute_path

            img = cv2.imread(key)
            width = img.shape[1]
            height = img.shape[0]
            elem.append(width)
            elem.append(height)

            boxes = []
            box_infos = name_box_id[key]
            for info in box_infos:
                x_min = info[0][0]
                y_min = info[0][1]
                x_max = x_min + info[0][2]
                y_max = y_min + info[0][3]
                boxes.append(info[1])
                boxes.append(x_min)
                boxes.append(y_min)
                boxes.append(x_max)
                boxes.append(y_max)

            elem = elem + boxes
            for index in range(len(elem)):
                if index == 1:
                    f.write(elem[index] + ' ')

                elif index == (len(elem) - 1):
                    f.write(str(round(elem[index], 2)) + '\n')

                else:
                    f.write(str(round(elem[index], 2)) + ' ')
            print('num:', counter)
        f.close()

parser = argparse.ArgumentParser()

# -----------------------------m4_BE_GAN_network-----------------------------
parser.add_argument("--dataset_dir", default='/media/yang/F/DataSet/Tracking', type=str, help="the dir of dataset")
parser.add_argument("--dataset_name", default='val2017', type=str, help="the name of dataset")
parser.add_argument("--label_dir", default='/media/yang/F/DataSet/Tracking/annotations_trainval2017/annotations',
                    type=str, help="the dir of label")
parser.add_argument("--label_name", default='instances_val2017.json', type=str, help="the name of label")
parser.add_argument("--save_path", default='/media/yang/F/DataSet/Tracking/val.txt', type=str,
                    help="the path to save generate label")
cfg = parser.parse_args()

read_annotation(dataset_dir=cfg.dataset_dir,
                dataset_name=cfg.dataset_name,
                label_dir=cfg.label_dir,
                label_name=cfg.label_name,
                save_path=cfg.save_path)

# python mx_coco_label_maker.py --dataset_dir='/gs/home/yangjb/My_Job/dataset/coco' --dataset_name='train2017' --label_dir='/gs/home/yangjb/My_Job/dataset/coco/annotations' --label_name='instances_train2017.json' --save_path='./Train_labels/trian_coco.txt'