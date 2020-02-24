# Introduction

## step1: generate train label
````
    cd /the project path
    python mx_Dataset/mx_coco_label_maker.py --dataset_dir='/gs/home/yangjb/My_Job/dataset/coco' --dataset_name='train2017' --label_dir='/gs/home/yangjb/My_Job/dataset/coco/annotations' --label_name='instances_train2017.json' --save_path='./Train_labels/trian_coco.txt'
````
