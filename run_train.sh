python run_train.py \
--gpu_assign='gpu' \
--is_train=True \
--dataset_dir='/gs/home/yangjb/My_Job/dataset/face/cartoon' \
--dataset_name='faces' \
--batch_size=128 \
--epoch=100 \
--img_size=[64, 64, 3] \
--tmp_result_name='1'