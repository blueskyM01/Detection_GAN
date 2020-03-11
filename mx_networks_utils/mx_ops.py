import  os, time
import  numpy as np
import  tensorflow as tf
from    tensorflow import keras
from tensorflow import random

def celoss_ones(logits):
    # 计算属于与标签为1的交叉熵
    y = tf.ones_like(logits)
    loss = keras.losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)


def celoss_zeros(logits):
    # 计算属于与便签为0的交叉熵
    y = tf.zeros_like(logits)
    loss = keras.losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)

def d_loss_fn(d_fake_logits, d_real_logits):
    # 真实图片与1之间的误差
    d_loss_real = celoss_ones(d_real_logits)
    # 生成图片与0之间的误差
    d_loss_fake = celoss_zeros(d_fake_logits)
    # 合并误差
    loss = d_loss_fake + d_loss_real
    return loss


def g_loss_fn(d_fake_logits):
    # 计算生成图片与1之间的误差
    loss = celoss_ones(d_fake_logits)
    return loss

def w_loss_fn(d_fake_logits, d_real_logits):
    '''

    :param d_fake_logits:
    :param d_real_logits:
    :return:
    '''
    d_fake_logits = tf.reduce_mean(d_fake_logits, axis=1)
    d_real_logits = tf.reduce_mean(d_real_logits, axis=1)
    d_loss = -d_real_logits + d_fake_logits
    g_loss = -d_fake_logits
    return d_loss, g_loss

def w_AE_loss_fn(AE_real, AE_fake, real, fake, k_t):
    '''

    :param d_fake_logits:
    :param d_real_logits:
    :return:
    '''
    AE_real_loss = tf.reduce_mean(tf.abs(AE_real - real), axis=[1, 2, 3])
    AE_fake_loss = tf.reduce_mean(tf.abs(AE_fake - fake), axis=[1, 2, 3])
    d_loss = AE_real_loss - k_t * AE_fake_loss
    g_loss = AE_fake_loss
    return d_loss, g_loss, AE_real_loss

def gradient_penalty(D, real, fake, batch_size, is_train=True):
    alpha = random.uniform([batch_size, 1, 1, 1], 0., 1.)
    # diff = fake - real
    diff = real - fake
    inter = real + (alpha * diff)
    with tf.GradientTape() as t:
        t.watch(inter)
        pred = D(inter, is_train)
    grad = t.gradient(pred, [inter])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
    gp = tf.reduce_mean((slopes - 1.)**2, axis=1)
    return gp

def mx_generate_anchors(feature_shape, anchor_scale, anchor_ratio, image_size):
    '''
    Introduciton: 以feature-map的每个点为中心，各生成num_anchor_per_center个anchor-box，
                  因此共有feature_h * feature_w * num_anchor_per_center个。
                  num_anchor_per_center个anchor的计算：以anchor_scale=[128,256,512]，anchor_ratio=[0.5, 1., 2.0]为例：
                  anchor高度(heights)的计算：
                      每个anchor_scale 除以 每个anchor_ratio的开方，即：
                      128有3个: 128 / sqrt(0.5), 128 / sqrt(1), 128 / sqrt(2) ----> h1, h2, h3
                      256有3个: 256 / sqrt(0.5), 256 / sqrt(1), 256 / sqrt(2) ----> h4, h5, h6
                      512有3个: 256 / sqrt(0.5), 512 / sqrt(1), 512 / sqrt(2) ----> h7, h8, h9
                      共9个：[h1, h2, h3, h4, h5, h6, h7, h8, h9]

                  anchor宽度(widths)的计算：
                      每个anchor_scale 乘以 每个anchor_ratio的开方，即：
                      128有3个: 128 * sqrt(0.5), 128 * sqrt(1), 128 * sqrt(2) ----> w1, w2, w3
                      256有3个: 256 * sqrt(0.5), 256 * sqrt(1), 256 * sqrt(2) ----> w4, w5, w6
                      512有3个: 256 * sqrt(0.5), 512 * sqrt(1), 512 * sqrt(2) ----> w7, w8, w9
                      共9个：[w1, w2, w3, w4, w5, w6, w7, w8, w9]
                  因此，共有9个anchor，分别为[[h1, w1], [h2, w2], [h3, w3], [h4, w4], [h5, w5], [h6, w6], [h7, w7], [h8, w8], [h9, w9]]

    :param feature_shape: 特征层的大小，tensor, [h, w]， eg. feature_shape=[32, 16]
    :param anchor_scale: list，eg. anchor_scale=[128,256,512]
    :param anchor_ratio: list, eg. anchor_ratio=[0.5, 1., 2.0]
    :param image_size:  输入网络图像的大小，tensor, eg. 设图像大小为 image_size=[128, 128], [h, w]
    :return: anchor_boxes
            shape = [num_anchors, 4] = [feature_h * feature_w * num_anchor_per_center, 4]， 注意：坐标为原图上的坐标
            [y_min, x_min, y_max, x_max]
    '''
    # 若原图像大小为[128, 128]，映射的feature_map为[32, 16]，则feature_strides= [4,8]
    feature_stride = image_size / feature_shape

    anchor_scale = tf.constant(anchor_scale, dtype=tf.float32)
    anchor_ratio = tf.constant(anchor_ratio, dtype=tf.float32)
    anchor_ratio = tf.reshape(anchor_ratio, [-1, 1])

    heights = anchor_scale / tf.sqrt(anchor_ratio) # [[h1, h2, h3],
                                                   #  [h4, h5, h6],
                                                   #  [h7, h8, h9]]
    widths = anchor_scale * tf.sqrt(anchor_ratio)  # [[w1, w2, w3],
                                                   #  [w4, w5, w6],
                                                   #  [w7, w8, w9]]

    # Enumerate shifts in feature space
    shifts_y = tf.range(tf.cast(feature_shape[0], tf.int32)) * tf.cast(feature_stride[0], tf.int32) # [0, 4, ... , 128-4]
    shifts_x = tf.range(tf.cast(feature_shape[1], tf.int32)) * tf.cast(feature_stride[1], tf.int32) # [0, 8, ... , 128-8]

    # [A,B]=Meshgrid(a,b), meshgrid用于从数组a和b产生网格。生成的网格矩阵A和B大小是相同的, 它相当于a从一行重复增加到size(b)行，把b转置成一列再重复增加到size(a)列
    # 如果a，b为2维以上的，则先拍扁，例：
    # a:                b:
    #   [[1,2],           [[5,6],
    #    [3,4]]            [7,8]]
    # A:                             B:
    #   [[1,2,3,4],                     [[5,5,5,5],
    #    [1,2,3,4],                      [6,6,6,6],
    #    [1,2,3,4],                      [7,7,7,7],
    #    [1,2,3,4]]                      [8,8,8,8]]
    shifts_x, shifts_y = tf.cast(shifts_x, tf.float32), tf.cast(shifts_y, tf.float32)
    shifts_x, shifts_y = tf.meshgrid(shifts_x, shifts_y)

    # mesh A: [num_anchor_per_center] B:[feature_shape[0]*feature_shape[1]]=> [feature_shape[0]*feature_shape[1], num_anchor_per_center]
    box_widths, box_centers_x = tf.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = tf.meshgrid(heights, shifts_y)

    box_centers_y = tf.expand_dims(box_centers_y, axis=2)
    box_centers_x = tf.expand_dims(box_centers_x, axis=2)
    box_heights = tf.expand_dims(box_heights, axis=2)
    box_widths = tf.expand_dims(box_widths, axis=2)

    box_centers = tf.reshape(tf.concat([box_centers_y, box_centers_x], axis=-1), (-1, 2))
    box_sizes = tf.reshape(tf.concat([box_heights, box_widths], axis=-1), (-1, 2))

    # Convert to corner coordinates (y1, x1, y2, x2)
    anchor_boxes = tf.concat([box_centers - 0.5 * box_sizes,
                       box_centers + 0.5 * box_sizes], axis=1)
    anchor_boxes = tf.stop_gradient(anchor_boxes)
    return anchor_boxes

def build_target(anchors, gt_boxes, gt_labels, neg_iou_thr, pos_iou_thr, num_rpn_deltas, positive_fraction):
    '''

    :param anchors: [num_anchors, 4] ----> [y_min, x_min, y_max, x_max]
    :param gt_boxes: [-1, num_gt, 4] ----> [y_min, x_min, y_max, x_max]
    :param gt_labels: [-1, num_gt] ----> class的取值范围:0~num_classes-1
    :param neg_iou_thr: float
    :param pos_iou_thr: float
    :param num_rpn_deltas: int 取出样本的个数
    :param positive_fraction：float 正样本所占的比例
    :return:
            target_matchs: shape=[num_anchors] 1=positive, -1=negative, 0=neutral anchor.
            target_deltas: num_rpn_deltas 正样本---->[dy, dx, dh, dw]   负样本---->[0,0,0,0]
    '''
    gt_boxes = tf.reshape(gt_boxes, [-1, 4]) # [num_gt, 4]
    gt_labels = tf.reshape(gt_labels, [-1]) # [num_gt]

    # 初始都为0， 一系列操作后， -1：负样本， 0：不关心的样本， 1：正样本
    target_matchs = tf.zeros(anchors.shape[0], dtype=tf.int32)

    y_min_A, x_min_A, y_max_A, x_max_A = tf.split(anchors, [1, 1, 1, 1], axis=-1)
    anchors_convert = tf.concat([x_min_A, y_min_A, x_max_A, y_max_A], axis=-1)

    y_min_G, x_min_G, y_max_G, x_max_G = tf.split(gt_boxes, [1, 1, 1, 1], axis=-1)
    gt_boxes_convert = tf.concat([x_min_G, y_min_G, x_max_G, y_max_G], axis=-1)

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = mx_compute_overlap(anchors_convert, gt_boxes_convert)

    # Match anchors to GT Boxes
    # If an anchor overlaps ANY GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps ALL GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).

    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. [N_anchors, N_gt_boxes]
    anchor_iou_argmax = tf.argmax(overlaps, axis=1)  # [326396] get clost gt boxes for each anchors
    anchor_iou_max = tf.reduce_max(overlaps, axis=[1])  # [326396] get closet gt boxes's overlap scores
    # if an anchor box overlap all GT box with IoU < 0.3, marked as -1 background
    target_matchs = tf.where(anchor_iou_max < neg_iou_thr,
                             -tf.ones(anchors.shape[0], dtype=tf.int32), target_matchs)

    # 2. Set anchors with high overlap as positive.
    # if an anchor overlap with any GT box with IoU > 0.7, marked as foreground
    target_matchs = tf.where(anchor_iou_max >= pos_iou_thr,
                             tf.ones(anchors.shape[0], dtype=tf.int32), target_matchs)

    # 3. Set an anchor for each GT box (regardless of IoU value).
    # update corresponding value=>1 for GT boxes' closest boxes
    # 解释：将每个gt_box与所以anchor_boxes中iou最大的那个anchor设为正样本，这样就确保每个gt_box都对应一个anchor，即保证至少有num_gt个正样本
    gt_iou_argmax = tf.argmax(overlaps, axis=0)  # [N_gt_boxes]
    target_matchs = tf.compat.v1.scatter_update(tf.Variable(target_matchs), gt_iou_argmax, 1)

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = tf.where(tf.equal(target_matchs, 1))  # [N_pos_anchors, 1], [15, 1]
    ids = tf.squeeze(ids, 1) # [15]
    # 目前正样本的数量 - 应有正样本的数量
    extra = ids.shape.as_list()[0] - int(num_rpn_deltas * positive_fraction)
    # 如果目前正样本的数量 > 应有正样本的数量, 则随机去掉 (目前正样本的数量 - 应有正样本的数量)正样本
    if extra > 0: # extra means the redundant pos_anchors
        # Reset the extra random ones to neutral
        ids = tf.random.shuffle(ids)[:extra]
        target_matchs = tf.compat.v1.scatter_update(target_matchs, ids, 0)

    # Same for negative proposals
    ids = tf.where(tf.equal(target_matchs, -1)) # [213748, 1]
    ids = tf.squeeze(ids, 1)
    # 目前负样本的数量 - 应有负样本的数量
    extra = ids.shape.as_list()[0] - (num_rpn_deltas - tf.reduce_sum(tf.cast(tf.equal(target_matchs, 1), tf.int32)))
    # 如果目前负样本的数量 > 应有负样本的数量, 则随机去掉 (目前负样本的数量 - 应有负样本的数量)正样本
    if extra > 0: # 213507, so many negative anchors!
        # Rest the extra ones to neutral
        ids = tf.random.shuffle(ids)[:extra]
        target_matchs = tf.compat.v1.scatter_update(target_matchs, ids, 0)

    # since we only need 256 anchors, and it had better contains half positive anchors, and harlf neg .
    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = tf.where(tf.equal(target_matchs, 1))  # [15]

    # 取出正样本anchor
    a = tf.gather_nd(anchors, ids)
    # 取出对应的gt
    anchor_idx = tf.gather_nd(anchor_iou_argmax, ids)
    gt = tf.gather(gt_boxes, anchor_idx)
    target_deltas = mx_encode_bbox2delta(a, gt)

    padding = tf.maximum(num_rpn_deltas - tf.shape(target_deltas)[0], 0)
    target_deltas = tf.pad(target_deltas, [(0, padding), (0, 0)])
    return target_matchs, target_deltas


def mx_encode_bbox2delta(anchors, gt_boxes):
    '''

    :param anchors: anchors: [-1, 4] ----> [y_min, x_min, y_max, x_max]
    :param gt_boxes: [-1, 4] ----> [y_min, x_min, y_max, x_max]
    :return:
    '''
    anchors = tf.cast(anchors, tf.float32)
    gt_boxes = tf.cast(gt_boxes, tf.float32)
    height = anchors[:, 2:2+1] - anchors[:, 0:0+1]
    width = anchors[:, 3:3+1] - anchors[:, 1:1+1]
    center_y = anchors[:, 0:0+1] + 0.5 * height
    center_x = anchors[:, 1:1+1] + 0.5 * width

    gt_height = gt_boxes[:, 2:2+1] - gt_boxes[:, 0:0+1]
    gt_width = gt_boxes[:, 3:3+1] - gt_boxes[:, 1:1+1]
    gt_center_y = gt_boxes[:, 0:0+1] + 0.5 * gt_height
    gt_center_x = gt_boxes[:, 1:1+1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = tf.math.log(gt_height / height)
    dw = tf.math.log(gt_width / width)

    delta = tf.concat([dy, dx, dh, dw], axis=-1)
    return delta

def mx_decode_bbox2delta(anchors, pred_boxes):
    '''

    :param anchors: anchors: [-1, 4] ----> [y_min, x_min, y_max, x_max]
    :param pred_boxes: 预测的结果 [-1, 4] ----> [dy, dx, dh, dw]
    :return:
    '''
    anchors = tf.cast(anchors, tf.float32)
    pred_boxes = tf.cast(pred_boxes, tf.float32)

    height = anchors[:, 2:2+1] - anchors[:, 0:0+1]
    width = anchors[:, 3:3+1] - anchors[:, 1:1+1]
    center_y = anchors[:, 0:0+1] + 0.5 * height
    center_x = anchors[:, 1:1+1] + 0.5 * width

    pred_h = tf.exp(pred_boxes[:, 2:2+1]) * height
    pred_w = tf.exp(pred_boxes[:, 3:4+1]) * width
    pred_center_y = pred_boxes[:, 0:0+1] * height + center_y
    pred_center_x = pred_boxes[:, 1:1+1] * width + center_x

    y_min = pred_center_y - 0.5 * pred_h
    x_min = pred_center_x - 0.5 * pred_w
    y_max = pred_center_y + 0.5 * pred_h
    x_max = pred_center_x + 0.5 * pred_w

    result = tf.concat([y_min, x_min, y_max, x_max], axis=-1)
    return result

def bbox_clip(box, window):
    '''
    Args
    ---
        box: [N, (y1, x1, y2, x2)]
        window: [4] in the form y1, x1, y2, x2
    '''
    # Split
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(box, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1)
    clipped.set_shape((clipped.shape[0], 4))
    return clipped
def mx_compute_overlap(anchors, gt_boxes):
    '''
    Introduction: 计算iou
    anchors:                        gt_boxes:
        [[0.0, 0.0, 1.0, 1.0],              [[0.5, 0.5, 1.0, 1.0],
         [0.2, 0.2, 1.5, 1.5],               [2.0, 1.5, 3.0, 4.0]]
         [2.2, 1.7, 2.8, 4.1]
    输出为iou:
            [[0.25       0.        ]
             [0.14792901 0.        ]
             [0.         0.53076917]]
    从上面可以看出：是将“anchors”中的“每行”依次与“gt_boxes”中的“所有行”计算iou，
                  最终的输出iou：其“每行”对应着“anchors”中的“每行”与“bbox_target”中的“所有行”计算出的iou
                  shape=[num_anchors, num_gt_boxes]
    :param anchors: tensor, [-1, 4], 每行为：[x0, y0, x1, y1]
    :param gt_boxex:tensor, [-1, 4], 每行为：[x0, y0, x1, y1]
    :return: shape=[num_anchors, num_gt_boxes]
    '''
    anchors = tf.expand_dims(anchors, 1)
    u_x0 = tf.where(anchors < gt_boxes, anchors, gt_boxes)[:,:,0]
    u_y0 = tf.where(anchors < gt_boxes, anchors, gt_boxes)[:,:,1]
    u_x1 = tf.where(anchors > gt_boxes, anchors, gt_boxes)[:,:,2]
    u_y1 = tf.where(anchors > gt_boxes, anchors, gt_boxes)[:,:,3]

    i_x0 = tf.where(anchors > gt_boxes, anchors, gt_boxes)[:,:,0]
    i_y0 = tf.where(anchors > gt_boxes, anchors, gt_boxes)[:,:,1]
    i_x1 = tf.where(anchors < gt_boxes, anchors, gt_boxes)[:,:,2]
    i_y1 = tf.where(anchors < gt_boxes, anchors, gt_boxes)[:,:,3]

    i_w = i_x1 - i_x0
    i_h = i_y1 - i_y0
    u_w = u_x1 - u_x0
    u_h = u_y1 - u_y0

    i_w = tf.clip_by_value(i_w, 0, 1e10)
    i_h = tf.clip_by_value(i_h, 0, 1e10)
    u_w = tf.clip_by_value(u_w, 0, 1e10)
    u_h = tf.clip_by_value(u_h, 0, 1e10)

    u_arera = u_w * u_h
    i_arera = i_w * i_h
    iou = i_arera / u_arera

    return iou

def smooth_l1_loss(y_true, y_pred):
    '''

    :param y_true: [-1, 4]
    :param y_pred: [-1, 4]
    :return:
    '''
    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), tf.float32)
    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
    return loss
