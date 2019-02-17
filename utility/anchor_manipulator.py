# Copyright 2018 Changan Wang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import math

import tensorflow as tf
import numpy as np

from tensorflow.contrib.image.python.ops import image_ops

def areas(gt_bboxes):
    with tf.name_scope('bboxes_areas', [gt_bboxes]):
        ymin, xmin, ymax, xmax = tf.split(gt_bboxes, 4, axis=1)
        return (xmax - xmin) * (ymax - ymin)

def intersection(gt_bboxes, default_bboxes):
    with tf.name_scope('bboxes_intersection', [gt_bboxes, default_bboxes]):
        # num_anchors x 1
        ymin, xmin, ymax, xmax = tf.split(gt_bboxes, 4, axis=1)
        # 1 x num_anchors
        gt_ymin, gt_xmin, gt_ymax, gt_xmax = [tf.transpose(b, perm=[1, 0]) for b in tf.split(default_bboxes, 4, axis=1)]
        # broadcast here to generate the full matrix
        int_ymin = tf.maximum(ymin, gt_ymin)
        int_xmin = tf.maximum(xmin, gt_xmin)
        int_ymax = tf.minimum(ymax, gt_ymax)
        int_xmax = tf.minimum(xmax, gt_xmax)
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)

        return h * w
def iou_matrix(gt_bboxes, default_bboxes):
    with tf.name_scope('iou_matrix', [gt_bboxes, default_bboxes]):
        # ******************************************************1.计算交集面积
        # (num_boxes*num_anchors)
        inter_vol = intersection(gt_bboxes, default_bboxes)
        # broadcast
        # ******************************************************2.计算并集面积
        # (num_boxes*num_anchors)
        union_vol = areas(gt_bboxes) + tf.transpose(areas(default_bboxes), perm=[1, 0]) - inter_vol
        # 是否有并集给出iou矩阵
        return tf.where(tf.equal(union_vol, 0.0),
                        tf.zeros_like(inter_vol), tf.truediv(inter_vol, union_vol))

def do_dual_max_match(overlap_matrix, low_thres, high_thres, ignore_between=True, gt_max_first=True):
    '''
    overlap_matrix: num_gt * num_anchors
    '''
    with tf.name_scope('dual_max_match', [overlap_matrix]):
        # first match from anchors' side
        # **********************************************1.首先确定每个anchor对符合的一个bbox，因为一个anchor只能匹配一个bbox
        # anchors_to_gt：shape(num_anchors,)
        anchors_to_gt = tf.argmax(overlap_matrix, axis=0)
        # the matching degree
        # **********************************************2.确定该anchor的匹配最大值
        match_values = tf.reduce_max(overlap_matrix, axis=0)

        #positive_mask = tf.greater(match_values, high_thres)
        less_mask = tf.less(match_values, low_thres)
        between_mask = tf.logical_and(tf.less(match_values, high_thres), tf.greater_equal(match_values, low_thres))
        negative_mask = less_mask if ignore_between else between_mask#小于最小阈值的是负样本
        ignore_mask = between_mask if ignore_between else less_mask#处于两个阈值之间的需要忽略
        # fill all negative positions with -1, all ignore positions is -2
        # *********************************************3.标记负样本，以及需要忽略的样本：anchor_to_gt是应该的索引
        match_indices = tf.where(negative_mask, -1 * tf.ones_like(anchors_to_gt), anchors_to_gt)
        match_indices = tf.where(ignore_mask, -2 * tf.ones_like(match_indices), match_indices)

        # negtive values has no effect in tf.one_hot, that means all zeros along that axis
        # so all positive match positions in anchors_to_gt_mask is 1, all others are 0
        anchors_to_gt_mask = tf.one_hot(tf.clip_by_value(match_indices, -1, tf.cast(tf.shape(overlap_matrix)[0], tf.int64)),
                                        tf.shape(overlap_matrix)[0], on_value=1, off_value=0, axis=0, dtype=tf.int32)
        # match from ground truth's side
        # ***********************************************4.bbox侧的匹配情况
        gt_to_anchors = tf.argmax(overlap_matrix, axis=1)

        if gt_max_first:#优先找到一个bbox匹配的，防止bbox找不到匹配的anchors
            # the max match from ground truth's side has higher priority
            left_gt_to_anchors_mask = tf.one_hot(gt_to_anchors, tf.shape(overlap_matrix)[1], on_value=1, off_value=0, axis=1, dtype=tf.int32)
        else:
            # the max match from anchors' side has higher priority
            # use match result from ground truth's side only when the the matching degree from anchors' side is lower than position threshold
            left_gt_to_anchors_mask = tf.cast(tf.logical_and(tf.reduce_max(anchors_to_gt_mask, axis=1, keep_dims=True) < 1,
                                                            tf.one_hot(gt_to_anchors, tf.shape(overlap_matrix)[1],
                                                                        on_value=True, off_value=False, axis=1, dtype=tf.bool)
                                                            ), tf.int64)
        # can not use left_gt_to_anchors_mask here, because there are many ground truthes match to one anchor, we should pick the highest one even when we are merging matching from ground truth side
        left_gt_to_anchors_scores = overlap_matrix * tf.to_float(left_gt_to_anchors_mask)
        # merge matching results from ground truth's side with the original matching results from anchors' side
        # then select all the overlap score of those matching pairs
        selected_scores = tf.gather_nd(overlap_matrix,  tf.stack([tf.where(tf.reduce_max(left_gt_to_anchors_mask, axis=0) > 0,
                                                                            tf.argmax(left_gt_to_anchors_scores, axis=0),
                                                                            anchors_to_gt),
                                                                    tf.range(tf.cast(tf.shape(overlap_matrix)[1], tf.int64))], axis=1))
        # return the matching results for both foreground anchors and background anchors, also with overlap scores
        # **********************************************5.
        return tf.where(tf.reduce_max(left_gt_to_anchors_mask, axis=0) > 0,
                        tf.argmax(left_gt_to_anchors_scores, axis=0),
                        match_indices), selected_scores

# def save_anchors(bboxes, labels, anchors_point):
#     if not hasattr(save_image_with_bbox, "counter"):
#         save_image_with_bbox.counter = 0  # it doesn't exist yet, so initialize it
#     save_image_with_bbox.counter += 1

#     np.save('./debug/bboxes_{}.npy'.format(save_image_with_bbox.counter), np.copy(bboxes))
#     np.save('./debug/labels_{}.npy'.format(save_image_with_bbox.counter), np.copy(labels))
#     np.save('./debug/anchors_{}.npy'.format(save_image_with_bbox.counter), np.copy(anchors_point))
#     return save_image_with_bbox.counter

class AnchorEncoder(object):
    def __init__(self, allowed_borders, positive_threshold, ignore_threshold, prior_scaling, clip=False):
        super(AnchorEncoder, self).__init__()
        self._all_anchors = None
        self._allowed_borders = allowed_borders
        self._positive_threshold = positive_threshold
        self._ignore_threshold = ignore_threshold
        self._prior_scaling = prior_scaling
        self._clip = clip

    def center2point(self, center_y, center_x, height, width):
        return center_y - height / 2., center_x - width / 2., center_y + height / 2., center_x + width / 2.,

    def point2center(self, ymin, xmin, ymax, xmax):
        height, width = (ymax - ymin), (xmax - xmin)
        return ymin + height / 2., xmin + width / 2., height, width

    def encode_all_anchors(self, labels, bboxes, all_anchors, all_num_anchors_depth, all_num_anchors_spatial, debug=False):
        # y, x, h, w are all in range [0, 1] relative to the original image size
        # shape info:
        # y_on_image, x_on_image: layers_shapes[0] * layers_shapes[1]
        # h_on_image, w_on_image: num_anchors
        # *****************************************************1.确保all_num_anchors_depth和all_num_anchors_spatial第1维是相同的
        # all_anchors: ([],[])
        assert (len(all_num_anchors_depth)==len(all_num_anchors_spatial)) and (len(all_num_anchors_depth)==len(all_anchors)), 'inconsist num layers for anchors.'
        with tf.name_scope('encode_all_anchors'):
            num_layers = len(all_num_anchors_depth)
            list_anchors_ymin = []
            list_anchors_xmin = []
            list_anchors_ymax = []
            list_anchors_xmax = []
            tiled_allowed_borders = []
            # *************************************************2.将所有anchor转化为（ymin,xmin,ymax,xmax）
            for ind, anchor in enumerate(all_anchors):
                # **********************************************3.这里注意:anchor[0]是一个矩阵，anchor[2]是1维向量
                # h*w*1  1*num_scale可以broadcast为 h*w*num_scale
                # anchor[0]:y_on_image  anchor[1]:x_on_image
                anchors_ymin_, anchors_xmin_, anchors_ymax_, anchors_xmax_ = self.center2point(anchor[0], anchor[1], anchor[2], anchor[3])
                # print("**********************************打印anchors_ymin_：",anchors_ymin_.graph)
                # anchors_ymin_的形状为[(feature_shape[0]*feature_shape[1]*num_anchor_per_depth),...]
                list_anchors_ymin.append(tf.reshape(anchors_ymin_, [-1]))
                list_anchors_xmin.append(tf.reshape(anchors_xmin_, [-1]))
                list_anchors_ymax.append(tf.reshape(anchors_ymax_, [-1]))
                list_anchors_xmax.append(tf.reshape(anchors_xmax_, [-1]))
                # 表示允许的边框范围 1*anchor_ratios*feature_map size  是每层feature map所有anchors的数量[1]*num_anchor_per_layer
                tiled_allowed_borders.extend([self._allowed_borders[ind]] * all_num_anchors_depth[ind] * all_num_anchors_spatial[ind])
            #一张图片所有anchor concat成一个列表
            anchors_ymin = tf.concat(list_anchors_ymin, 0, name='concat_ymin')
            anchors_xmin = tf.concat(list_anchors_xmin, 0, name='concat_xmin')
            anchors_ymax = tf.concat(list_anchors_ymax, 0, name='concat_ymax')
            anchors_xmax = tf.concat(list_anchors_xmax, 0, name='concat_xmax')

            # ***********************************************4.注意将超越边界的坐标修剪,默认不进行修剪
            if self._clip:
                anchors_ymin = tf.clip_by_value(anchors_ymin, 0., 1.)
                anchors_xmin = tf.clip_by_value(anchors_xmin, 0., 1.)
                anchors_ymax = tf.clip_by_value(anchors_ymax, 0., 1.)
                anchors_xmax = tf.clip_by_value(anchors_xmax, 0., 1.)
            # 边框缓冲区为1（即1倍大小）
            anchor_allowed_borders = tf.stack(tiled_allowed_borders, 0, name='concat_allowed_borders')
            # ***********************************************5.？？？？找到四个边框都在图片内部的边框？？
            inside_mask = tf.logical_and(tf.logical_and(anchors_ymin > -anchor_allowed_borders * 1.,
                                                        anchors_xmin > -anchor_allowed_borders * 1.),
                                        tf.logical_and(anchors_ymax < (1. + anchor_allowed_borders * 1.),
                                                        anchors_xmax < (1. + anchor_allowed_borders * 1.)))

            # ***********************************************6.得到所有坐标的tensor（有可能有负数的，坐标）
            anchors_point = tf.stack([anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax], axis=-1)

            # save_anchors_op = tf.py_func(save_anchors,
            #                 [bboxes,
            #                 labels,
            #                 anchors_point],
            #                 tf.int64, stateful=True)

            # anchors_point：shape:（num_all_anchors,4）

            # with tf.control_dependencies([save_anchors_op]):
            # ***********************************************7.得到bboxes和所有anchor的iou矩阵(num_bboxes,num_anchors)
            overlap_matrix = iou_matrix(bboxes, anchors_point) * tf.cast(tf.expand_dims(inside_mask, 0), tf.float32)
            # **********************************************8.根据threshold，构造正样本和负样本
            # matched_gt:指的是每个anchor匹配的最佳bboxes的索引 range(-2，-1,num_bboxes) gt_scores：指的是匹配的分数
            # -2表示ignore,-1表示负样本，其他表示正样本，并对应有bboxes
            # matched_gt: (num_anchors,)
            matched_gt, gt_scores = do_dual_max_match(overlap_matrix, self._ignore_threshold, self._positive_threshold)
            # get all positive matching positions
            # ***********************************************9.得到正样本的mask
            matched_gt_mask = matched_gt > -1
            # matched_indices shape: (num_anchors,) 保证索引为正
            matched_indices = tf.clip_by_value(matched_gt, 0, tf.int64.max)
            # the labels here maybe chaos at those non-positive positions：负anchors和ingnore可能也会被考虑
            #labels: (num_boxes,)  -> gt_labels: (num_anchors,)
            gt_labels = tf.gather(labels, matched_indices)
            # filter the invalid labels：这里背景类是0，前景类>0,
            gt_labels = gt_labels * tf.cast(matched_gt_mask, tf.int64)
            # set those ignored positions to -1
            # *********************************************10.最终设置，-1表示忽略该anchors忽略，0表示是背景类，其他表示对应的类
            gt_labels = gt_labels + (-1 * tf.cast(matched_gt < -1, tf.int64))
            # *********************************************11.得到每个anchor对应的gt_bbox的坐标
            # bboxes:(num_bboes,4)  matched_indices:(num_anchors,)
            gt_ymin, gt_xmin, gt_ymax, gt_xmax = tf.unstack(tf.gather(bboxes, matched_indices), 4, axis=-1)

            # transform to center / size.
            # *******************************************12.把anchor和gt_boxes的坐标转化为中心坐标，并计算增量
            gt_cy, gt_cx, gt_h, gt_w = self.point2center(gt_ymin, gt_xmin, gt_ymax, gt_xmax)
            anchor_cy, anchor_cx, anchor_h, anchor_w = self.point2center(anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax)
            # print("*****************************打印anchor_h:", anchor_h.graph)
            # encode features.
            # the prior_scaling (in fact is 5 and 10) is use for balance the regression loss of center and with(or height)
            gt_cy = (gt_cy - anchor_cy) / anchor_h / self._prior_scaling[0]
            gt_cx = (gt_cx - anchor_cx) / anchor_w / self._prior_scaling[1]
            gt_h = tf.log(gt_h / anchor_h) / self._prior_scaling[2]
            gt_w = tf.log(gt_w / anchor_w) / self._prior_scaling[3]
            # now gt_localizations is our regression object, but also maybe chaos at those non-positive positions
            # *******************************************13.回归用的target，stack成（num_anchors,4）
            if debug:
                gt_targets = tf.stack([anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax], axis=-1)
            else:
                gt_targets = tf.stack([gt_cy, gt_cx, gt_h, gt_w], axis=-1)

            # set all targets of non-positive positions to 0
            # *******************************************14.这里也将那些负anchors和忽略的anchor设置为0
            gt_targets = tf.expand_dims(tf.cast(matched_gt_mask, tf.float32), -1) * gt_targets
            # *******************************************15.更新_all_anchors：将所有的anchor情况存储起来
            # 需要注意：训练时，decode一个batch可以共用用同一组anchors,验证时只能单独一张图片，不然后面用于decode时不匹配
            self._all_anchors = (anchor_cy, anchor_cx, anchor_h, anchor_w)
            # all_anchors = (anchor_cy, anchor_cx, anchor_h, anchor_w)
            # ******************************************16.返回边框回归用的坐标和分类用的类别
            return gt_targets, gt_labels, gt_scores

    # return a list, of which each is:
    #   shape: [feature_h, feature_w, num_anchors, 4]
    #   order: ymin, xmin, ymax, xmax
    def decode_all_anchors(self, pred_location, num_anchors_per_layer):
        assert self._all_anchors is not None, 'no anchors to decode.'
        with tf.name_scope('decode_all_anchors', [pred_location]):
            # *******************************************1.读取default anchors的坐标
            anchor_cy, anchor_cx, anchor_h, anchor_w = self._all_anchors
            # *******************************************2.将预测的边框偏移转化为中心坐标
            pred_h = tf.exp(pred_location[:, -2] * self._prior_scaling[2]) * anchor_h
            pred_w = tf.exp(pred_location[:, -1] * self._prior_scaling[3]) * anchor_w
            pred_cy = pred_location[:, 0] * self._prior_scaling[0] * anchor_h + anchor_cy
            pred_cx = pred_location[:, 1] * self._prior_scaling[1] * anchor_w + anchor_cx
            # ******************************************3.将中心坐标转化为（ymin,xmin,ymax,xmax）并转化为ymin_list列表
            return tf.split(tf.stack(self.center2point(pred_cy, pred_cx, pred_h, pred_w), axis=-1), num_anchors_per_layer, axis=0)

    def ext_decode_all_anchors(self, pred_location, all_anchors, all_num_anchors_depth, all_num_anchors_spatial):
        assert (len(all_num_anchors_depth)==len(all_num_anchors_spatial)) and (len(all_num_anchors_depth)==len(all_anchors)), 'inconsist num layers for anchors.'
        with tf.name_scope('ext_decode_all_anchors', [pred_location]):
            num_anchors_per_layer = []
            for ind in range(len(all_anchors)):
                num_anchors_per_layer.append(all_num_anchors_depth[ind] * all_num_anchors_spatial[ind])

            num_layers = len(all_num_anchors_depth)
            list_anchors_ymin = []
            list_anchors_xmin = []
            list_anchors_ymax = []
            list_anchors_xmax = []
            tiled_allowed_borders = []
            for ind, anchor in enumerate(all_anchors):
                anchors_ymin_, anchors_xmin_, anchors_ymax_, anchors_xmax_ = self.center2point(anchor[0], anchor[1], anchor[2], anchor[3])

                list_anchors_ymin.append(tf.reshape(anchors_ymin_, [-1]))
                list_anchors_xmin.append(tf.reshape(anchors_xmin_, [-1]))
                list_anchors_ymax.append(tf.reshape(anchors_ymax_, [-1]))
                list_anchors_xmax.append(tf.reshape(anchors_xmax_, [-1]))

            anchors_ymin = tf.concat(list_anchors_ymin, 0, name='concat_ymin')
            anchors_xmin = tf.concat(list_anchors_xmin, 0, name='concat_xmin')
            anchors_ymax = tf.concat(list_anchors_ymax, 0, name='concat_ymax')
            anchors_xmax = tf.concat(list_anchors_xmax, 0, name='concat_xmax')

            anchor_cy, anchor_cx, anchor_h, anchor_w = self.point2center(anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax)

            pred_h = tf.exp(pred_location[:,-2] * self._prior_scaling[2]) * anchor_h
            pred_w = tf.exp(pred_location[:, -1] * self._prior_scaling[3]) * anchor_w
            pred_cy = pred_location[:, 0] * self._prior_scaling[0] * anchor_h + anchor_cy
            pred_cx = pred_location[:, 1] * self._prior_scaling[1] * anchor_w + anchor_cx

            return tf.split(tf.stack(self.center2point(pred_cy, pred_cx, pred_h, pred_w), axis=-1), num_anchors_per_layer, axis=0)

class AnchorCreator(object):
    def __init__(self, img_shape, layers_shapes, anchor_scales, extra_anchor_scales, anchor_ratios, layer_steps):
        super(AnchorCreator, self).__init__()
        # img_shape -> (height, width)
        self._img_shape = img_shape
        self._layers_shapes = layers_shapes
        self._anchor_scales = anchor_scales
        self._extra_anchor_scales = extra_anchor_scales
        self._anchor_ratios = anchor_ratios
        self._layer_steps = layer_steps
        self._anchor_offset = [0.5] * len(self._layers_shapes)

    def get_layer_anchors(self, layer_shape, anchor_scale, extra_anchor_scale, anchor_ratio, layer_step, offset = 0.5):
        ''' assume layer_shape[0] = 6, layer_shape[1] = 5
        x_on_layer = [[0, 1, 2, 3, 4],
                       [0, 1, 2, 3, 4],
                       [0, 1, 2, 3, 4],
                       [0, 1, 2, 3, 4],
                       [0, 1, 2, 3, 4],
                       [0, 1, 2, 3, 4]]
        y_on_layer = [[0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1],
                       [2, 2, 2, 2, 2],
                       [3, 3, 3, 3, 3],
                       [4, 4, 4, 4, 4],
                       [5, 5, 5, 5, 5]]
        '''
        with tf.name_scope('get_layer_anchors'):
            # *************************************************1.获取每层feature_map的x,y坐标，
            x_on_layer, y_on_layer = tf.meshgrid(tf.range(layer_shape[1]), tf.range(layer_shape[0]))

            # *************************************************2.对anchors坐标进行归一化
            y_on_image = (tf.cast(y_on_layer, tf.float32) + offset) * layer_step / self._img_shape[0]
            x_on_image = (tf.cast(x_on_layer, tf.float32) + offset) * layer_step / self._img_shape[1]

            # *************************************************3.统计anchor深度方向的数量
            num_anchors_along_depth = len(anchor_scale) * len(anchor_ratio) + len(extra_anchor_scale)
            num_anchors_along_spatial = layer_shape[1] * layer_shape[0]

            list_h_on_image = []
            list_w_on_image = []

            global_index = 0
            # for square anchors
            # ************************************************4.anchor scale就是每层feature map anchor的基础单位
            # 1:1大小正方形的anchors
            for _, scale in enumerate(extra_anchor_scale):
                list_h_on_image.append(scale)
                list_w_on_image.append(scale)
                global_index += 1
            # for other aspect ratio anchors
            # 不同比例的anchors
            for scale_index, scale in enumerate(anchor_scale):
                for ratio_index, ratio in enumerate(anchor_ratio):
                    list_h_on_image.append(scale / math.sqrt(ratio))
                    list_w_on_image.append(scale * math.sqrt(ratio))
                    global_index += 1
            # shape info:
            # y_on_image, x_on_image: layers_shapes[0] * layers_shapes[1]
            # h_on_image, w_on_image: num_anchors_along_depth
            # ******************************************************3.扩展坐标的维度？
            return tf.expand_dims(y_on_image, axis=-1), tf.expand_dims(x_on_image, axis=-1), \
                    tf.constant(list_h_on_image, dtype=tf.float32), \
                    tf.constant(list_w_on_image, dtype=tf.float32), num_anchors_along_depth, num_anchors_along_spatial

    def get_all_anchors(self):
        all_anchors = []
        all_num_anchors_depth = []
        all_num_anchors_spatial = []
        for layer_index, layer_shape in enumerate(self._layers_shapes):
            # *****************************************1.逐feature_map生成anchor,得到的坐标是（x,y,w,h）
            anchors_this_layer = self.get_layer_anchors(layer_shape,
                                                        self._anchor_scales[layer_index],
                                                        self._extra_anchor_scales[layer_index],
                                                        self._anchor_ratios[layer_index],
                                                        self._layer_steps[layer_index],
                                                        self._anchor_offset[layer_index])
            # print("************************打印anchors_this_layer：",anchors_this_layer[0].graph)
            all_anchors.append(anchors_this_layer[:-2])
            all_num_anchors_depth.append(anchors_this_layer[-2])
            all_num_anchors_spatial.append(anchors_this_layer[-1])
        # *********************************************2.返回所有anchors，每层feature_map的anchor都是一个4维张量
        return all_anchors, all_num_anchors_depth, all_num_anchors_spatial

