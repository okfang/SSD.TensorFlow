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
    # gt_bboxes ->shape: [num_bboxes, 4]
    # default_bboxes ->shape: [num_all_anchors, 4]
    with tf.name_scope('bboxes_intersection', [gt_bboxes, default_bboxes]):
        # ymin -> shape: [num_bboxes, 1]
        ymin, xmin, ymax, xmax = tf.split(gt_bboxes, 4, axis=1)
        # gt_ymin ->shape: [1, num_all_anchors]
        gt_ymin, gt_xmin, gt_ymax, gt_xmax = [tf.transpose(b, perm=[1, 0]) for b in tf.split(default_bboxes, 4, axis=1)]
        # int_ymin ->shape: [num_bboxes, num_all_anchors]
        int_ymin = tf.maximum(ymin, gt_ymin)
        int_xmin = tf.maximum(xmin, gt_xmin)
        int_ymax = tf.minimum(ymax, gt_ymax)
        int_xmax = tf.minimum(xmax, gt_xmax)
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # h*w ->shape: [num_bboxes, num_all_anchors]  intersection of two boxes
        return h * w

def iou_matrix(gt_bboxes, default_bboxes):
    # gt_bboxes ->shape: [num_bboxes, 4]
    # default_bboxes ->shape: [num_all_anchors, 4]
    with tf.name_scope('iou_matrix', [gt_bboxes, default_bboxes]):
        # inter_vol ->shape: [num_bboxes, num_all_anchors]
        inter_vol = intersection(gt_bboxes, default_bboxes)
        # areas(gt_bboxes)  ->shape: [num_bboxes,1]
        # tf.transpose(areas(default_bboxes), perm=[1, 0])  -> shape:[1,num_all_anchors]
        # union_vol ->shape: [num_bboxes, num_all_anchors]
        union_vol = areas(gt_bboxes) + tf.transpose(areas(default_bboxes), perm=[1, 0]) - inter_vol

        # union_vol cannot be 0, only if all coordinate are 0 or coordinate are not valid
        # inter_vol can be 0, and this time iou=tf.truediv(inter_vol, union_vol) will be 0
        return tf.where(tf.equal(union_vol, 0.0),
                        tf.zeros_like(inter_vol), tf.truediv(inter_vol, union_vol))

def do_dual_max_match(overlap_matrix, low_thres, high_thres, ignore_between=True, gt_max_first=True):
    '''
    overlap_matrix: num_gt * num_anchors
    '''
    with tf.name_scope('dual_max_match', [overlap_matrix]):
        # find best matched gtbox for each anchors
        # anchor_to_gt -> shape: [num_all_anchors,]
        anchors_to_gt = tf.argmax(overlap_matrix, axis=0)
        # match_values -> shape: [num_all_anchors,]    max score
        match_values = tf.reduce_max(overlap_matrix, axis=0)

        less_mask = tf.less(match_values, low_thres)
        between_mask = tf.logical_and(tf.less(match_values, high_thres), tf.greater_equal(match_values, low_thres))
        # negative example (iou<l ow_thres mean hard example)
        negative_mask = less_mask if ignore_between else between_mask
        ignore_mask = between_mask if ignore_between else less_mask

        # negative: -1
        match_indices = tf.where(negative_mask, -1 * tf.ones_like(anchors_to_gt), anchors_to_gt)
        # ignore: -2
        match_indices = tf.where(ignore_mask, -2 * tf.ones_like(match_indices), match_indices)

        # anchors_to_gt_mask ->shape: [num_all_anchors, num_bboxes ]    -2 be changed to -1
        anchors_to_gt_mask = tf.one_hot(tf.clip_by_value(match_indices, -1, tf.cast(tf.shape(overlap_matrix)[0], tf.int64)),
                                        tf.shape(overlap_matrix)[0], on_value=1, off_value=0, axis=0, dtype=tf.int32)
        # best matched anchor for each gt_box
        # gt_to_anchors ->shape: [num_bboxes, ]
        gt_to_anchors = tf.argmax(overlap_matrix, axis=1)
        if gt_max_first:
            # left_gt_to_anchors_mask -> shape: [num_bboxes, num_all_anchors]
            left_gt_to_anchors_mask = tf.one_hot(gt_to_anchors, tf.shape(overlap_matrix)[1], on_value=1, off_value=0, axis=1, dtype=tf.int32)
        else:
            left_gt_to_anchors_mask = tf.cast(tf.logical_and(tf.reduce_max(anchors_to_gt_mask, axis=1, keep_dims=True) < 1,
                                                            tf.one_hot(gt_to_anchors, tf.shape(overlap_matrix)[1],
                                                                        on_value=True, off_value=False, axis=1, dtype=tf.bool)
                                                            ), tf.int64)
        # select best matched anchor iou_score for each gt_box
        left_gt_to_anchors_scores = overlap_matrix * tf.to_float(left_gt_to_anchors_mask)

        # if an anchor match multi gt_boxes, anchor will choose the best match gt_box, and this make sure gt_box always has one matched anchors
        # else the anchor will be set -1(negative) or -2(ignore)
        # matched_gt ->shape: [num_all_anchors,]
        matched_gt_indices = tf.where(tf.reduce_max(left_gt_to_anchors_mask, axis=0) > 0,
                              tf.argmax(left_gt_to_anchors_scores, axis=0),
                              match_indices)

        # matched_index ->shape:[num_all_anchors,]
        matched_index = tf.where(tf.reduce_max(left_gt_to_anchors_mask, axis=0) > 0,
                                 tf.argmax(left_gt_to_anchors_scores, axis=0),
                                 anchors_to_gt)
        # selected_index -> shape: [num_all_anchors,2]
        selected_index = tf.stack([matched_index, tf.range(tf.cast(tf.shape(overlap_matrix)[1], tf.int64))], axis=1)

        # selected_scores -> shape:[num_all_anchors,]
        matched_iou_scores = tf.gather_nd(overlap_matrix,selected_index)

        return matched_gt_indices, matched_iou_scores

def center2point(center_y, center_x, height, width):
    return center_y - height / 2., center_x - width / 2., center_y + height / 2., center_x + width / 2.,

def point2center(ymin, xmin, ymax, xmax):
    height, width = (ymax - ymin), (xmax - xmin)
    return ymin + height / 2., xmin + width / 2., height, width

class AnchorEncoder(object):
    def __init__(self, allowed_borders, positive_threshold, neg_threshold, prior_scaling, clip=False):
        super(AnchorEncoder, self).__init__()
        self._all_anchors = None
        self._allowed_borders = allowed_borders
        self._positive_threshold = positive_threshold
        self._neg_threshold = neg_threshold
        self._prior_scaling = prior_scaling
        self._clip = clip


    def encode_all_anchors(self, labels=None, bboxes=None, all_anchors=None, all_num_anchors_depth=None, all_num_anchors_spatial=None,debug=False):
        """
        :param
            encode all anchors for one image
            labels  = [num_bboxes,]
            bboxes  = [num_bboxes,4]
            all_anchors =  [num_all_anchors,4]
        :return:
            gt_targets ->shape: [num_all_anchors,4]
            gt_labels ->shape: [num_all_anchors,]   labels:  -1:ignore 0:negative >0:positive
            matched_iou_scores ->shape: [num_all_anchors,]
        """
        # y, x, h, w are all in range [0, 1] relative to the original image size
        with tf.name_scope('encode_all_anchors'):
            anchor_cy, anchor_cx, anchor_h, anchor_w = all_anchors
            # anchors_ymin  -> shape:（num_all_anchors）
            anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax = center2point(anchor_cy, anchor_cx, anchor_h, anchor_w)

            # inside_mask to filter default anchors that out size the original image with 'tiled_allowed_borders' constraint
            tiled_allowed_borders = []
            for ind,_ in enumerate(self._allowed_borders):
                tiled_allowed_borders.extend(
                    [self._allowed_borders[ind]] * all_num_anchors_depth[ind] * all_num_anchors_spatial[ind])
            anchor_allowed_borders = tf.stack(tiled_allowed_borders, 0, name='concat_allowed_borders')
            inside_mask = tf.logical_and(tf.logical_and(anchors_ymin > -anchor_allowed_borders * 1.,
                                                        anchors_xmin > -anchor_allowed_borders * 1.),
                                         tf.logical_and(anchors_ymax < (1. + anchor_allowed_borders * 1.),
                                                        anchors_xmax < (1. + anchor_allowed_borders * 1.)))

            # anchors_point  ->shape: [num_all_anchors, 4]
            anchors_point = tf.stack([anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax], axis=-1)
            overlap_matrix = iou_matrix(bboxes, anchors_point) * tf.cast(tf.expand_dims(inside_mask, 0), tf.float32)
            # matched_gt_indices ->shape:[num_all_anchors,]  best matched bbox indices of each anchors
            # matched_iou_scores ->shape:[num_all_anchors,]  best matched bbox iou_scores of each anchor
            matched_gt_indices, matched_iou_scores = do_dual_max_match(overlap_matrix, self._neg_threshold, self._positive_threshold)

            matched_gt_mask = matched_gt_indices > -1
            matched_indices = tf.clip_by_value(matched_gt_indices, 0, tf.int64.max)
            # gt_labels -> shape: [num_all_anchors,]
            gt_labels = tf.gather(labels, matched_indices)
            # set none positive anhors to labels 0
            gt_labels = gt_labels * tf.cast(matched_gt_mask, tf.int64)
            # set ignore anchors to labels -1
            gt_labels = gt_labels + (-1 * tf.cast(matched_gt_indices < -1, tf.int64))

            # gt_labels -> shape: [num_all_anchors,4]
            gt_boxes = tf.gather(bboxes, matched_indices)
            # gt_ymin ->shape: [num_all_anchors,]
            gt_ymin, gt_xmin, gt_ymax, gt_xmax = tf.unstack(gt_boxes, 4, axis=-1)
            gt_cy, gt_cx, gt_h, gt_w = point2center(gt_ymin, gt_xmin, gt_ymax, gt_xmax)
            anchor_cy, anchor_cx, anchor_h, anchor_w = point2center(anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax)

            # calculate offset
            gt_cy = (gt_cy - anchor_cy) / anchor_h / self._prior_scaling[0]
            gt_cx = (gt_cx - anchor_cx) / anchor_w / self._prior_scaling[1]
            gt_h = tf.log(gt_h / anchor_h) / self._prior_scaling[2]
            gt_w = tf.log(gt_w / anchor_w) / self._prior_scaling[3]

            gt_targets = tf.stack([gt_cy, gt_cx, gt_h, gt_w], axis=-1)
            # only keep positive anchors
            gt_targets = tf.expand_dims(tf.cast(matched_gt_mask, tf.float32), -1) * gt_targets

            return gt_targets, gt_labels, matched_iou_scores

    def decode_all_anchors(self, pred_location, all_anchors=None,num_anchors_per_layer=None):
        with tf.name_scope('decode_all_anchors', [pred_location]):
            # anchor_cy ->shape: [num_all_anchors,]
            anchor_cy, anchor_cx, anchor_h, anchor_w = all_anchors
            
            # change predict offset to coordinate(loc_loss is too small ,_prior_scaling for scale loss)
            pred_h = tf.exp(pred_location[:, -2] * self._prior_scaling[2]) * anchor_h
            pred_w = tf.exp(pred_location[:, -1] * self._prior_scaling[3]) * anchor_w
            pred_cy = pred_location[:, 0] * self._prior_scaling[0] * anchor_h + anchor_cy
            pred_cx = pred_location[:, 1] * self._prior_scaling[1] * anchor_w + anchor_cx
            # pred_point -> shpep: [num_all_anchors,4]
            bboxes_pred_per_image = tf.stack(center2point(pred_cy, pred_cx, pred_h, pred_w),axis=-1)
            # list [[num_anchors_per_layer,4],[][][][][]]
            # pred_point_per_layer_list = tf.split(bboxes_pred_per_image, num_anchors_per_layer, axis=0)
            return bboxes_pred_per_image

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
                anchors_ymin_, anchors_xmin_, anchors_ymax_, anchors_xmax_ = center2point(anchor[0], anchor[1], anchor[2], anchor[3])

                list_anchors_ymin.append(tf.reshape(anchors_ymin_, [-1]))
                list_anchors_xmin.append(tf.reshape(anchors_xmin_, [-1]))
                list_anchors_ymax.append(tf.reshape(anchors_ymax_, [-1]))
                list_anchors_xmax.append(tf.reshape(anchors_xmax_, [-1]))

            anchors_ymin = tf.concat(list_anchors_ymin, 0, name='concat_ymin')
            anchors_xmin = tf.concat(list_anchors_xmin, 0, name='concat_xmin')
            anchors_ymax = tf.concat(list_anchors_ymax, 0, name='concat_ymax')
            anchors_xmax = tf.concat(list_anchors_xmax, 0, name='concat_xmax')

            anchor_cy, anchor_cx, anchor_h, anchor_w = point2center(anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax)

            pred_h = tf.exp(pred_location[:,-2] * self._prior_scaling[2]) * anchor_h
            pred_w = tf.exp(pred_location[:, -1] * self._prior_scaling[3]) * anchor_w
            pred_cy = pred_location[:, 0] * self._prior_scaling[0] * anchor_h + anchor_cy
            pred_cx = pred_location[:, 1] * self._prior_scaling[1] * anchor_w + anchor_cx

            return tf.split(tf.stack(center2point(pred_cy, pred_cx, pred_h, pred_w), axis=-1), num_anchors_per_layer, axis=0)

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
        self._clip=False

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
            x_on_layer, y_on_layer = tf.meshgrid(tf.range(layer_shape[1]), tf.range(layer_shape[0]))

            # normalize to (0,1)
            y_on_image = (tf.cast(y_on_layer, tf.float32) + offset) * layer_step / self._img_shape[0]
            x_on_image = (tf.cast(x_on_layer, tf.float32) + offset) * layer_step / self._img_shape[1]

            num_anchors_along_depth = len(anchor_scale) * len(anchor_ratio) + len(extra_anchor_scale)
            num_anchors_along_spatial = layer_shape[1] * layer_shape[0]

            list_h_on_image = []
            list_w_on_image = []

            # for square anchors
            for _, scale in enumerate(extra_anchor_scale):
                list_h_on_image.append(scale)
                list_w_on_image.append(scale)
            # for other aspect ratio anchors
            for scale_index, scale in enumerate(anchor_scale):
                for ratio_index, ratio in enumerate(anchor_ratio):
                    list_h_on_image.append(scale / math.sqrt(ratio))
                    list_w_on_image.append(scale * math.sqrt(ratio))

            return y_on_image, x_on_image, tf.constant(list_h_on_image, dtype=tf.float32), \
                    tf.constant(list_w_on_image, dtype=tf.float32), num_anchors_along_depth, num_anchors_along_spatial

    def get_all_anchors(self):

        all_anchors = []
        all_num_anchors_depth = []
        all_num_anchors_spatial = []
        for layer_index, layer_shape in enumerate(self._layers_shapes):
            # y_on_image:[feature_map_size,feature_map_size]
            # list_h_on_image: [num_scale_ratio]
            y_on_image,x_on_image,list_h_on_image,list_w_on_image,num_anchors_along_depth,num_anchors_along_spatial= self.get_layer_anchors(layer_shape,
                                                        self._anchor_scales[layer_index],
                                                        self._extra_anchor_scales[layer_index],
                                                        self._anchor_ratios[layer_index],
                                                        self._layer_steps[layer_index],
                                                        self._anchor_offset[layer_index])
            all_anchors.append([y_on_image,x_on_image,list_h_on_image,list_w_on_image])
            all_num_anchors_depth.append(num_anchors_along_depth)
            all_num_anchors_spatial.append(num_anchors_along_spatial)

        with tf.name_scope('create_all_anchors'):
            num_layers = len(all_num_anchors_depth)
            list_anchors_ymin = []
            list_anchors_xmin = []
            list_anchors_ymax = []
            list_anchors_xmax = []
            tiled_allowed_borders = []
            for y_on_image,x_on_image,list_h_on_image,list_w_on_image in all_anchors:
                # expand dims for broadcast [feature_map_size, feature_map_size,1]
                y_on_image, x_on_image = tf.expand_dims(y_on_image,axis=-1), tf.expand_dims(x_on_image,axis=-1)

                # anchors_ymin_ -> shape: [feature_map_size, feature_map_size, num_anchors_per_position]
                anchors_ymin_, anchors_xmin_, anchors_ymax_, anchors_xmax_ = center2point(y_on_image,x_on_image,list_h_on_image,list_w_on_image)
                list_anchors_ymin.append(tf.reshape(anchors_ymin_, [-1]))
                list_anchors_xmin.append(tf.reshape(anchors_xmin_, [-1]))
                list_anchors_ymax.append(tf.reshape(anchors_ymax_, [-1]))
                list_anchors_xmax.append(tf.reshape(anchors_xmax_, [-1]))

            # anchors_ymin  -> shape:[num_all_anchors,]
            anchors_ymin = tf.concat(list_anchors_ymin, 0, name='concat_ymin')
            anchors_xmin = tf.concat(list_anchors_xmin, 0, name='concat_xmin')
            anchors_ymax = tf.concat(list_anchors_ymax, 0, name='concat_ymax')
            anchors_xmax = tf.concat(list_anchors_xmax, 0, name='concat_xmax')
            if self._clip:
                anchors_ymin = tf.clip_by_value(anchors_ymin, 0., 1.)
                anchors_xmin = tf.clip_by_value(anchors_xmin, 0., 1.)
                anchors_ymax = tf.clip_by_value(anchors_ymax, 0., 1.)
                anchors_xmax = tf.clip_by_value(anchors_xmax, 0., 1.)
            anchor_cy, anchor_cx, anchor_h, anchor_w = point2center(anchors_ymin, anchors_xmin, anchors_ymax,
                                                                    anchors_xmax)
            all_anchors = (anchor_cy, anchor_cx, anchor_h, anchor_w)
        return all_anchors,all_num_anchors_depth, all_num_anchors_spatial

