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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# 添加路径
# import sys
# from os.path import abspath, join, dirname
# sys.path.insert(0,abspath(dirname(__file__)))

import tensorflow as tf

from dataset.dataset_util import get_dataset
from net import ssd_net

from dataset import dataset_common
from preprocessing import ssd_preprocessing
from utility import anchor_manipulator
from utility import scaffolds

# hardware related configuration
tf.app.flags.DEFINE_integer(
    'num_readers', 8,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 24,
    'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer(
    'num_cpu_threads', 0,
    'The number of cpu cores used to train.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 1., 'GPU memory fraction to use.')
# scaffold related configuration
tf.app.flags.DEFINE_string(
    'data_dir', '/home/dxfang/dataset/tfrecords/pascal_voc/',
    'The directory where the dataset input data is stored.')
tf.app.flags.DEFINE_integer(
    'num_classes', 21, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
    'model_dir', './logs/',
    'The directory where the model will be stored.')
tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are printed.')
tf.app.flags.DEFINE_integer(
    'save_summary_steps', 100,
    'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer(
    'save_checkpoints_secs', 7200,
    'The frequency with which the model is saved, in seconds.')
# model related configuration
tf.app.flags.DEFINE_integer(
    'train_image_size', 300,
    'The size of the input image for the model to use.')
tf.app.flags.DEFINE_integer(
    'train_epochs', None,
    'The number of epochs to use for training.')
tf.app.flags.DEFINE_integer(
    'max_number_of_steps', 120000,
    'The max number of steps to use for training.')
tf.app.flags.DEFINE_integer(
    'batch_size', 32,
    'Batch size for training and evaluation.')
tf.app.flags.DEFINE_string(
    'data_format', 'channels_first',  # 'channels_first' or 'channels_last'
    'A flag to override the data format used in the model. channels_first '
    'provides a performance boost on GPU but is not always compatible '
    'with CPU. If left unspecified, the data format will be chosen '
    'automatically based on whether TensorFlow was built for CPU or GPU.')
tf.app.flags.DEFINE_float(
    'negative_ratio', 3., 'Negative ratio in the loss function.')
tf.app.flags.DEFINE_float(
    'match_threshold', 0.5, 'Matching threshold in the loss function.')
tf.app.flags.DEFINE_float(
    'neg_threshold', 0.5, 'Matching threshold for the negtive examples in the loss function.')
# optimizer related configuration
tf.app.flags.DEFINE_integer(
    'tf_random_seed', 20180503, 'Random seed for TensorFlow initializers.')
tf.app.flags.DEFINE_float(
    'weight_decay', 5e-4, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.000001,
    'The minimal end learning rate used by a polynomial decay learning rate.')
# for learning rate piecewise_constant decay
tf.app.flags.DEFINE_string(
    'decay_boundaries', '500, 80000, 100000',
    'Learning rate decay boundaries by global_step (comma-separated list).')
tf.app.flags.DEFINE_string(
    'lr_decay_factors', '0.1, 1, 0.1, 0.01',
    'The values of learning_rate decay factor for each segment between boundaries (comma-separated list).')
# checkpoint related configuration
tf.app.flags.DEFINE_string(
    'checkpoint_path', './model',
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'checkpoint_model_scope', 'vgg_16',
    'Model scope in the checkpoint. None if the same as the trained model.')
tf.app.flags.DEFINE_string(
    'model_scope', 'ssd300',
    'Model scope name used to replace the name_scope in checkpoint.')
tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', 'ssd300/multibox_head, ssd300/additional_layers, ssd300/conv4_3_scale',
    'Comma-separated list of scopes of variables to exclude when restoring from a checkpoint.')
tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', True,
    'When restoring a checkpoint would ignore missing variables.')
tf.app.flags.DEFINE_boolean(
    'multi_gpu', True,
    'Whether there is GPU to use for training.')

FLAGS = tf.app.flags.FLAGS


# CUDA_VISIBLE_DEVICES
def validate_batch_size_for_multi_gpu(batch_size):
    """For multi-gpu, batch-size must be a multiple of the number of
    available GPUs.

    Note that this should eventually be handled by replicate_model_fn
    directly. Multi-GPU support is currently experimental, however,
    so doing the work here until that feature is in place.
    """
    if FLAGS.multi_gpu:
        from tensorflow.python.client import device_lib

        local_device_protos = device_lib.list_local_devices()
        num_gpus = sum([1 for d in local_device_protos if d.device_type == 'GPU'])
        if not num_gpus:
            raise ValueError('Multi-GPU mode was specified, but no GPUs '
                             'were found. To use CPU, run --multi_gpu=False.')

        remainder = batch_size % num_gpus
        if remainder:
            err = ('When running with multiple GPUs, batch size '
                   'must be a multiple of the number of available GPUs. '
                   'Found {} GPUs with a batch size of {}; try --batch_size={} instead.'
                   ).format(num_gpus, batch_size, batch_size - remainder)
            raise ValueError(err)
        return num_gpus
    return 0


def get_init_fn():
    return scaffolds.get_init_fn_for_scaffold(FLAGS.model_dir, FLAGS.checkpoint_path,
                                              FLAGS.model_scope, FLAGS.checkpoint_model_scope,
                                              FLAGS.checkpoint_exclude_scopes, FLAGS.ignore_missing_vars,
                                              name_remap={'/kernel': '/weights', '/bias': '/biases'})


# couldn't find better way to pass params from input_fn to model_fn
# some tensors used by model_fn must be created in input_fn to ensure they are in the same graph
# but when we put these tensors to labels's dict, the replicate_model_fn will split them into each GPU
# the problem is that they shouldn't be splited

# global_anchor_info = dict()


def input_pipeline(file_pattern='train-*', is_training=True, batch_size=FLAGS.batch_size):
    def input_fn(params=None):
        # ****************************************************1.规定训练时图片尺寸的要求（300*300）
        out_shape = [FLAGS.train_image_size] * 2

        # *************************************************6.定义模型的预处理方法：可以根据是否训练进行不同的处理
        image_preprocessing_fn = lambda image_, labels_, bboxes_: ssd_preprocessing.preprocess_image(image_, labels_,
                                                                                                     bboxes_, out_shape,
                                                                                                     is_training=is_training,
                                                                                                     data_format=FLAGS.data_format,
                                                                                                     output_rgb=False)
        # 构建数据的输入流：可以根据训练过程，选择数据源
        # 重构：使用原生的tf api
        # image, _, shape, loc_targets, cls_targets, match_scores = dataset_common.slim_get_batch(FLAGS.num_classes,
        #                                                                                                 batch_size,
        #                                                                                                 (
        #                                                                                                     'train' if is_training else 'val'),
        #                                                                                                 os.path.join(
        #                                                                                                     FLAGS.data_dir,
        #                                                                                                     dataset_pattern),
        #                                                                                                 FLAGS.num_readers,
        #                                                                                                 FLAGS.num_preprocessing_threads,
        #                                                                                                 image_preprocessing_fn,
        #                                                                                                 anchor_encoder_fn,
        #                                                                                                 num_epochs=FLAGS.train_epochs,
        #                                                                                                 is_training=is_training)

        # **************************************************10.为model_fn构建features和labels
        # 为了使用MirroredStrategy.需要使用dataset
        dataset = get_dataset(file_pattern=file_pattern, is_training=True, batch_size=batch_size,
                              image_preprocessing_fn=image_preprocessing_fn)
        return dataset

    return input_fn


def modified_smooth_l1(bbox_pred, bbox_targets, bbox_inside_weights=1., bbox_outside_weights=1., sigma=1.):
    """
        ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
        SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                      |x| - 0.5 / sigma^2,    otherwise
    """
    with tf.name_scope('smooth_l1', [bbox_pred, bbox_targets]):
        sigma2 = sigma * sigma

        inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))

        smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
        smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
        smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
        smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                                  tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

        outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)

        return outside_mul


# from scipy.misc import imread, imsave, imshow, imresize
# import numpy as np
# from utility import draw_toolbox

# def save_image_with_bbox(image, labels_, scores_, bboxes_):
#     if not hasattr(save_image_with_bbox, "counter"):
#         save_image_with_bbox.counter = 0  # it doesn't exist yet, so initialize it
#     save_image_with_bbox.counter += 1

#     img_to_draw = np.copy(image)

#     img_to_draw = draw_toolbox.bboxes_draw_on_img(img_to_draw, labels_, scores_, bboxes_, thickness=2)
#     imsave(os.path.join('./debug/{}.jpg').format(save_image_with_bbox.counter), img_to_draw)
#     return save_image_with_bbox.counter

def ssd_model_fn(features, labels, mode, params):
    """model_fn for SSD to be used with our Estimator."""
    # glabels: label列表
    # gbboxes: boxes列表
    shape = labels['shape']
    glabels = labels['glabels']
    gbboxes = labels['gbboxes']

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~打印glabels：",glabels)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~打印gbboxes：",gbboxes)
    print(gbboxes)

    # -------------------------------------进行数据处理，生成训练样本---------------------------------

    # *********************1.规定训练时图片尺寸的要求（300*300）
    out_shape = [FLAGS.train_image_size] * 2

    # ********************2.定义anchor生成器，用于生成一系列default anchors
    anchor_creator = anchor_manipulator.AnchorCreator(out_shape,
                                                      layers_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3),
                                                                     (1, 1)],
                                                      anchor_scales=[(0.1,), (0.2,), (0.375,), (0.55,), (0.725,),
                                                                     (0.9,)],
                                                      extra_anchor_scales=[(0.1414,), (0.2739,), (0.4541,),
                                                                           (0.6315,), (0.8078,), (0.9836,)],
                                                      anchor_ratios=[(1., 2., .5), (1., 2., 3., .5, 0.3333),
                                                                     (1., 2., 3., .5, 0.3333),
                                                                     (1., 2., 3., .5, 0.3333), (1., 2., .5),
                                                                     (1., 2., .5)],
                                                      layer_steps=[8, 16, 32, 64, 100, 300])

    # *******************3.使用anchor生成器获取所欲的anchors（x,y,h,w）
    # all_anchors格式：[(y_on_image,x_on_image),....] 大小：num_layer, y_on_image(feature_shape[0],feature_shape[1],1)
    all_anchors, all_num_anchors_depth, all_num_anchors_spatial = anchor_creator.get_all_anchors()

    # ******************4.计算每层的anchor数量
    num_anchors_per_layer = []
    for ind in range(len(all_anchors)):
        num_anchors_per_layer.append(
            all_num_anchors_depth[ind] * all_num_anchors_spatial[ind])  # feature map大小以及每个位置的anchor数量

    # ****************5.构建anchor处理类：主要完成anchor坐标的转化
    anchor_encoder_decoder = anchor_manipulator.AnchorEncoder(allowed_borders=[1.0] * 6,
                                                              positive_threshold=FLAGS.match_threshold,
                                                              ignore_threshold=FLAGS.neg_threshold,
                                                              prior_scaling=[0.1, 0.1, 0.2, 0.2])

    # ****************7.将anchor和gtbox编码成（ymin,xmin,ymax,xmax）的格式
    anchor_encoder_fn = lambda glabels_, gbboxes_: anchor_encoder_decoder.encode_all_anchors(glabels_, gbboxes_,
                                                                                             all_anchors,
                                                                                             all_num_anchors_depth,
                                                                                             all_num_anchors_spatial)

    # ***************8.获取解码函数，将预测结果解码为边框
    decode_fn = lambda pred: anchor_encoder_decoder.decode_all_anchors(pred, num_anchors_per_layer)
    num_anchors_per_layer = num_anchors_per_layer
    all_num_anchors_depth = all_num_anchors_depth

    # 生成训练样本
    loc_targets, cls_targets, match_scores = anchor_encoder_fn(glabels,gbboxes)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~打印loc_targets：", loc_targets)


    # bboxes_pred = decode_fn(loc_targets[0])
    # bboxes_pred = [tf.reshape(preds, [-1, 4]) for preds in bboxes_pred]
    # bboxes_pred = tf.concat(bboxes_pred, axis=0)
    # save_image_op = tf.py_func(save_image_with_bbox,
    #                         [ssd_preprocessing.unwhiten_image(features[0]),
    #                         tf.clip_by_value(cls_targets[0], 0, tf.int64.max),
    #                         match_scores[0],
    #                         bboxes_pred],
    #                         tf.int64, stateful=True)
    # with tf.control_dependencies([save_image_op]):

    # print(all_num_anchors_depth)
    with tf.variable_scope(params['model_scope'], default_name=None, values=[features], reuse=tf.AUTO_REUSE):
        # **************************************************1.获取模型基本框架类
        backbone = ssd_net.VGG16Backbone(params['data_format'])
        # **************************************************2.获取feature_map
        print("************************************打印features:",features.graph)
        feature_layers = backbone.forward(features, training=(mode == tf.estimator.ModeKeys.TRAIN))
        # **************************************************3.使用feature map进行预测，得到预测框和类别
        # location_pred:feature_map_layer的预测列表，[(batch,feature_map_shape[0],feature_map_shape[1],anchor_per_layer_depth*4)]
        location_pred, cls_pred = ssd_net.multibox_head(feature_layers, params['num_classes'], all_num_anchors_depth,
                                                        data_format=params['data_format'])

        # **************************************************4.如果使用channels_first，需要将结果翻转回来
        if params['data_format'] == 'channels_first':
            cls_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in cls_pred]
            location_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in location_pred]

        # *************************************************5.将预测结构reshape为：（batch,num_anchors,num_classes）
        cls_pred = [tf.reshape(pred, [tf.shape(features)[0], -1, params['num_classes']]) for pred in cls_pred]
        location_pred = [tf.reshape(pred, [tf.shape(features)[0], -1, 4]) for pred in location_pred]
        # (batch,num_anchors,4)  (batch,num_anchors,num_classes)
        cls_pred = tf.concat(cls_pred, axis=1)
        location_pred = tf.concat(location_pred, axis=1)

        # ************************************************6.将所有预测结果平铺下来(去掉batch维度)
        cls_pred = tf.reshape(cls_pred, [-1, params['num_classes']])
        location_pred = tf.reshape(location_pred, [-1, 4])

    with tf.device('/cpu:0'):
        with tf.control_dependencies([cls_pred, location_pred]):
            with tf.name_scope('post_forward'):
                # bboxes_pred = decode_fn(location_pred)
                # ***************************************7.将预测结果解码，得到（ymin,xmin,ymax,ymin）格式坐标
                # 这里又将平铺的location_pred转为（batch，...，4）
                # 需要注意：训练时可以batch，验证时只能一张图片（验证时可以不用裁剪图片）
                # 这里预测的boxes_pred是以feature_layer来组织，
                print("****************************打印location_pred：",location_pred.graph)
                bboxes_pred = tf.map_fn(lambda _preds: decode_fn(_preds),
                                        tf.reshape(location_pred, [tf.shape(features)[0], -1, 4]),
                                        dtype=[tf.float32] * len(num_anchors_per_layer), back_prop=False)
                # cls_targets = tf.Print(cls_targets, [tf.shape(bboxes_pred[0]),tf.shape(bboxes_pred[1]),tf.shape(bboxes_pred[2]),tf.shape(bboxes_pred[3])])
                # ***************************************8.reshape成（batch*num_anchors,4）
                bboxes_pred = [tf.reshape(preds, [-1, 4]) for preds in bboxes_pred]
                bboxes_pred = tf.concat(bboxes_pred, axis=0)
                # *****************************************9.将样本gound_turth也平铺（去掉batch维度）
                flaten_cls_targets = tf.reshape(cls_targets, [-1])
                flaten_match_scores = tf.reshape(match_scores, [-1])
                flaten_loc_targets = tf.reshape(loc_targets, [-1, 4])

                # each positive examples has one label
                # *******************************************10.找到所有的正样本
                positive_mask = flaten_cls_targets > 0
                n_positives = tf.count_nonzero(positive_mask)
                # *******************************************11.计算正负样本的数目，为了做hard negative mining
                # batch_n_positives：（batch,num_anchors）  ignore -1??这里应该有误，应该是只计算为1的正样本个数
                batch_n_positives = tf.count_nonzero(cls_targets, -1)

                batch_negtive_mask = tf.equal(cls_targets,
                                              0)  # tf.logical_and(tf.equal(cls_targets, 0), match_scores > 0.)
                batch_n_negtives = tf.count_nonzero(batch_negtive_mask, -1)
                # ******************************************12.根据negative_ratio计算需要保留的负样本个数
                batch_n_neg_select = tf.cast(params['negative_ratio'] * tf.cast(batch_n_positives, tf.float32),
                                             tf.int32)
                batch_n_neg_select = tf.minimum(batch_n_neg_select, tf.cast(batch_n_negtives, tf.int32))

                # hard negative mining for classification
                # ****************************************13.进行softmax计算，得到背景类的概率
                # 每个step训练都要重新选择样本吗？这里用到了cls_pred
                predictions_for_bg = tf.nn.softmax(
                    tf.reshape(cls_pred, [tf.shape(features)[0], -1, params['num_classes']]))[:, :, 0]
                # *****************************************14.这里貌似有问题。背景类预测越大，应该是负样本
                prob_for_negtives = tf.where(batch_negtive_mask,
                                             0. - predictions_for_bg,
                                             # ignore all the positives
                                             0. - tf.ones_like(predictions_for_bg))
                # *********************************************15.定位背景类概率最高的k个负样本（下标）
                topk_prob_for_bg, _ = tf.nn.top_k(prob_for_negtives, k=tf.shape(prob_for_negtives)[1])
                # *********************************************使用gather_nd获取第k大negative的值
                # （batch,num_samples）
                score_at_k = tf.gather_nd(topk_prob_for_bg,
                                          tf.stack([tf.range(tf.shape(features)[0]), batch_n_neg_select - 1], axis=-1))
                # 这里才找到每个batch的negative样本
                selected_neg_mask = prob_for_negtives >= tf.expand_dims(score_at_k, axis=-1)

                # include both selected negtive and all positive examples
                # ********************************************16.对那些非正样本和非负样本，不纳入loss的计算
                #这里找到所有参与计算loss的anchors
                final_mask = tf.stop_gradient(
                    tf.logical_or(tf.reshape(tf.logical_and(batch_negtive_mask, selected_neg_mask), [-1]),
                                  positive_mask))
                total_examples = tf.count_nonzero(final_mask)
                # 这里找到所有参与计算loss的anchors
                # (all_anchors,)
                cls_pred = tf.boolean_mask(cls_pred, final_mask)
                location_pred = tf.boolean_mask(location_pred, tf.stop_gradient(positive_mask))
                flaten_cls_targets = tf.boolean_mask(tf.clip_by_value(flaten_cls_targets, 0, params['num_classes']),
                                                     final_mask)
                flaten_loc_targets = tf.stop_gradient(tf.boolean_mask(flaten_loc_targets, positive_mask))
                # ********************************************17.得到预测结果：这里预测结果的格式也没有对齐
                predictions = {
                    'classes': tf.argmax(cls_pred, axis=-1),#这里是部分参与训练的anchors
                    'probabilities': tf.reduce_max(tf.nn.softmax(cls_pred, name='softmax_tensor'), axis=-1),
                    'loc_predict': bboxes_pred}#这里是所有anchors，应该也进行boolean mask
                #*********************************************18. 计算分类的准确率
                cls_accuracy = tf.metrics.accuracy(flaten_cls_targets, predictions['classes'])
                metrics = {'cls_accuracy': cls_accuracy}

                # Create a tensor named train_accuracy for logging purposes.
                tf.identity(cls_accuracy[1], name='cls_accuracy')
                tf.summary.scalar('cls_accuracy', cls_accuracy[1])
    # ******************************************************19.构造预测时的输出
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    # cross_entropy = tf.cond(n_positives > 0, lambda: tf.losses.sparse_softmax_cross_entropy(labels=flaten_cls_targets, logits=cls_pred), lambda: 0.)# * (params['negative_ratio'] + 1.)
    # flaten_cls_targets=tf.Print(flaten_cls_targets, [flaten_loc_targets],summarize=50000)
    # *****************************************************20.训练时计算loss
    # 交叉熵
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=flaten_cls_targets, logits=cls_pred) * (
                params['negative_ratio'] + 1.)

    # Create a tensor named cross_entropy for logging purposes.
    # identity是为了创建一个tensor,为了后续的日志记录
    tf.identity(cross_entropy, name='cross_entropy_loss')
    tf.summary.scalar('cross_entropy_loss', cross_entropy)

    # loc_loss = tf.cond(n_positives > 0, lambda: modified_smooth_l1(location_pred, tf.stop_gradient(flaten_loc_targets), sigma=1.), lambda: tf.zeros_like(location_pred))
    # 边框回归loss
    loc_loss = modified_smooth_l1(location_pred, flaten_loc_targets, sigma=1.)
    # loc_loss = modified_smooth_l1(location_pred, tf.stop_gradient(gtargets))
    loc_loss = tf.reduce_mean(tf.reduce_sum(loc_loss, axis=-1), name='location_loss')
    tf.summary.scalar('location_loss', loc_loss)
    tf.losses.add_loss(loc_loss)

    # ***************************************************21.进行正则化约束（除去了bn层）
    l2_loss_vars = []
    for trainable_var in tf.trainable_variables():
        if '_bn' not in trainable_var.name:
            if 'conv4_3_scale' not in trainable_var.name:
                l2_loss_vars.append(tf.nn.l2_loss(trainable_var))
            else:
                l2_loss_vars.append(tf.nn.l2_loss(trainable_var) * 0.1)
    # Add weight decay to the loss. We exclude the batch norm variables because
    # doing so leads to a small improvement in accuracy.

    # *************************************************22.统计所有的loss,正则化loss需要加入参数
    total_loss = tf.add(cross_entropy + loc_loss,
                        tf.multiply(params['weight_decay'], tf.add_n(l2_loss_vars), name='l2_loss'), name='total_loss')

    # *************************************************23.进行训练。
    if mode == tf.estimator.ModeKeys.TRAIN:
        # ******************************************************24.获取global_step（训练的基本单位是step）
        global_step = tf.train.get_or_create_global_step()
        # ******************************************************25.根据global_step选择学习率
        lr_values = [params['learning_rate'] * decay for decay in params['lr_decay_factors']]
        learning_rate = tf.train.piecewise_constant(tf.cast(global_step, tf.int32),
                                                    [int(_) for _ in params['decay_boundaries']],
                                                    lr_values)
        # *****************************************************26.对学习率进行截断：防止学习率过大
        truncated_learning_rate = tf.maximum(learning_rate,
                                             tf.constant(params['end_learning_rate'], dtype=learning_rate.dtype),
                                             name='learning_rate')
        # Create a tensor named learning_rate for logging purposes.
        tf.summary.scalar('learning_rate', truncated_learning_rate)
        # ****************************************************27.设置优化器
        optimizer = tf.train.MomentumOptimizer(learning_rate=truncated_learning_rate,
                                               momentum=params['momentum'])
        # optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

        # Batch norm requires update_ops to be added as a train_op dependency.
        # 需要注意，batch normal涉及到更新每一层的feature maps
        # ****************************************************28.注意依赖，防止最后进行优化时，用了没有更新的参数
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(total_loss, global_step)
    else:
        train_op = None
    # *******************************************************29.返回训练用的estimator实例
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=total_loss,
        train_op=train_op,
        eval_metric_ops=metrics,
        scaffold=tf.train.Scaffold(init_fn=None))


def parse_comma_list(args):
    return [float(s.strip()) for s in args.split(',')]


def main(_):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    # **************************************************************1.进行硬件相关配置
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    # *************************************************************2.配置多GPU训练
    strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=2)
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False,
                            intra_op_parallelism_threads=FLAGS.num_cpu_threads,
                            inter_op_parallelism_threads=FLAGS.num_cpu_threads,
                            gpu_options=gpu_options, )

    # num_gpus = validate_batch_size_for_multi_gpu(FLAGS.batch_size)

    # Set up a RunConfig to only save checkpoints once per training cycle.
    # *****************************************************************3.配置estimator
    run_config = tf.estimator.RunConfig().replace(
        save_checkpoints_secs=FLAGS.save_checkpoints_secs).replace(
        save_checkpoints_steps=None).replace(
        save_summary_steps=FLAGS.save_summary_steps).replace(
        keep_checkpoint_max=5).replace(
        tf_random_seed=FLAGS.tf_random_seed).replace(
        log_step_count_steps=FLAGS.log_every_n_steps).replace(
        session_config=config).replace(
        # 添加gpu训练策略
        train_distribute=strategy)

    # replicate_ssd_model_fn = tf.contrib.estimator.replicate_model_fn(ssd_model_fn, loss_reduction=tf.losses.Reduction.MEAN)
    # ********************************************************************4.构建estimator
    ssd_detector = tf.estimator.Estimator(
        model_fn=ssd_model_fn, model_dir=FLAGS.model_dir, config=run_config,
        params={
            'num_gpus': 2,
            'data_format': FLAGS.data_format,
            'batch_size': FLAGS.batch_size,
            'model_scope': FLAGS.model_scope,
            'num_classes': FLAGS.num_classes,
            'negative_ratio': FLAGS.negative_ratio,
            'match_threshold': FLAGS.match_threshold,
            'neg_threshold': FLAGS.neg_threshold,
            'weight_decay': FLAGS.weight_decay,
            'momentum': FLAGS.momentum,
            'learning_rate': FLAGS.learning_rate,
            'end_learning_rate': FLAGS.end_learning_rate,
            'decay_boundaries': parse_comma_list(FLAGS.decay_boundaries),
            'lr_decay_factors': parse_comma_list(FLAGS.lr_decay_factors),
        })
    # **********************************************************5.定义需要监控的tensor
    tensors_to_log = {
        'lr': 'learning_rate',
        'ce': 'cross_entropy_loss',
        'loc': 'location_loss',
        'loss': 'total_loss',
        'l2': 'l2_loss',
        'acc': 'post_forward/cls_accuracy',
    }
    # ********************************************************6.构造监控用的hook,用于接入训练的运行时环境
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=FLAGS.log_every_n_steps,
                                              formatter=lambda dicts: (
                                                  ', '.join(['%s=%.6f' % (k, v) for k, v in dicts.items()])))

    # hook = tf.train.ProfilerHook(save_steps=50, output_dir='.', show_memory=True)
    print('Starting a training cycle.')
    ssd_detector.train(
        input_fn=input_pipeline(file_pattern='/home/dxfang/dataset/tfrecords/pascal_voc/train-000*', is_training=True, batch_size=FLAGS.batch_size),
        hooks=[logging_hook], max_steps=FLAGS.max_number_of_steps)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

    # cls_targets = tf.reshape(cls_targets, [-1])
    # match_scores = tf.reshape(match_scores, [-1])
    # loc_targets = tf.reshape(loc_targets, [-1, 4])

    # # each positive examples has one label
    # positive_mask = cls_targets > 0
    # n_positives = tf.count_nonzero(positive_mask)

    # negtive_mask = tf.logical_and(tf.equal(cls_targets, 0), match_scores > 0.)
    # n_negtives = tf.count_nonzero(negtive_mask)

    # n_neg_to_select = tf.cast(params['negative_ratio'] * tf.cast(n_positives, tf.float32), tf.int32)
    # n_neg_to_select = tf.minimum(n_neg_to_select, tf.cast(n_negtives, tf.int32))

    # # hard negative mining for classification
    # predictions_for_bg = tf.nn.softmax(cls_pred)[:, 0]

    # prob_for_negtives = tf.where(negtive_mask,
    #                        0. - predictions_for_bg,
    #                        # ignore all the positives
    #                        0. - tf.ones_like(predictions_for_bg))
    # topk_prob_for_bg, _ = tf.nn.top_k(prob_for_negtives, k=n_neg_to_select)
    # selected_neg_mask = prob_for_negtives > topk_prob_for_bg[-1]

    # # include both selected negtive and all positive examples
    # final_mask = tf.stop_gradient(tf.logical_or(tf.logical_and(negtive_mask, selected_neg_mask), positive_mask))
    # total_examples = tf.count_nonzero(final_mask)

    # glabels = tf.boolean_mask(tf.clip_by_value(cls_targets, 0, FLAGS.num_classes), final_mask)
    # cls_pred = tf.boolean_mask(cls_pred, final_mask)
    # location_pred = tf.boolean_mask(location_pred, tf.stop_gradient(positive_mask))
    # loc_targets = tf.boolean_mask(loc_targets, tf.stop_gradient(positive_mask))
