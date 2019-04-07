import functools

import tensorflow as tf

import eval_util
import inputs
from net import ssd_net
from net import se_ssd_net
# from train_ssd import FLAGS
from utils import anchor_manipulator, visualization_utils, scaffolds
from utils.postprocessing import per_image_post_process
from utils.shape_util import pad_or_clip_nd, unpad_tensor

global_anchor_info = inputs.global_anchor_info

VOC_LABELS = {
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}

def get_init_fn(model_dir,checkpoint_path,model_scope,checkpoint_model_scope,checkpoint_exclude_scopes,ignore_missing_vars):
    return scaffolds.get_init_fn_for_scaffold(model_dir, checkpoint_path,
                                              model_scope, checkpoint_model_scope,
                                              checkpoint_exclude_scopes, ignore_missing_vars,
                                              name_remap={'/kernel': '/weights', '/bias': '/biases'},
                                              )

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


def predict(features,mode,params,all_num_anchors_depth):
    with tf.variable_scope(params["model_scope"], default_name=None, values=[features], reuse=tf.AUTO_REUSE):
        # get vgg16 net
        # model = ssd_net.VGG16Backbone(params['data_format'],backbone_batch_normal=params["backbone_batch_normal"],additional_batch_normal=params['additional_batch_normal'])
        model = se_ssd_net.SE_SSD300_NET(params['data_format'],backbone_batch_normal=params["backbone_batch_normal"],additional_batch_normal=params['additional_batch_normal'])
        # return feature layers
        # feature_layers -> ['conv4','fc7','conv8','conv9','conv10','conv11']
        feature_layers = model.forward(features, training=(mode == tf.estimator.ModeKeys.TRAIN))

        # location_pred -> [[batch_size, num_anchor_per_position*4, feature_map_size, feature_map_size,],[],[],[],[],[]]
        # cls_pred -> [[batch_size, num_anchor_per_position*21,   , feature_map_size,],[],[],[],[],[]]
        location_pred, cls_pred = model.multibox_head(feature_layers, params['num_classes'], all_num_anchors_depth,
                                                        data_format=params['data_format'],bn_detection_head=params['bn_detection_head'],
                                                         training=(mode == tf.estimator.ModeKeys.TRAIN))

        # whether using "channels_first", can accelerate calculation
        if params['data_format'] == 'channels_first':
            cls_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in cls_pred]
            location_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in location_pred]

        cls_pred = [tf.reshape(pred, [tf.shape(features)[0], -1, params['num_classes']]) for pred in cls_pred]
        location_pred = [tf.reshape(pred, [tf.shape(features)[0], -1, 4]) for pred in location_pred]
        # cls_pred -> shape: [batch_size,num_all_anchors,21]
        # cls_pred -> shape: [batch_size,num_all_anchors,4]
        cls_pred = tf.concat(cls_pred, axis=1)
        location_pred = tf.concat(location_pred, axis=1)
        return cls_pred, location_pred

def hard_example_mining(cls_targets, cls_pred, params):
    """ 
    :param cls_targets  [batch_size,num_all_anchors]
    :param cls_pred  [batch_size,num_all_anchors,21]
    :return:
        final_mask: for ce_loss  [batch_size*num_all_anchors,]
        positive_mask: for loc_loss  [batch_size*num_all_anchors,]
    """
    with tf.name_scope('hard_example_mining'):
        # flatten
        flatten_cls_targets = tf.reshape(cls_targets, [-1])
        # find all positive anchors
        positive_mask = flatten_cls_targets > 0

        # batch_n_positives：->shape: [batch_size,]
        batch_n_positives_mask = tf.greater(cls_targets, 0)
        batch_n_positives = tf.count_nonzero(batch_n_positives_mask, -1)
        tf.identity(batch_n_positives[0], name="num_positives")

        batch_negtive_mask = tf.equal(cls_targets, 0)
        batch_n_negtives = tf.count_nonzero(batch_negtive_mask, -1)

        # number of negative
        batch_n_neg_select = tf.cast(params['negative_ratio'] * tf.cast(batch_n_positives, tf.float32),
                                     tf.int32)
        batch_n_neg_select = tf.minimum(batch_n_neg_select, tf.cast(batch_n_negtives, tf.int32))
        tf.identity(batch_n_neg_select[0], name="num_negatives_select")

        # hard negative mining  select negative depend on   predictions_for_bg  not matched_IOU_score

        # predictions_for_bg -> shape: [batch_size,num_all_anchors,]
        predictions_for_bg = tf.nn.softmax(cls_pred)[:, :, 0]
        # # prob_for_negtives -> shape: [batch_size,num_all_anchors,]
        prob_for_negtives = tf.where(batch_negtive_mask,
                                     0. - predictions_for_bg,
                                     0. - tf.ones_like(predictions_for_bg))

        # default sorted
        # topk_prob_for_bg -> shape: [batch_size,num_all_anchors,]
        topk_prob_for_bg, _ = tf.nn.top_k(prob_for_negtives, k=tf.shape(prob_for_negtives)[1])
        # score_at_k : anchors grater than this score will have lower probability( hard negative example)
        score_at_k = tf.gather_nd(topk_prob_for_bg,
                                  tf.stack([tf.range(tf.shape(cls_targets)[0]), batch_n_neg_select - 1], axis=-1))
        selected_neg_mask = prob_for_negtives >= tf.expand_dims(score_at_k, axis=-1)

        # include both selected negtive and all positive examples
        negative_mask = tf.reshape(tf.logical_and(batch_negtive_mask, selected_neg_mask), [-1])
        final_mask = tf.stop_gradient(tf.logical_or(negative_mask,positive_mask))

        return final_mask, positive_mask


def build_losses(cls_targets=None,cls_pred=None, loc_targets=None,loc_pred=None, params=None,is_task_A=False,is_task_B=False):

    # if is_task_A:
    #     cls_pred = cls_pred[:, :11]
    # if is_task_B:
    #     cls_pred = tf.gather(cls_pred, [0]+list(range(11, 21)), axis=1)

    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=cls_targets,logits=cls_pred) * (params['negative_ratio'] + 1.)
    tf.identity(cross_entropy, name='cross_entropy_loss')
    tf.summary.scalar('cross_entropy_loss', cross_entropy)

    # loc_loss = tf.cond(n_positives > 0, lambda: modified_smooth_l1(location_pred, tf.stop_gradient(flatten_loc_targets), sigma=1.), lambda: tf.zeros_like(location_pred))
    loc_loss = modified_smooth_l1(loc_pred, loc_targets,sigma=1.)
    loc_loss = tf.reduce_mean(tf.reduce_sum(loc_loss, axis=-1), name='location_loss')
    tf.summary.scalar('location_loss', loc_loss)
    tf.losses.add_loss(loc_loss)

    # loss for regularization
    l2_loss_vars = []
    for trainable_var in tf.trainable_variables():
        if '_bn' not in trainable_var.name:
            if 'conv4_3_scale' not in trainable_var.name:
                l2_loss_vars.append(tf.nn.l2_loss(trainable_var))
            else:
                l2_loss_vars.append(tf.nn.l2_loss(trainable_var) * 0.1)
    # Add weight decay to the loss. We exclude the batch norm variables because
    # doing so leads to a small improvement in accuracy.
    l2_loss = tf.multiply(params['weight_decay'], tf.add_n(l2_loss_vars), name='l2_loss')

    return cross_entropy, loc_loss, l2_loss

def dist_build_losses(cls_targets=None,cls_pred=None, loc_targets=None,loc_pred=None, params=None):
    cls_targets = cls_targets[:]
    cls_pred = cls_pred[:,:11]
    # loc_targets = None
    # loc_pred = None

    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=cls_targets, logits=cls_pred) * (
                params['negative_ratio'] + 1.)
    tf.identity(cross_entropy, name='cross_entropy_loss')
    tf.summary.scalar('cross_entropy_loss', cross_entropy)

    # loc_loss = tf.cond(n_positives > 0, lambda: modified_smooth_l1(location_pred, tf.stop_gradient(flatten_loc_targets), sigma=1.), lambda: tf.zeros_like(location_pred))
    loc_loss = modified_smooth_l1(loc_pred, loc_targets, sigma=1.)
    loc_loss = tf.reduce_mean(tf.reduce_sum(loc_loss, axis=-1), name='location_loss')
    tf.summary.scalar('location_loss', loc_loss)
    tf.losses.add_loss(loc_loss)

    # loss for regularization
    l2_loss_vars = []
    for trainable_var in tf.trainable_variables():
        if '_bn' not in trainable_var.name:
            if 'conv4_3_scale' not in trainable_var.name:
                l2_loss_vars.append(tf.nn.l2_loss(trainable_var))
            else:
                l2_loss_vars.append(tf.nn.l2_loss(trainable_var) * 0.1)
    # Add weight decay to the loss. We exclude the batch norm variables because
    # doing so leads to a small improvement in accuracy.
    l2_loss = tf.multiply(params['weight_decay'], tf.add_n(l2_loss_vars), name='l2_loss')

    return cross_entropy, loc_loss, l2_loss


def build_distillation_loss(features,cls_pred,location_pred,mode,all_num_anchors_depth,params):
    with tf.variable_scope('distillation', values=[features]):
        # dist_backbone = ssd_net.VGG16Backbone(params['data_format'])
        dist_backbone = se_ssd_net.SE_SSD300_NET(params['data_format'])
        dist_feature_layers = dist_backbone.forward(features, training=(mode == tf.estimator.ModeKeys.TRAIN))
        dist_location_pred, dist_cls_pred = dist_backbone.multibox_head(dist_feature_layers, params['num_classes'],
                                                                  all_num_anchors_depth,
                                                                  data_format=params['data_format'])
        if params['data_format'] == 'channels_first':
            dist_cls_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in dist_cls_pred]
            dist_location_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in dist_location_pred]
        dist_cls_pred = [tf.reshape(pred, [tf.shape(features)[0], -1, params['num_classes']]) for pred in dist_cls_pred]
        dist_location_pred = [tf.reshape(pred, [tf.shape(features)[0], -1, 4]) for pred in dist_location_pred]
        dist_cls_pred = tf.concat(dist_cls_pred, axis=1)
        dist_location_pred = tf.concat(dist_location_pred, axis=1)
        dist_cls_pred = tf.reshape(dist_cls_pred, [-1, params['num_classes']])
        dist_location_pred = tf.reshape(dist_location_pred, [-1, 4])
        # class_distillation_loss
        cls_logits = cls_pred[:, :params['cached_classes'] + 1] - tf.reduce_mean(
            cls_pred[:, :params['cached_classes'] + 1], axis=1, keep_dims=True)
        cls_distillated_logits = dist_cls_pred[:, :params['cached_classes'] + 1] - tf.reduce_mean(
            dist_cls_pred[:, :params['cached_classes'] + 1], axis=1, keep_dims=True)
        class_distillation_loss = tf.reduce_mean(tf.square(cls_logits - cls_distillated_logits),
                                                 name='class_distillation_loss')
        class_distillation_loss *= params.get('class_distillation_loss_coef', 1)
        tf.summary.scalar('class_distillation_loss', class_distillation_loss)

        # 只惩罚那些negative的bboxes?
        bboxes_distillation_loss = tf.reduce_mean(
            tf.reduce_mean(tf.square(location_pred - dist_location_pred), axis=1, keep_dims=True),
            name='bboxes_distillation_loss')
        bboxes_distillation_loss *= params.get('bbox_distillation_loss_coef', 1)
        tf.summary.scalar('bboxes_distillation_loss', bboxes_distillation_loss)

        return class_distillation_loss, bboxes_distillation_loss


def ssd_model_fn(features, labels, mode, params):

    global global_anchor_info
    # (anchor_cy, anchor_cx, anchor_h, anchor_w)  -> shape: [num_all_anchors,]
    all_anchors = global_anchor_info["all_anchors"]
    all_num_anchors_depth = global_anchor_info["all_num_anchors_depth"]
    all_num_anchors_spatial = global_anchor_info["all_num_anchors_spatial"]
    # calculate the number of anchors of each feature layer.
    num_anchors_per_layer = [depth * spatial for depth, spatial in zip(all_num_anchors_depth, all_num_anchors_spatial)]

    processed_image = features['preprocessed_image']

    # for prediction
    shape = features['original_shape']
    filename = features['filename']
    original_image_spatial_shape = features['original_image_spatial_shape']
    true_image_shape = features['true_image_shape']
    key = features["key"]



    anchor_encoder_decoder = anchor_manipulator.AnchorEncoder(positive_threshold=params['positive_threshold'],
                                                              neg_threshold=params['neg_threshold'],
                                                              prior_scaling=[0.1, 0.1, 0.2, 0.2],
                                                              allowed_borders=[1.0] * 6)
    # encode function for anchors: assign labels by the iou_matrix
    anchor_encoder_fn = functools.partial(anchor_encoder_decoder.encode_all_anchors,
                                          all_anchors=all_anchors,
                                          all_num_anchors_depth=all_num_anchors_depth,
                                          all_num_anchors_spatial=all_num_anchors_spatial)

    # decode function for anchors: convert the location_pred  to anchor coordinates
    decode_fn = functools.partial(anchor_encoder_decoder.decode_all_anchors,
                                  all_anchors=all_anchors,
                                  num_anchors_per_layer=num_anchors_per_layer)


    # cls_pred -> shape: [batch_size,num_all_anchors,21]
    # location_pred -> shape: [batch_size,num_all_anchors,4]
    cls_pred, location_pred = predict(processed_image,mode,params,all_num_anchors_depth)

    # decode boxes  这里可能有问题？ bboxes_pred 的所有anchors 是按照feature layer的顺序排列的

    # bboxes_pred ->shape: [[batch_size,num_all_anchors,4],[][][][][]]
    bboxes_pred = tf.map_fn(decode_fn, location_pred, back_prop=False)

    # calculate accuracy
    with tf.control_dependencies([cls_pred, location_pred]):
        with tf.name_scope('post_forward'):
            post_process_for_single_example = functools.partial(per_image_post_process,
                                                                select_threshold=params['select_threshold'],
                                                                min_size=params['min_size'],
                                                                keep_topk=params['keep_topk'],
                                                                nms_topk=params['nms_topk'],
                                                                nms_threshold=params['nms_threshold'],
                                                                is_tack_A=params['is_tack_A'],
                                                                is_task_B=params['is_tack_B'],
                                                                num_classes = params['num_classes']
                                                                )
    if mode == tf.estimator.ModeKeys.PREDICT:
        cls_pred = tf.reshape(cls_pred, [-1, params['num_classes']])
        bboxes_pred = tf.reshape(bboxes_pred, [-1, 4])
        # cls_pred = tf.squeeze(cls_pred, axis=0)
        # bboxes_pred = tf.squeeze(bboxes_pred, axis=0)
        selected_bboxes,selected_scores = post_process_for_single_example(cls_pred, bboxes_pred, mode)
        predictions = {'filename': filename, 'shape': shape}
        for class_ind in range(1, params['num_classes']):
            predictions['scores_{}'.format(class_ind)] = tf.expand_dims(selected_scores[class_ind], axis=0)
            predictions['bboxes_{}'.format(class_ind)] = tf.expand_dims(selected_bboxes[class_ind], axis=0)
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=None, train_op=None)

    # assignment target
    num_groundtruth_boxes = labels['num_groundtruth_boxes']
    groundtruth_classes = labels['groundtruth_classes']
    groundtruth_boxes = labels['groundtruth_boxes']

    #  revert normalize coordinate to absolute coordinate  for metircs
    # scale = tf.cast(tf.expand_dims(tf.stack([shape[:,0],shape[:,1],shape[:,0],shape[:,1]],axis=1),axis=1),dtype=tf.float32)
    # print(scale)
    # groundtruth_boxes = tf.multiply(groundtruth_boxes,scale)

    # unpadding  num_boxes dimension for real num_groundtruth_boxes
    unpaded_groundtruth_classes = unpad_tensor(groundtruth_classes, num_groundtruth_boxes)
    unpaded_groundtruth_boxes = unpad_tensor(groundtruth_boxes, num_groundtruth_boxes)
    # construct train example
    loc_targets_list, cls_targets_list, matched_iou_scores_list = [], [], []
    for _groundtruth_classes, _groundtruth_boxes in zip(unpaded_groundtruth_classes, unpaded_groundtruth_boxes):
        loc_target, cls_target, matched_iou_score = anchor_encoder_fn(labels=_groundtruth_classes,
                                                                      bboxes=_groundtruth_boxes)
        loc_targets_list.append(loc_target)
        cls_targets_list.append(cls_target)
        matched_iou_scores_list.append(matched_iou_score)

    # loc_targets ->shape: [batch_size, num_all_anchors,4]
    # cls_targets ->shape: [batch_size, num_all_anchors]
    # matched_iou_scores ->shape: [batch_size, num_all_anchors]
    loc_targets = tf.stack(loc_targets_list, axis=0)
    cls_targets = tf.stack(cls_targets_list, axis=0)
    matched_iou_scores = tf.stack(matched_iou_scores_list, axis=0)

    # hard example mining
    final_mask, positive_mask = hard_example_mining(cls_targets, cls_pred, params)
    # flatten targets
    flatten_cls_targets = tf.reshape(cls_targets, [-1])
    flatten_loc_targets = tf.reshape(loc_targets, [-1, 4])
    # flatten preds
    flatten_cls_pred = tf.reshape(cls_pred, [-1, params['num_classes']])
    flatten_location_pred = tf.reshape(location_pred, [-1, 4])
    # apply hard_example_mining
    flatten_cls_pred_hard_exam_mining = tf.boolean_mask(flatten_cls_pred, final_mask)
    flatten_location_pred_hard_exam_mining = tf.boolean_mask(flatten_location_pred,
                                                             tf.stop_gradient(positive_mask))
    flatten_cls_targets_hard_exam_mining = tf.boolean_mask(
        tf.clip_by_value(flatten_cls_targets, 0, params['num_classes']),
        final_mask)
    flatten_loc_targets_hard_exam_mining = tf.stop_gradient(
        tf.boolean_mask(flatten_loc_targets, positive_mask))

    # classification accuracy
    cls_accuracy = tf.metrics.accuracy(flatten_cls_targets_hard_exam_mining,
                                       tf.argmax(flatten_cls_pred_hard_exam_mining, axis=-1))
    tf.identity(cls_accuracy[1], name='cls_accuracy')
    tf.summary.scalar('cls_accuracy', cls_accuracy[1])
    # losses
    cross_entropy, loc_loss, l2_loss = build_losses(cls_targets=flatten_cls_targets_hard_exam_mining,
                                                    cls_pred=flatten_cls_pred_hard_exam_mining,
                                                    loc_targets=flatten_loc_targets_hard_exam_mining,
                                                    loc_pred=flatten_location_pred_hard_exam_mining,
                                                    params=params)
    total_loss = tf.add_n([cross_entropy, loc_loss, l2_loss], name='total_loss')

    #  evaluation
    metrics = {}
    if mode == tf.estimator.ModeKeys.EVAL:
        with tf.name_scope("evaluation_scope"):
            cls_pred_list = tf.unstack(cls_pred)
            bboxes_pred_list = tf.unstack(bboxes_pred)
            detection_boxes_list = []
            detection_scores_list = []
            detection_classes_list = []
            num_detections_list = []
            for ind,(cls_pred_, bboxes_pred_ )in enumerate(zip(cls_pred_list, bboxes_pred_list)):
                detection_boxes, detection_scores, detection_classes, \
                num_detections = post_process_for_single_example(cls_pred_, bboxes_pred_, mode)

                # revert normalize coordinate to absolute coordinate  for metircs
                # detection_boxes = tf.multiply(detection_boxes,[shape[ind][0],shape[ind][1],shape[ind][0],shape[ind][1]])+1

                # padding along num_detections dimension
                detection_boxes = pad_or_clip_nd(detection_boxes, [params['pad_nms_detections'], 4])
                detection_scores = pad_or_clip_nd(detection_scores, [params['pad_nms_detections']])
                detection_classes = pad_or_clip_nd(detection_classes, [params['pad_nms_detections']])

                detection_boxes_list.append(detection_boxes)
                detection_scores_list.append(detection_scores)
                detection_classes_list.append(detection_classes)
                num_detections_list.append(num_detections)

            # revert normalize coordinate to absolute coordinate  for metircs
            # scaled_unpaded_groundtruth_boxes = [tf.multiply(unpaded_groundtruth_box, [shape[ind][0], shape[ind][1], shape[ind][0], shape[ind][1]]) + 1
            #                                                     for ind,unpaded_groundtruth_box in enumerate(unpaded_groundtruth_boxes)]

            # batched detections
            # detection_boxes -> shape: [batch_size,max_nms_detections，4]
            # detection_scores -> shape: [batch_size,max_nms_detections，]
            # detection_classes -> shape: [batch_size,max_nms_detections，]   labels: 1,2,...20
            detection_boxes = tf.stack(detection_boxes_list, axis=0)
            detection_scores = tf.stack(detection_scores_list, axis=0)
            detection_classes = tf.cast(tf.stack(detection_classes_list, axis=0), tf.int32)
            num_detections = tf.cast(tf.stack(num_detections_list, axis=0), tf.int32)

            tf.identity(num_detections[0], name="num_detections_after_nms")

            eval_input_dict = {
                'original_image_spatial_shape': original_image_spatial_shape,
                'true_image_shape': true_image_shape,
                'original_image': features['original_image'],
                # goundtruths
                'num_groundtruth_boxes_per_image': num_groundtruth_boxes,
                'groundtruth_boxes': unpaded_groundtruth_boxes,
                'groundtruth_classes': unpaded_groundtruth_classes,
                # detections
                'detection_boxes': detection_boxes,
                'detection_scores': detection_scores,
                'detection_classes': detection_classes,
                "num_det_boxes_per_image": num_detections,
                # image id
                'key': key
            }

            category_index = {VOC_LABELS[name][0]: {"id": VOC_LABELS[name][0], "name": name} for name in
                              VOC_LABELS.keys()}
            categories = list(category_index.values())
            # visualization detections result
            # eval_metric_op_vis = visualization_utils.VisualizeSingleFrameDetections(
            #     category_index,
            #     max_examples_to_draw=params['max_examples_to_draw'],
            #     max_boxes_to_draw=params['max_boxes_to_draw'],
            #     min_score_thresh=params['min_score_thresh'],
            #     use_normalized_coordinates=False)
            # vis_metric_ops = eval_metric_op_vis.get_estimator_eval_metric_ops(eval_input_dict)
            # metrics.update(vis_metric_ops)

            eval_input_dict['groundtruth_boxes'] = groundtruth_boxes
            eval_input_dict['groundtruth_classes'] = groundtruth_classes
            # eval metrics ops
            evaluator = eval_util.get_evaluators(categories, eval_metric_fn_key=params["eval_metric_fn_key"])
            eval_metric_ops = evaluator.get_estimator_eval_metric_ops(eval_input_dict)
            metrics.update(eval_metric_ops)

    if mode == tf.estimator.ModeKeys.TRAIN:
        # distillate knowledge
        if params["distillation"] == True:
            class_distillation_loss, bboxes_distillation_loss = build_distillation_loss(processed_image,cls_pred,location_pred,mode,all_num_anchors_depth,params)
            total_loss = tf.add_n(total_loss,class_distillation_loss,bboxes_distillation_loss,name="total_loss_with_distillation")
            tf.summary.scalar('total_loss_with_distillation', total_loss)

        global_step = tf.train.get_or_create_global_step()
        # dynamic learning rate
        lr_values = [params['learning_rate'] * decay for decay in params['lr_decay_factors']]
        learning_rate = tf.train.piecewise_constant(tf.cast(global_step, tf.int32),
                                                    [int(_) for _ in params['decay_boundaries']],
                                                    lr_values)
        # learning_rate = tf.constant(params['learning_rate'])
        # execute truncated_learning_rate
        truncated_learning_rate = tf.maximum(learning_rate,
                                             tf.constant(params['end_learning_rate'], dtype=learning_rate.dtype),
                                             name='learning_rate')
        # Create a tensor named learning_rate for logging purposes.
        tf.summary.scalar('learning_rate', truncated_learning_rate)

        optimizer = tf.train.MomentumOptimizer(learning_rate=truncated_learning_rate,
                                               momentum=params['momentum'])
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # Batch norm requires update_ops to be added as a train_op dependency.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(total_loss, global_step)
    else:
        train_op = None
    #
    # init_op = tf.train.init_from_checkpoint(ckpt_dir_or_file=params['checkpoint_path'])
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=None,
        loss=total_loss,
        train_op=train_op,
        eval_metric_ops=metrics,
        scaffold=tf.train.Scaffold(init_fn=get_init_fn(params['model_dir'],params['checkpoint_path'],params['model_scope'],params['checkpoint_model_scope'],params['checkpoint_exclude_scopes'],params['ignore_missing_vars'])),
        # scaffold=tf.train.Scaffold(init_op=init_op)
    )
