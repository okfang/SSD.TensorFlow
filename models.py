import tensorflow as tf

import eval_util
import inputs
from net import ssd_net
from utils import anchor_manipulator, visualization_utils
from utils.postprocessing import per_image_post_process
from utils.shape_util import pad_or_clip_nd, unpad_tensor

global_anchor_info = inputs.global_anchor_info

VOC_LABELS = {
    'none': (0, 'Background'),
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

def ssd_model_fn(features, labels, mode, params):
    """model_fn for SSD to be used with our Estimator."""
    print("-------------------------------------features tensor:",features)
    num_groundtruth_boxes = labels['num_groundtruth_boxes']
    groundtruth_classes = labels['groundtruth_classes']
    groundtruth_boxes = labels['groundtruth_boxes']
    original_image_spatial_shape =  labels['original_image_spatial_shape']
    true_image_shape = labels['true_image_shape']

    # unpadding  num_boxes dimension for real num_groundtruth_boxes
    groundtruth_classes_list = unpad_tensor(groundtruth_classes,num_groundtruth_boxes)
    groundtruth_boxes_list = unpad_tensor(groundtruth_boxes,num_groundtruth_boxes)

    #  generate anchors
    #  resize anchors can be totally revert by the proportion of resized image and original image
    # ,so there is no need to generate different anchors for different process
    global global_anchor_info
    all_anchors = global_anchor_info["all_anchors"]
    all_num_anchors_depth = global_anchor_info["all_num_anchors_depth"]
    all_num_anchors_spatial = global_anchor_info["all_num_anchors_spatial"]

    # calculate the number of anchors in each feature layer.
    num_anchors_per_layer = [depth*spatial for depth,spatial in
                             zip(all_num_anchors_depth,all_num_anchors_spatial)]

    anchor_encoder_decoder = anchor_manipulator.AnchorEncoder(positive_threshold=params['match_threshold'],
                                                              ignore_threshold=params['neg_threshold'],
                                                              prior_scaling=[0.1, 0.1, 0.2, 0.2],
                                                              allowed_borders=[1.0] * 6)
    # encode function for anchors: assign labels by the iou matrics
    anchor_encoder_fn = lambda glabels_, gbboxes_: anchor_encoder_decoder.encode_all_anchors(glabels_, gbboxes_,
                                                                                             all_anchors,
                                                                                             all_num_anchors_depth,
                                                                                             all_num_anchors_spatial)

    # decode function for anchors: convert the prediction to anchor coordinate
    decode_fn = lambda pred: anchor_encoder_decoder.decode_all_anchors(pred, all_anchors,num_anchors_per_layer)

    # use "anchor_encoder_fn" to calculate the true labels:
    loc_targets_list, cls_targets_list, match_scores_list = [], [], []
    for _groundtruth_classes, _groundtruth_boxes in zip(groundtruth_classes_list,groundtruth_boxes_list):
        loc_target,cls_target,match_score = anchor_encoder_fn(_groundtruth_classes, _groundtruth_boxes)
        loc_targets_list.append(loc_target)
        cls_targets_list.append(cls_target)
        match_scores_list.append(match_score)
    loc_targets = tf.stack(loc_targets_list,axis=0)
    cls_targets = tf.stack(cls_targets_list,axis=0)
    match_scores = tf.stack(match_scores_list,axis=0)

    with tf.variable_scope(params['model_scope'], default_name=None, values=[features], reuse=tf.AUTO_REUSE):
        # get vgg16 net
        backbone = ssd_net.VGG16Backbone(params['data_format'])

        # calculate the feature layers and return
        feature_layers = backbone.forward(features, training=(mode == tf.estimator.ModeKeys.TRAIN))

        # execute prediction, and the prediction organized like:
        # [(batch,feature_map_shape[0],feature_map_shape[1],anchor_per_position*4)]
        location_pred, cls_pred = ssd_net.multibox_head(feature_layers, params['num_classes'], all_num_anchors_depth,
                                                        data_format=params['data_format'])

        # whether using "channels_first", can accelerate calculation
        if params['data_format'] == 'channels_first':
            cls_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in cls_pred]
            location_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in location_pred]

        # flatten tensor
        cls_pred = [tf.reshape(pred, [tf.shape(features)[0], -1, params['num_classes']]) for pred in cls_pred]
        location_pred = [tf.reshape(pred, [tf.shape(features)[0], -1, 4]) for pred in location_pred]
        # final shape: (batch,num_anchors,4)  (batch,num_anchors,num_classes)
        cls_pred = tf.concat(cls_pred, axis=1)
        location_pred = tf.concat(location_pred, axis=1)
        cls_pred = tf.reshape(cls_pred, [-1, params['num_classes']])
        location_pred = tf.reshape(location_pred, [-1, 4])

    with tf.device('/cpu:0'):
        with tf.control_dependencies([cls_pred, location_pred]):
            with tf.name_scope('post_forward'):
                # bboxes_pred = decode_fn(location_pred)

                # decode predictions to a list which element has shape[(num_anchors_per_layer[0],4),(num_anchors_per_layer[1],4),...]
                bboxes_pred = tf.map_fn(lambda _preds: decode_fn(_preds),
                                        tf.reshape(location_pred, [tf.shape(features)[0], -1, 4]),
                                        dtype=[tf.float32] * len(num_anchors_per_layer), back_prop=False)

                bboxes_pred = [tf.reshape(preds, [-1, 4]) for preds in bboxes_pred]
                bboxes_pred = tf.concat(bboxes_pred, axis=0)

                # flaten for calculate accuracy
                flaten_cls_targets = tf.reshape(cls_targets, [-1])
                flaten_match_scores = tf.reshape(match_scores, [-1])
                flaten_loc_targets = tf.reshape(loc_targets, [-1, 4])

                # hard negative mining for claculate loss

                # each positive examples has one label
                # find all positive anchors
                positive_mask = flaten_cls_targets > 0
                n_positives = tf.count_nonzero(positive_mask)

                # batch_n_positives：（batch,num_anchors） ?? cls_targets: -1 indicate ignore; 0 indicate background ;else
                batch_n_positives = tf.count_nonzero(cls_targets, -1)
                tf.identity(batch_n_positives[0],name="num_positives")

                # tf.logical_and(tf.equal(cls_targets, 0), match_scores > 0.)
                batch_negtive_mask = tf.equal(cls_targets,0)
                batch_n_negtives = tf.count_nonzero(batch_negtive_mask, -1)

                # number of negative anchors should be retained
                batch_n_neg_select = tf.cast(params['negative_ratio'] * tf.cast(batch_n_positives, tf.float32),
                                             tf.int32)
                batch_n_neg_select = tf.minimum(batch_n_neg_select, tf.cast(batch_n_negtives, tf.int32))
                tf.identity(batch_n_neg_select[0], name="num_negatives_select")

                # hard negative mining for classification
                predictions_for_bg = tf.nn.softmax(
                    tf.reshape(cls_pred, [tf.shape(features)[0], -1, params['num_classes']]))[:, :, 0]
                prob_for_negtives = tf.where(batch_negtive_mask,
                                             0. - predictions_for_bg,
                                             # ignore all the positives
                                             0. - tf.ones_like(predictions_for_bg))

                # choose top k negatives, which has high probability to be background
                topk_prob_for_bg, _ = tf.nn.top_k(prob_for_negtives, k=tf.shape(prob_for_negtives)[1])

                # shape:（batch,num_samples）
                score_at_k = tf.gather_nd(topk_prob_for_bg,
                                          tf.stack([tf.range(tf.shape(features)[0]), batch_n_neg_select - 1], axis=-1))
                selected_neg_mask = prob_for_negtives >= tf.expand_dims(score_at_k, axis=-1)

                # include both selected negtive and all positive examples
                final_mask = tf.stop_gradient(
                    tf.logical_or(tf.reshape(tf.logical_and(batch_negtive_mask, selected_neg_mask), [-1]),
                                  positive_mask))
                total_examples = tf.count_nonzero(final_mask)

                # tensors used in loss calculation
                cls_pred_after_hard_neg_mining = tf.boolean_mask(cls_pred, final_mask)
                location_pred = tf.boolean_mask(location_pred, tf.stop_gradient(positive_mask))
                flaten_cls_targets = tf.boolean_mask(tf.clip_by_value(flaten_cls_targets, 0, params['num_classes']),
                                                     final_mask)
                flaten_loc_targets = tf.stop_gradient(tf.boolean_mask(flaten_loc_targets, positive_mask))


                cls_accuracy = tf.metrics.accuracy(flaten_cls_targets, tf.argmax(cls_pred_after_hard_neg_mining,axis=-1))
                metrics = {'cls_accuracy': cls_accuracy}

                # Create a tensor named train_accuracy for logging purposes.
                tf.identity(cls_accuracy[1], name='cls_accuracy')
                tf.summary.scalar('cls_accuracy', cls_accuracy[1])

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    # cross_entropy = tf.cond(n_positives > 0, lambda: tf.losses.sparse_softmax_cross_entropy(labels=flaten_cls_targets, logits=cls_pred), lambda: 0.)# * (params['negative_ratio'] + 1.)
    # flaten_cls_targets=tf.Print(flaten_cls_targets, [flaten_loc_targets],summarize=50000)
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=flaten_cls_targets, logits=cls_pred_after_hard_neg_mining) * (
                params['negative_ratio'] + 1.)

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy_loss')
    tf.summary.scalar('cross_entropy_loss', cross_entropy)

    # loc_loss = tf.cond(n_positives > 0, lambda: modified_smooth_l1(location_pred, tf.stop_gradient(flaten_loc_targets), sigma=1.), lambda: tf.zeros_like(location_pred))
    loc_loss = modified_smooth_l1(location_pred, flaten_loc_targets, sigma=1.)

    # loc_loss = modified_smooth_l1(location_pred, tf.stop_gradient(gtargets))
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

    # construct total loss
    total_loss = tf.add(cross_entropy + loc_loss,l2_loss, name='total_loss')

    if mode in (tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT):
        # add metrics
        # visualization
        # execute none maximum suppression
        post_process_for_signle_example = lambda _cls_pred, _bboxes_pred: per_image_post_process(_cls_pred, _bboxes_pred,
                                                          params['num_classes'], params['select_threshold'],
                                                          params['min_size'],
                                                          params['keep_topk'], params['nms_topk'], params['nms_threshold'])

        cls_pred_list, bboxes_pred_list = tf.unstack(tf.reshape(cls_pred,[tf.shape(features)[0],-1,params['num_classes']])), \
                                          tf.unstack(tf.reshape(bboxes_pred,[tf.shape(features)[0],-1,4]))

        detection_boxes_list, detection_scores_list,detection_classes_list,num_detections_list= [],[],[],[]
        max_detction_num = 100
        for cls_pred, bboxes_pred in zip(cls_pred_list,bboxes_pred_list):
            # post_process func only proceess one image once a time
            detection_boxes, detection_scores,detection_classes,\
                         num_detections = post_process_for_signle_example(cls_pred, bboxes_pred)

            # padding along num_detections dimension
            detection_boxes = pad_or_clip_nd(detection_boxes,[max_detction_num,4])
            detection_scores = pad_or_clip_nd(detection_scores,[max_detction_num])
            detection_classes= pad_or_clip_nd(detection_classes,[max_detction_num])

            detection_boxes_list.append(detection_boxes)
            detection_scores_list.append(detection_scores)
            detection_classes_list.append(detection_classes)
            num_detections_list.append(num_detections)

        # batched detections
        detection_boxes = tf.stack(detection_boxes_list,axis=0)
        detection_scores = tf.stack(detection_scores_list,axis=0)
        detection_classes = tf.stack(detection_classes_list,axis=0)
        num_detections = tf.stack(num_detections_list,axis=0)

        tf.identity(num_detections[0], name="num_detections_after_nms")

        eval_dict = {
            'original_image_spatial_shape': original_image_spatial_shape,
            'true_image_shape':true_image_shape,
            'original_image':labels['original_image'],
            # goundtruths
            'num_groundtruth_boxes_per_image': num_groundtruth_boxes,
            'groundtruth_boxes': groundtruth_boxes,
            'groundtruth_classes': groundtruth_classes,
            # detections
            'detection_boxes': detection_boxes,
            'detection_scores': detection_scores,
            'detection_classes': detection_classes,
            "num_det_boxes_per_image": num_detections,
            # image id
            'key': labels["key"]
        }
        category_index = {id: {"id": id, "name": name} for (id, name) in VOC_LABELS.values()}
        categories = category_index.values()
        # visualize detected boxes
        eval_metric_op_vis = visualization_utils.VisualizeSingleFrameDetections(
            category_index,
            max_examples_to_draw=params['num_visualizations'],
            max_boxes_to_draw=params['max_num_boxes_to_visualize'],
            min_score_thresh=params['min_score_threshold'],
            use_normalized_coordinates=False)
        vis_metric_ops = eval_metric_op_vis.get_estimator_eval_metric_ops(eval_dict)
        metrics.update(vis_metric_ops)

        # dataset metrics ops
        evaluator = eval_util.get_evaluators(categories,eval_metric_fn_key=params["eval_metric_fn_key"])
        eval_metric_ops = evaluator.get_estimator_eval_metric_ops(eval_dict)
        metrics.update(eval_metric_ops)

    if mode == tf.estimator.ModeKeys.TRAIN:
        # distillate knowledge
        if params["distillation"] == True:
            with tf.variable_scope('distillation',values=[features]):
                dist_backbone = ssd_net.VGG16Backbone(params['data_format'])
                dist_feature_layers = dist_backbone.forward(features, training=(mode == tf.estimator.ModeKeys.TRAIN))
                dist_location_pred, dist_cls_pred = ssd_net.multibox_head(dist_feature_layers, params['num_classes'],
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
                cls_logits = cls_pred[:,:params['cached_classes']+1] - tf.reduce_mean(cls_pred[:,:params['cached_classes']+1],axis=1,keep_dims=True)
                cls_distillated_logits = dist_cls_pred[:,:params['cached_classes']+1] - tf.reduce_mean(dist_cls_pred[:,:params['cached_classes']+1],axis=1,keep_dims=True)
                class_distillation_loss = tf.reduce_mean(tf.square(cls_logits-cls_distillated_logits),name='class_distillation_loss')
                class_distillation_loss *= params.get('class_distillation_loss_coef',1)
                tf.summary.scalar('class_distillation_loss', class_distillation_loss)

                # 只惩罚那些negative的bboxes?
                bboxes_distillation_loss = tf.reduce_mean(tf.reduce_mean(tf.square(location_pred-dist_location_pred),axis=1,keep_dims=True),name='bboxes_distillation_loss')
                bboxes_distillation_loss *= params.get('bbox_distillation_loss_coef',1)
                tf.summary.scalar('bboxes_distillation_loss', bboxes_distillation_loss)

                total_loss =tf.add_n(total_loss,class_distillation_loss,bboxes_distillation_loss,name="total_loss_with_distillation")
                tf.summary.scalar('total_loss_with_distillation',total_loss)

        global_step = tf.train.get_or_create_global_step()
        # dynamic learning rate
        lr_values = [params['learning_rate'] * decay for decay in params['lr_decay_factors']]
        learning_rate = tf.train.piecewise_constant(tf.cast(global_step, tf.int32),
                                                    [int(_) for _ in params['decay_boundaries']],
                                                    lr_values)
        # execute truncated_learning_rate
        truncated_learning_rate = tf.maximum(learning_rate,
                                             tf.constant(params['end_learning_rate'], dtype=learning_rate.dtype),
                                             name='learning_rate')
        # Create a tensor named learning_rate for logging purposes.
        tf.summary.scalar('learning_rate', truncated_learning_rate)

        optimizer = tf.train.MomentumOptimizer(learning_rate=truncated_learning_rate,
                                               momentum=params['momentum'])
        # optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

        # Batch norm requires update_ops to be added as a train_op dependency.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(total_loss, global_step)
    else:
        train_op = None

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=None,
        loss=total_loss,
        train_op=train_op,
        eval_metric_ops=metrics,
        scaffold=tf.train.Scaffold(init_fn=None))