import tensorflow as tf

def select_bboxes(scores_pred, bboxes_pred, class_list, select_threshold):
    """
    :return:
    selected_bboxes = {1:[num_all_anchors,4],2:[]}
    selected_scores = {1:[num_all_anchors,]}
    """
    selected_bboxes = {}
    selected_scores = {}
    with tf.name_scope('select_bboxes', [scores_pred, bboxes_pred]):
        for class_ind in class_list:
            # class_scores ->shape: [num_all_anchors,]
            class_scores = scores_pred[:, class_ind]
            select_mask = class_scores > select_threshold
            select_mask = tf.cast(select_mask, tf.float32)
            # choose anchors: unmatched anchors will be set 0
            selected_bboxes[class_ind] = tf.multiply(bboxes_pred, tf.expand_dims(select_mask, axis=-1))
            selected_scores[class_ind] = tf.multiply(class_scores, select_mask)

    return selected_bboxes, selected_scores

def clip_bboxes(ymin, xmin, ymax, xmax, name):
    with tf.name_scope(name, 'clip_bboxes', [ymin, xmin, ymax, xmax]):
        ymin = tf.maximum(ymin, 0.)
        xmin = tf.maximum(xmin, 0.)
        ymax = tf.minimum(ymax, 1.)
        xmax = tf.minimum(xmax, 1.)

        ymin = tf.minimum(ymin, ymax)
        xmin = tf.minimum(xmin, xmax)

        return ymin, xmin, ymax, xmax

def filter_bboxes(scores_pred, ymin, xmin, ymax, xmax, min_size, name):
    with tf.name_scope(name, 'filter_bboxes', [scores_pred, ymin, xmin, ymax, xmax]):
        width = xmax - xmin
        height = ymax - ymin
        filter_mask = tf.logical_and(width > min_size, height > min_size)
        filter_mask = tf.cast(filter_mask, tf.float32)
        return tf.multiply(ymin, filter_mask), tf.multiply(xmin, filter_mask), \
                tf.multiply(ymax, filter_mask), tf.multiply(xmax, filter_mask), tf.multiply(scores_pred, filter_mask)

def sort_bboxes(scores_pred, ymin, xmin, ymax, xmax, keep_topk, name):
    with tf.name_scope(name, 'sort_bboxes', [scores_pred, ymin, xmin, ymax, xmax]):
        cur_bboxes = tf.shape(scores_pred)[0]
        scores, idxes = tf.nn.top_k(scores_pred, k=tf.minimum(keep_topk, cur_bboxes), sorted=True)

        ymin, xmin, ymax, xmax = tf.gather(ymin, idxes), tf.gather(xmin, idxes), tf.gather(ymax, idxes), tf.gather(xmax, idxes)

        paddings_scores = tf.expand_dims(tf.stack([0, tf.maximum(keep_topk-cur_bboxes, 0)], axis=0), axis=0)

        return tf.pad(ymin, paddings_scores, "CONSTANT"), tf.pad(xmin, paddings_scores, "CONSTANT"),\
                tf.pad(ymax, paddings_scores, "CONSTANT"), tf.pad(xmax, paddings_scores, "CONSTANT"),\
                tf.pad(scores, paddings_scores, "CONSTANT")

def nms_bboxes(scores_pred, bboxes_pred, nms_topk, nms_threshold, name):
    with tf.name_scope(name, 'nms_bboxes', [scores_pred, bboxes_pred]):
        idxes = tf.image.non_max_suppression(bboxes_pred, scores_pred, nms_topk, nms_threshold)
        num_detections = tf.shape(idxes)[0]
        return tf.gather(scores_pred, idxes), tf.gather(bboxes_pred, idxes),num_detections

def parse_by_class(cls_pred, bboxes_pred, num_classes, select_threshold, min_size, keep_topk, nms_topk, nms_threshold):
    with tf.name_scope('select_bboxes', [cls_pred, bboxes_pred]):
        scores_pred = tf.nn.softmax(cls_pred)
        selected_bboxes, selected_scores = select_bboxes(scores_pred, bboxes_pred, num_classes, select_threshold)
        for class_ind in range(1, num_classes):
            ymin, xmin, ymax, xmax = tf.unstack(selected_bboxes[class_ind], 4, axis=-1)
            #ymin, xmin, ymax, xmax = tf.split(selected_bboxes[class_ind], 4, axis=-1)
            #ymin, xmin, ymax, xmax = tf.squeeze(ymin), tf.squeeze(xmin), tf.squeeze(ymax), tf.squeeze(xmax)
            ymin, xmin, ymax, xmax = clip_bboxes(ymin, xmin, ymax, xmax, 'clip_bboxes_{}'.format(class_ind))
            ymin, xmin, ymax, xmax, selected_scores[class_ind] = filter_bboxes(selected_scores[class_ind],
                                                ymin, xmin, ymax, xmax, min_size, 'filter_bboxes_{}'.format(class_ind))
            ymin, xmin, ymax, xmax, selected_scores[class_ind] = sort_bboxes(selected_scores[class_ind],
                                                ymin, xmin, ymax, xmax, keep_topk, 'sort_bboxes_{}'.format(class_ind))
            selected_bboxes[class_ind] = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
            selected_scores[class_ind], selected_bboxes[class_ind],num_detections = nms_bboxes(selected_scores[class_ind], selected_bboxes[class_ind], nms_topk, nms_threshold, 'nms_bboxes_{}'.format(class_ind))

        return selected_bboxes, selected_scores

def per_image_post_process(cls_pred, bboxes_pred,mode,select_threshold=None, min_size=None, keep_topk=None, nms_topk=None, nms_threshold=None,is_tack_A=False,is_task_B=False,num_classes=None):
    """
    select boxes per image
    :param cls_pred:  [num_all_anchors,21]
    :param bboxes_pred: [num_all_anchors,4]
    :param num_classes:  21
    :param select_threshold:
    :param min_size:
    :param keep_topk:
    :param nms_topk:
    :param nms_threshold:
    :return:
        per_image_detection_boxes :  [num_selected_boxes,4]
        per_image_detection_scores: [num_selected_boxes]
        per_image_detection_classes: [num_selected_boxes]
        per_image_total_detections: [num_selected_boxes]
    """
    with tf.name_scope('select_bboxes', [cls_pred, bboxes_pred]):
        class_list = list(range(1,num_classes))
        if is_tack_A:
            # cls_pred = cls_pred[:,:11]
            class_list = list(range(0,11))
        if is_task_B:
            # cls_pred = tf.gather(cls_pred, [0] + list(range(11, 21)), axis=1)
            class_list = list(range(11, 21))

        # calculate probability
        scores_pred = tf.nn.softmax(cls_pred)
        # selected_bboxes ->shape: {1:[num_all_anchors,4],2:[],...}
        # selected_scores ->shape: {1:[num_all_anchors,],2:[],....}
        selected_bboxes, selected_scores = select_bboxes(scores_pred, bboxes_pred, class_list, select_threshold)

        per_image_detection_boxes = []
        per_image_detection_classes = []
        per_image_detection_scores = []
        each_classes_detections = []
        for class_ind in class_list:
            ymin, xmin, ymax, xmax = tf.unstack(selected_bboxes[class_ind], 4, axis=-1)
            # predicted boxes may be invalid
            ymin, xmin, ymax, xmax = clip_bboxes(ymin, xmin, ymax, xmax, 'clip_bboxes_{}'.format(class_ind))
            # filter boxes with too small size
            ymin, xmin, ymax, xmax, selected_scores[class_ind] = filter_bboxes(selected_scores[class_ind],
                                                ymin, xmin, ymax, xmax, min_size, 'filter_bboxes_{}'.format(class_ind))

            # sort bboxes to choose candidate boxes
            # ymin ->shape: [num_all_anchors, keep_topk]   have top_k score
            ymin, xmin, ymax, xmax, selected_scores[class_ind] = sort_bboxes(selected_scores[class_ind],
                                                ymin, xmin, ymax, xmax, keep_topk, 'sort_bboxes_{}'.format(class_ind))
            selected_bboxes[class_ind] = tf.stack([ymin, xmin, ymax, xmax], axis=-1)

            # execute nms to get the final boxes(  max num detections for each class)
            detection_score, detection_boxes,num_detections = nms_bboxes(selected_scores[class_ind], selected_bboxes[class_ind], nms_topk, nms_threshold, 'nms_bboxes_{}'.format(class_ind))
            if mode==tf.estimator.ModeKeys.PREDICT:
                selected_scores[class_ind] = detection_score
                selected_bboxes[class_ind] = detection_boxes

            # collect all detections from one image
            per_image_detection_boxes.append(detection_boxes)
            per_image_detection_scores.append(detection_score)
            per_image_detection_classes.append(tf.zeros_like(detection_score,dtype=tf.int32)+class_ind)
            each_classes_detections.append(num_detections)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return selected_bboxes, selected_scores
        # concat all classes
        per_image_detection_boxes = tf.concat(per_image_detection_boxes,axis=0)
        per_image_detection_scores = tf.concat(per_image_detection_scores,axis=0)
        per_image_detection_classes = tf.concat(per_image_detection_classes,axis=0)

        # sort all detections
        # per_image_detection_scores, ind = tf.nn.top_k(per_image_detection_scores,
        #                                               k=tf.shape(per_image_detection_scores)[0])
        # per_image_detection_boxes = tf.gather(per_image_detection_boxes, ind)
        # per_image_detection_classes = tf.gather(per_image_detection_classes, ind)

        per_image_total_detections = tf.shape(per_image_detection_scores)[0]

        return per_image_detection_boxes, per_image_detection_scores,per_image_detection_classes, per_image_total_detections

