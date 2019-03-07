import tensorflow as tf

def select_bboxes(scores_pred, bboxes_pred, num_classes, select_threshold):
    selected_bboxes = {}
    selected_scores = {}
    with tf.name_scope('select_bboxes', [scores_pred, bboxes_pred]):
        for class_ind in range(1, num_classes):
            # ************************1.找到每个类别对应的边框
            class_scores = scores_pred[:, class_ind]
            select_mask = class_scores > select_threshold
            # ************************2.把不对应的边框置为0
            select_mask = tf.cast(select_mask, tf.float32)
            # 存储每个类别可能的边框
            selected_bboxes[class_ind] = tf.multiply(bboxes_pred, tf.expand_dims(select_mask, axis=-1))
            # 存储每个类别，每个边框得到的分数
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

def per_image_post_process(cls_pred, bboxes_pred, num_classes, select_threshold, min_size, keep_topk, nms_topk, nms_threshold):
    with tf.name_scope('select_bboxes', [cls_pred, bboxes_pred]):
        # calculate probability
        scores_pred = tf.nn.softmax(cls_pred)
        # organize boxes by classs, return a dict: {"class_id":suitable_bboxes_tensor}
        selected_bboxes, selected_scores = select_bboxes(scores_pred, bboxes_pred, num_classes, select_threshold)

        per_image_detection_boxes = []
        per_image_detection_classes = []
        per_image_detection_scores = []
        per_image_total_detections = 0

        for class_ind in range(1, num_classes):
            ymin, xmin, ymax, xmax = tf.unstack(selected_bboxes[class_ind], 4, axis=-1)
            #ymin, xmin, ymax, xmax = tf.split(selected_bboxes[class_ind], 4, axis=-1)
            #ymin, xmin, ymax, xmax = tf.squeeze(ymin), tf.squeeze(xmin), tf.squeeze(ymax), tf.squeeze(xmax)

            # predicted boxes may be invalid
            ymin, xmin, ymax, xmax = clip_bboxes(ymin, xmin, ymax, xmax, 'clip_bboxes_{}'.format(class_ind))

            # filter boxes with too small size
            ymin, xmin, ymax, xmax, selected_scores[class_ind] = filter_bboxes(selected_scores[class_ind],
                                                ymin, xmin, ymax, xmax, min_size, 'filter_bboxes_{}'.format(class_ind))

            # sort bboxes to choose candidate boxes for nms()
            ymin, xmin, ymax, xmax, selected_scores[class_ind] = sort_bboxes(selected_scores[class_ind],
                                                ymin, xmin, ymax, xmax, keep_topk, 'sort_bboxes_{}'.format(class_ind))

            selected_bboxes[class_ind] = tf.stack([ymin, xmin, ymax, xmax], axis=-1)

            # execute nms to get the final boxes(  max num detections for each class)
            detection_score, detection_boxes,num_detection= nms_bboxes(selected_scores[class_ind], selected_bboxes[class_ind], nms_topk, nms_threshold, 'nms_bboxes_{}'.format(class_ind))

            # collect all detections from one image
            per_image_detection_boxes.append(detection_boxes)
            per_image_detection_scores.append(detection_score)
            per_image_detection_classes.append(tf.zeros_like(detection_score)+class_ind)
            per_image_total_detections += num_detection

        per_image_detection_boxes = tf.concat(per_image_detection_boxes,axis=0)
        per_image_detection_scores = tf.concat(per_image_detection_scores,axis=0)
        per_image_detection_classes = tf.concat(per_image_detection_classes,axis=0)

        return per_image_detection_boxes, per_image_detection_scores,per_image_detection_classes,per_image_total_detections

