from preprocessing import ssd_preprocessing
from utils import anchor_manipulator, visualization_utils
from dataset.dataset_helper import build_dataset

import tensorflow as tf

from utils.shape_util import unpad_tensor

train_image_size = 300
global_anchor_info = dict()


def input_fn(class_list=None,file_pattern='train-*', is_training=True, batch_size=None,data_format='channels_first',num_readers=2,params=None):
    out_shape = [train_image_size] * 2
    anchor_creator = anchor_manipulator.AnchorCreator(out_shape,
                                                      layers_shapes=[(38, 38), (19, 19), (10, 10), (5, 5),
                                                                     (3, 3),
                                                                     (1, 1)],
                                                      anchor_scales=[(0.1,), (0.2,), (0.375,), (0.55,),
                                                                     (0.725,),
                                                                     (0.9,)],
                                                      extra_anchor_scales=[(0.1414,), (0.2739,), (0.4541,),
                                                                           (0.6315,), (0.8078,), (0.9836,)],
                                                      anchor_ratios=[(1., 2., .5), (1., 2., 3., .5, 0.3333),
                                                                     (1., 2., 3., .5, 0.3333),
                                                                     (1., 2., 3., .5, 0.3333), (1., 2., .5),
                                                                     (1., 2., .5)],
                                                      layer_steps=[8, 16, 32, 64, 100, 300], )
    # all_anchors shape:[num_all_defualt_anchors,]  8732= 38*38*(3*6+6)+ .....
    all_anchors, all_num_anchors_depth, all_num_anchors_spatial = anchor_creator.get_all_anchors()
    # Serialization anchors
    global global_anchor_info
    global_anchor_info["all_anchors"] = all_anchors
    global_anchor_info["all_num_anchors_depth"] = all_num_anchors_depth
    global_anchor_info["all_num_anchors_spatial"] = all_num_anchors_spatial

    image_preprocessing_fn = lambda image_, labels_, bboxes_: ssd_preprocessing.preprocess_image(image_, labels_,
                                                                                                 bboxes_, out_shape,
                                                                                                 is_training=is_training,
                                                                                                 data_format=data_format,
                                                                                                 output_rgb=False)
    dataset = build_dataset(file_pattern=file_pattern,
                            is_training=is_training,
                            batch_size=batch_size,
                            image_preprocessing_fn=image_preprocessing_fn,
                            num_readers=num_readers,data_format=data_format,)
    return dataset

def input_pipeline(class_list=None,file_pattern='train-*', is_training=True, batch_size=None,data_format='channels_first',num_readers=2):
    def input_fn(params=None):
        out_shape = [train_image_size] * 2
        anchor_creator = anchor_manipulator.AnchorCreator(out_shape,
                                                          layers_shapes=[(38, 38), (19, 19), (10, 10), (5, 5),
                                                                         (3, 3),
                                                                         (1, 1)],
                                                          anchor_scales=[(0.1,), (0.2,), (0.375,), (0.55,),
                                                                         (0.725,),
                                                                         (0.9,)],
                                                          extra_anchor_scales=[(0.1414,), (0.2739,), (0.4541,),
                                                                               (0.6315,), (0.8078,), (0.9836,)],
                                                          anchor_ratios=[(1., 2., .5), (1., 2., 3., .5, 0.3333),
                                                                         (1., 2., 3., .5, 0.3333),
                                                                         (1., 2., 3., .5, 0.3333), (1., 2., .5),
                                                                         (1., 2., .5)],
                                                          layer_steps=[8, 16, 32, 64, 100, 300],)
        # all_anchors shape:[num_all_defualt_anchors,]  8732= 38*38*(3*6+6)+ .....
        all_anchors, all_num_anchors_depth, all_num_anchors_spatial = anchor_creator.get_all_anchors()
        # Serialization anchors
        global global_anchor_info
        global_anchor_info["all_anchors"] = all_anchors
        global_anchor_info["all_num_anchors_depth"] = all_num_anchors_depth
        global_anchor_info["all_num_anchors_spatial"] = all_num_anchors_spatial

        image_preprocessing_fn = lambda image_, labels_, bboxes_: ssd_preprocessing.preprocess_image(image_, labels_,
                                                                                                     bboxes_, out_shape,
                                                                                                     is_training=is_training,
                                                                                                     data_format=data_format,
                                                                                                     output_rgb=False)
        dataset = build_dataset(file_pattern=file_pattern,
                                is_training=is_training,
                                batch_size=batch_size,
                                image_preprocessing_fn=image_preprocessing_fn,
                                num_readers = num_readers)
        return dataset

    return input_fn

if __name__ == '__main__':
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
    category_index = {VOC_LABELS[name][0]: {"id": VOC_LABELS[name][0], "name": name} for name in VOC_LABELS.keys()}
    categories = list(category_index.values())

    dataset = input_fn(file_pattern='F:\\dataset\\PASCALVOC\\tfrecords\\pascal_voc\\train*',is_training=False,batch_size=16)
    iterator = dataset.make_one_shot_iterator()
    with tf.Session() as sess:
        features, labels = iterator.get_next()



        num_groundtruth_boxes = labels['num_groundtruth_boxes']
        groundtruth_classes = labels['groundtruth_classes']
        groundtruth_boxes = labels['groundtruth_boxes']
        original_image_spatial_shape = labels['original_image_spatial_shape']
        true_image_shape = labels['true_image_shape']

        # unpadding  num_boxes dimension for real num_groundtruth_boxes
        groundtruth_classes = unpad_tensor(groundtruth_classes, num_groundtruth_boxes)
        groundtruth_boxes = unpad_tensor(groundtruth_boxes, num_groundtruth_boxes)

        groundtruth_classes_, groundtruth_boxes_ = sess.run([groundtruth_classes,groundtruth_boxes])
        print(groundtruth_classes_)
        print(groundtruth_boxes_)

        eval_input_dict = {
            'original_image_spatial_shape': original_image_spatial_shape,
            'true_image_shape': true_image_shape,
            'original_image': labels['original_image'],
            # goundtruths
            'num_groundtruth_boxes_per_image': num_groundtruth_boxes,
            'groundtruth_boxes': groundtruth_boxes,
            'groundtruth_classes': groundtruth_classes,
            # # detections
            # 'detection_boxes': detection_boxes,
            # 'detection_scores': detection_scores,
            # 'detection_classes': detection_classes,
            # "num_det_boxes_per_image": num_detections,
            # # image id
            # 'key': labels["key"]
        }
        eval_metric_op_vis = visualization_utils.VisualizeSingleFrameDetections(
            category_index,
            max_examples_to_draw=20,
            max_boxes_to_draw=20,
            min_score_thresh=None,
            use_normalized_coordinates=True)
        image_list = eval_metric_op_vis.images_from_evaluation_dict(eval_input_dict)
        original_iamge_summary = tf.summary.image("original_images",labels['original_image'])
        summary_images = {}
        for i,image in enumerate(image_list):
            summary_images[i] = tf.summary.image('ground_truth_image{}'.format(i),image)
        summary_merge_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter("F:\\tf_logs")

        summary = sess.run(summary_merge_op)

        summary_writer.add_summary(summary)
        summary_writer.close()


