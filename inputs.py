from preprocessing import ssd_preprocessing
from utils import anchor_manipulator
from dataset.dataset_helper import build_dataset

import tensorflow as tf

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
                            num_readers=num_readers)
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
    dataset = input_fn(file_pattern='/home/dxfang/dataset/tfrecords/pascal_voc/eval*',is_training=True,batch_size=16)
    iterator = dataset.make_one_shot_iterator()
    with tf.Session() as sess:
        iter = iterator.get_next()
        features,labels = sess.run(iter)
        print(labels['key'])
