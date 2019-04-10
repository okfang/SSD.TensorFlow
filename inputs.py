import functools

from preprocessing import ssd_preprocessing
from utils import anchor_manipulator, visualization_utils

import tensorflow as tf
# tf.enable_eager_execution()

slim = tf.contrib.slim

from utils.shape_util import unpad_tensor, pad_or_clip_nd

train_image_size = 300
global_anchor_info = dict()



def read_dataset(file_read_func, file_pattern,is_training=True, num_readers=2):
    """
    读取tfrecord文件，并构造dataset
    """
    filenames = tf.gfile.Glob(file_pattern)
    # print("------------check files-------------")
    # for filename in filenames:
    #     print(filename)
    filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if num_readers > 1:
        tf.logging.warning('`shuffle` is false, but the input data stream is '
                           'still slightly shuffled since `num_readers` > 1.')
    if is_training:
        filename_dataset = filename_dataset.repeat()
    records_dataset = filename_dataset.apply(
        tf.contrib.data.parallel_interleave(
            file_read_func,
            cycle_length=num_readers))
    return records_dataset


def build_dataset(class_list=None,file_pattern=None,is_training=True, batch_size=32,image_preprocessing_fn=None,num_readers=2,data_format='channels_first'):

    # Features in Pascal VOC TFRecords.
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/key/sha256': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64),
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'key': slim.tfexample_decoder.Tensor('image/key/sha256'),
        'filename': slim.tfexample_decoder.Tensor('image/filename'),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
            ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
        'object/difficult': slim.tfexample_decoder.Tensor('image/object/bbox/difficult'),
        'object/truncated': slim.tfexample_decoder.Tensor('image/object/bbox/truncated'),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    def decode_example(serialized_example):
        # slim decode tf example
        keys = decoder.list_items()
        tensors = decoder.decode(serialized_example, items=keys)
        tensor_dict = dict(zip(keys, tensors))
        original_image = tensor_dict['image']
        original_shape = tensor_dict['shape']
        filename = tensor_dict['filename']
        glabels_raw = tensor_dict['object/label']
        gbboxes_raw = tensor_dict['object/bbox']
        isdifficult = tensor_dict['object/difficult']
        key = tensor_dict['key']

        # # filter class
        if class_list:
            valid_class_mask = glabels_raw <= 10
            glabels_raw = tf.boolean_mask(glabels_raw, valid_class_mask)
            gbboxes_raw = tf.boolean_mask(gbboxes_raw, valid_class_mask)
            isdifficult = tf.boolean_mask(isdifficult, valid_class_mask)

        return (original_image,original_shape,filename,glabels_raw,gbboxes_raw,isdifficult,key)

    def filter_fn(original_image, original_shape, filename, glabels_raw, gbboxes_raw, isdifficult, key):
        return tf.not_equal(tf.count_nonzero(glabels_raw <= 10),0)

    def process_fn(original_image,original_shape,filename,glabels_raw,gbboxes_raw,isdifficult,key):
        # filter difficult example
        if is_training:
            # if all is difficult, then keep the first one
            isdifficult_mask = tf.cond(tf.count_nonzero(isdifficult, dtype=tf.int32) < tf.shape(isdifficult)[0],
                                       lambda: isdifficult < tf.ones_like(isdifficult),
                                       lambda: tf.one_hot(0, tf.shape(isdifficult)[0], on_value=True, off_value=False,
                                                          dtype=tf.bool))
            glabels_raw = tf.boolean_mask(glabels_raw, isdifficult_mask)
            gbboxes_raw = tf.boolean_mask(gbboxes_raw, isdifficult_mask)

        # Pre-processing image, labels and bboxes.
        if is_training:
            preprocessed_image, groundtruth_classes, groundtruth_boxes, true_image_shape = image_preprocessing_fn(original_image, glabels_raw, gbboxes_raw)
        else:
            preprocessed_image, true_image_shape = image_preprocessing_fn(original_image, glabels_raw, gbboxes_raw)
            groundtruth_classes, groundtruth_boxes = glabels_raw, gbboxes_raw

        max_num_bboxes = 50
        num_groundtruth_boxes = tf.minimum(tf.shape(groundtruth_boxes)[0],max_num_bboxes)


        # padding in num_bboxes dimension
        groundtruth_classes = pad_or_clip_nd(groundtruth_classes,output_shape = [max_num_bboxes])
        groundtruth_boxes = pad_or_clip_nd(groundtruth_boxes,output_shape = [max_num_bboxes,4])

        # [2]
        original_image_spatial_shape = tf.cast(original_shape[:2],dtype=tf.int32)

        if data_format == 'channels_first':
            preprocessed_image = tf.transpose(preprocessed_image, perm=(2, 0, 1))

        # [300,300,]
        features = {
            'preprocessed_image':preprocessed_image,
            'filename': filename,
            'original_shape': original_shape,
            'original_image_spatial_shape': original_image_spatial_shape,
            'true_image_shape': true_image_shape,
            'key': key
        }
        labels = {
                  'num_groundtruth_boxes': num_groundtruth_boxes,
                  'groundtruth_boxes': groundtruth_boxes,
                  'groundtruth_classes': groundtruth_classes,

                  }

        # if not is_training:
        #     features['original_image'] = pad_or_clip_nd(original_image,[2000,2000,3])

        return features, labels

    # 读取dataset
    dataset = read_dataset(functools.partial(tf.data.TFRecordDataset, buffer_size=8 * 1000 * 1000),
                           file_pattern,
                           is_training=is_training,
                           num_readers=num_readers)

    # 预处理,获取数据
    dataset = dataset.map(decode_example,num_parallel_calls=batch_size)
    if class_list:
        dataset = dataset.filter(filter_fn)
    dataset = dataset.map(process_fn,num_parallel_calls=batch_size)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=None)
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
        dataset = build_dataset(class_list=class_list,
                                file_pattern=file_pattern,
                                is_training=is_training,
                                batch_size=batch_size,
                                image_preprocessing_fn=image_preprocessing_fn,
                                num_readers = num_readers,
                                data_format=data_format)
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
    train_input_pattern = '/home/dxfang/dataset/tfrecords/pascal_voc/train-000*'
    eval_input_pattern = '/home/dxfang/dataset/tfrecords/pascal_voc/val-000*'
    taskA_class_list = list(range(1, 11))
    taskB_class_list = list(range(11, 21))
    dataset = input_pipeline(class_list=None,file_pattern=eval_input_pattern, is_training=False, batch_size=16)()
    iterator = dataset.make_one_shot_iterator()
    with tf.Session() as sess:
        features, labels = iterator.get_next()
        #
        num_groundtruth_boxes = labels['num_groundtruth_boxes']
        groundtruth_classes = labels['groundtruth_classes']
        groundtruth_boxes = labels['groundtruth_boxes']
        original_image_spatial_shape = labels['original_image_spatial_shape']
        true_image_shape = labels['true_image_shape']
        print(sess.run(labels['filename']))
        # print(groundtruth_classes)
        # # unpadding  num_boxes dimension for real num_groundtruth_boxes
        # groundtruth_classes = unpad_tensor(groundtruth_classes, num_groundtruth_boxes)
        # groundtruth_boxes = unpad_tensor(groundtruth_boxes, num_groundtruth_boxes)
        #
        # groundtruth_classes_, groundtruth_boxes_ = sess.run([groundtruth_classes,groundtruth_boxes])
        # print(groundtruth_classes_)



