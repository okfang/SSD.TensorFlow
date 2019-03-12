from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf

from utils.shape_util import pad_or_clip_nd

slim = tf.contrib.slim


def read_dataset(file_read_func, file_pattern,is_training=True,num_readers=2):
    """
    读取tfrecord文件，并构造dataset
    """
    filenames = tf.gfile.Glob(file_pattern)
    # 答应文件是否齐全
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


def build_dataset(class_list=None,file_pattern=None,is_training=True, batch_size=32,image_preprocessing_fn=None,num_readers=2):

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

    def process_fn(serialized_example):
        # slim decode tf example
        keys = decoder.list_items()
        tensors = decoder.decode(serialized_example, items=keys)
        tensor_dict = dict(zip(keys, tensors))
        original_image = tensor_dict['image']
        filename = tensor_dict['filename']
        glabels_raw = tensor_dict['object/label']
        gbboxes_raw = tensor_dict['object/bbox']
        isdifficult = tensor_dict['object/difficult']
        key = tensor_dict['key']

        # filter class
        if class_list != None:
            valid_class_mask = tf.map_fn(lambda label: label in class_list, glabels_raw)
            glabels_raw = tf.boolean_mask(glabels_raw,valid_class_mask)
            gbboxes_raw = tf.boolean_mask(gbboxes_raw,valid_class_mask)
            isdifficult = tf.boolean_mask(isdifficult,valid_class_mask)
        if tf.shape(glabels_raw) == 0:
            return None

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
            preprocessed_image, groundtruth_classes, groundtruth_boxes = image_preprocessing_fn(original_image, glabels_raw, gbboxes_raw)
        else:
            image_before_normalization, preprocessed_image = image_preprocessing_fn(original_image, glabels_raw, gbboxes_raw)
            groundtruth_classes, groundtruth_boxes = glabels_raw, gbboxes_raw

        num_groundtruth_boxes = tf.shape(groundtruth_boxes)[0]
        max_num_bboxes = 50

        # padding in num_bboxes dimension
        groundtruth_classes = pad_or_clip_nd(groundtruth_classes,output_shape = [max_num_bboxes])
        groundtruth_boxes = pad_or_clip_nd(groundtruth_boxes,output_shape = [max_num_bboxes,4])
        # [3]
        true_image_shape = tf.cast(tf.shape(preprocessed_image),dtype=tf.int32)

        # [2]
        original_image_spatial_shape = tf.cast(tensor_dict['shape'][:2],dtype=tf.int32)
        # [300,300,]
        features = preprocessed_image

        labels = {'original_image_spatial_shape': original_image_spatial_shape,
                  'true_image_shape': true_image_shape,
                  'num_groundtruth_boxes': num_groundtruth_boxes,
                  'groundtruth_boxes': groundtruth_boxes,
                  'groundtruth_classes': groundtruth_classes,
                  'key': key}

        if not is_training:
            labels['original_image'] = image_before_normalization

        return features, labels

    # 读取dataset
    dataset = read_dataset(functools.partial(tf.data.TFRecordDataset, buffer_size=8 * 1000 * 1000),
                           file_pattern,
                           is_training=is_training,
                           num_readers=num_readers)

    # 预处理
    dataset = dataset.map(process_fn,num_parallel_calls=batch_size)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=None)
    return dataset
