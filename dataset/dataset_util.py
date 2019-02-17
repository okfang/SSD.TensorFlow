from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf

slim = tf.contrib.slim

"""

"""

def read_dataset(file_read_func, file_pattern,num_readers=4):
    """
    读取tfrecord文件，并构造dataset
    """
    filenames = tf.gfile.Glob(file_pattern)
    # 答应文件是否齐全
    print("------------check files-------------")
    for filename in filenames:
        print(filename)
    filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if num_readers > 1:
        tf.logging.warning('`shuffle` is false, but the input data stream is '
                           'still slightly shuffled since `num_readers` > 1.')
    filename_dataset = filename_dataset.repeat()
    records_dataset = filename_dataset.apply(
        tf.contrib.data.parallel_interleave(
            file_read_func,
            cycle_length=num_readers))
    return records_dataset


def get_dataset(file_pattern=None,is_training=True, batch_size=32,image_preprocessing_fn=None, anchor_encoder_fn=None):
    """
    使用原生的dataset api
    得到用于estimator输入的dataset
    :param is_training:
    :param image_preprocessing_fn: 数据预处理
    :param anchor_encoder_fn: 生成对应的anchor
    :return:dataset
    """
    # Features in Pascal VOC TFRecords.
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
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
        """
        1.解码tf_example
        2.数据预处理
        :param value: tf record
        :return: features  and  labels
        """""
        # slim decode tf example
        keys = decoder.list_items()
        # print("************************************打印serialized_example",serialized_example.graph)
        tensors = decoder.decode(serialized_example, items=keys)
        tensor_dict = dict(zip(keys, tensors))
        org_image = tensor_dict['image']
        filename = tensor_dict['filename']
        shape = tensor_dict['shape']
        glabels_raw = tensor_dict['object/label']
        gbboxes_raw = tensor_dict['object/bbox']
        isdifficult = tensor_dict['object/difficult']

        # print("*****************************打印org_image:",org_image.graph)

        # preprocessing image
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
            image, glabels, gbboxes = image_preprocessing_fn(org_image, glabels_raw, gbboxes_raw)
            # print("*****************************打印image",image.graph)
            # print("*****************************打印glabels",glabels.graph)
            # print("*****************************打印gbboxes",gbboxes.graph)
        else:
            image = image_preprocessing_fn(org_image, glabels_raw, gbboxes_raw)
            glabels, gbboxes = glabels_raw, gbboxes_raw

        features = image
        labels = {'shape': shape, 'gbboxes': gbboxes, 'glabels': glabels}
        return (features,labels)
    # 读取dataset
    dataset = read_dataset(
        functools.partial(tf.data.TFRecordDataset, buffer_size=8 * 1000 * 1000),file_pattern)
    # 预处理
    dataset = dataset.map(process_fn,num_parallel_calls=batch_size*2)
    print("----------------------",dataset)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=None)

    return dataset
