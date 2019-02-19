from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf

slim = tf.contrib.slim

"""

"""
def pad_or_clip_nd(tensor, output_shape):
  """Pad or Clip given tensor to the output shape.

  Args:
    tensor: Input tensor to pad or clip.
    output_shape: A list of integers / scalar tensors (or None for dynamic dim)
      representing the size to pad or clip each dimension of the input tensor.

  Returns:
    Input tensor padded and clipped to the output shape.
  """
  tensor_shape = tf.shape(tensor)
  clip_size = [
      tf.where(tensor_shape[i] - shape > 0, shape, -1)
      if shape is not None else -1 for i, shape in enumerate(output_shape)
  ]
  clipped_tensor = tf.slice(
      tensor,
      begin=tf.zeros(len(clip_size), dtype=tf.int32),
      size=clip_size)

  # Pad tensor if the shape of clipped tensor is smaller than the expected
  # shape.
  clipped_tensor_shape = tf.shape(clipped_tensor)
  trailing_paddings = [
      shape - clipped_tensor_shape[i] if shape is not None else 0
      for i, shape in enumerate(output_shape)
  ]
  paddings = tf.stack(
      [
          tf.zeros(len(trailing_paddings), dtype=tf.int32),
          trailing_paddings
      ],
      axis=1)
  padded_tensor = tf.pad(clipped_tensor, paddings=paddings)
  output_static_shape = [
      dim if not isinstance(dim, tf.Tensor) else None for dim in output_shape
  ]
  padded_tensor.set_shape(output_static_shape)
  return padded_tensor

def read_dataset(file_read_func, file_pattern,num_readers=4):
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
        tensors = decoder.decode(serialized_example, items=keys)
        tensor_dict = dict(zip(keys, tensors))
        org_image = tensor_dict['image']
        filename = tensor_dict['filename']
        original_shape = tensor_dict['shape']
        glabels_raw = tensor_dict['object/label']
        gbboxes_raw = tensor_dict['object/bbox']
        isdifficult = tensor_dict['object/difficult']

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
        else:
            image = image_preprocessing_fn(org_image, glabels_raw, gbboxes_raw)
            glabels, gbboxes = glabels_raw, gbboxes_raw

        # padding in num_bboxes dimension 防止batch内样本形状不一
        num_groundtruth_boxes = tf.shape(gbboxes)[0]
        max_num_bboxes = 50
        glabels = pad_or_clip_nd(glabels,[max_num_bboxes])
        gbboxes = pad_or_clip_nd(gbboxes,[max_num_bboxes,4])
        true_shape = tf.shape(image)

        features = image
        labels = {'original_shape':original_shape,'true_shape':true_shape ,'num_groundtruth_boxes':num_groundtruth_boxes, 'gbboxes': gbboxes, 'glabels': glabels}
        if not is_training:
            labels['original_image'] =org_image
        return (features,labels)

    # 读取dataset
    dataset = read_dataset(
        functools.partial(tf.data.TFRecordDataset, buffer_size=8 * 1000 * 1000),file_pattern)
    # 预处理
    dataset = dataset.map(process_fn,num_parallel_calls=batch_size*2)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=None)

    return dataset
