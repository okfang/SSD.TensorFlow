# Copyright 2018 Changan Wang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from models import ssd_model_fn

from utils import scaffolds

# hardware related configuration
from inputs import input_pipeline

tf.app.flags.DEFINE_integer(
    'num_readers', 8,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 24,
    'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer(
    'num_cpu_threads', 0,
    'The number of cpu cores used to train.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 1., 'GPU memory fraction to use.')
# scaffold related configuration
tf.app.flags.DEFINE_string(
    'data_dir', '/home/dxfang/dataset/tfrecords/pascal_voc/',
    'The directory where the dataset input data is stored.')
tf.app.flags.DEFINE_integer(
    'num_classes', 21, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
    'model_dir', './logs/',
    'The directory where the model will be stored.')
tf.app.flags.DEFINE_integer(
    'log_every_n_steps',10,
    'The frequency with which logs are printed.')
tf.app.flags.DEFINE_integer(
    'save_summary_steps', 100,
    'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer(
    'save_checkpoints_secs', None,
    'The frequency with which the model is saved, in seconds.')
# model related configuration
tf.app.flags.DEFINE_integer(
    'train_image_size', 300,
    'The size of the input image for the model to use.')
tf.app.flags.DEFINE_integer(
    'train_epochs', None,
    'The number of epochs to use for training.')
tf.app.flags.DEFINE_integer(
    'max_number_of_steps', 120000,
    'The max number of steps to use for training.')
tf.app.flags.DEFINE_integer(
    'batch_size',16,
    'Batch size for training and evaluation.')
tf.app.flags.DEFINE_string(
    'data_format', 'channels_first',  # 'channels_first' or 'channels_last'
    'A flag to override the data format used in the model. channels_first '
    'provides a performance boost on GPU but is not always compatible '
    'with CPU. If left unspecified, the data format will be chosen '
    'automatically based on whether TensorFlow was built for CPU or GPU.')
tf.app.flags.DEFINE_float(
    'negative_ratio', 3., 'Negative ratio in the loss function.')
tf.app.flags.DEFINE_float(
    'positive_threshold', 0.5, 'Matching threshold in the loss function.')
tf.app.flags.DEFINE_float(
    'neg_threshold', 0.5, 'Matching threshold for the negtive examples in the loss function.')
# optimizer related configuration
tf.app.flags.DEFINE_integer(
    'tf_random_seed', 20180503, 'Random seed for TensorFlow initializers.')
tf.app.flags.DEFINE_float(
    'weight_decay', 5e-4, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.000001,
    'The minimal end learning rate used by a polynomial decay learning rate.')
# for learning rate piecewise_constant decay
tf.app.flags.DEFINE_string(
    'decay_boundaries', '500, 80000, 100000',
    'Learning rate decay boundaries by global_step (comma-separated list).')
tf.app.flags.DEFINE_string(
    'lr_decay_factors', '0.1, 1, 0.1, 0.01',
    'The values of learning_rate decay factor for each segment between boundaries (comma-separated list).')
# checkpoint related configuration
tf.app.flags.DEFINE_string(
    'checkpoint_path', './model',
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'checkpoint_model_scope', 'vgg_16',
    'Model scope in the checkpoint. None if the same as the trained model.')
tf.app.flags.DEFINE_string(
    'model_scope', 'ssd300',
    'Model scope name used to replace the name_scope in checkpoint.')
tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', 'ssd300/multibox_head, ssd300/additional_layers, ssd300/conv4_3_scale',
    'Comma-separated list of scopes of variables to exclude when restoring from a checkpoint.')
tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', True,
    'When restoring a checkpoint would ignore missing variables.')
tf.app.flags.DEFINE_boolean(
    'multi_gpu', True,
    'Whether there is GPU to use for training.')

# evaluation related configuration
tf.app.flags.DEFINE_float(
    'select_threshold', 0.01, 'Class-specific confidence score threshold for selecting a box.')
tf.app.flags.DEFINE_float(
    'min_size', 0.03, 'The min size of bboxes to keep.')
tf.app.flags.DEFINE_float(
    'nms_threshold', 0.45, 'Matching threshold in NMS algorithm.')
tf.app.flags.DEFINE_integer(
    'nms_topk', 200, 'Number of total object to keep after NMS.')
tf.app.flags.DEFINE_integer(
    'keep_topk', 400, 'Number of total object to keep for each image before nms.')

# visualization realted configuration
tf.app.flags.DEFINE_integer(
    'max_examples_to_draw', 20, 'Number of image to draw while eval.')
tf.app.flags.DEFINE_integer(
    'max_boxes_to_draw', 50, 'Number of bbox to draw per image while eval.')
tf.app.flags.DEFINE_integer(
    'min_score_thresh', 20, 'min score of bbox to draw')

# distillation configuration
tf.app.flags.DEFINE_integer(
    'step1_classes',10,'Number of classes use for training first task'
)
tf.app.flags.DEFINE_integer(
    'step2_classes',10,'Number of classes use for training second task'
)

FLAGS = tf.app.flags.FLAGS

def parse_comma_list(args):
    return [float(s.strip()) for s in args.split(',')]

def get_init_fn():
    return scaffolds.get_init_fn_for_scaffold(FLAGS.model_dir, FLAGS.checkpoint_path,
                                              FLAGS.model_scope, FLAGS.checkpoint_model_scope,
                                              FLAGS.checkpoint_exclude_scopes, FLAGS.ignore_missing_vars,
                                              name_remap={'/kernel': '/weights', '/bias': '/biases'})


def main(_):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    # gpu config
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    # multi gpu training strategy
    distribute_strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=2)

    # Set up a RunConfig to only save checkpoints once per training cycle.
    run_config = tf.estimator.RunConfig().replace(
        save_checkpoints_secs=FLAGS.save_checkpoints_secs).replace(
        save_checkpoints_steps=10).replace(
        save_summary_steps=FLAGS.save_summary_steps).replace(
        keep_checkpoint_max=5).replace(
        tf_random_seed=FLAGS.tf_random_seed).replace(
        log_step_count_steps=FLAGS.log_every_n_steps).replace(
        train_distribute=distribute_strategy
    )

    # replicate_ssd_model_fn = tf.contrib.estimator.replicate_model_fn(ssd_model_fn, loss_reduction=tf.losses.Reduction.MEAN)
    ssd_detector = tf.estimator.Estimator(
        model_fn=ssd_model_fn, model_dir=FLAGS.model_dir, config=run_config,
        params={
            # training
            'num_gpus': 2,
            'data_format': FLAGS.data_format,
            'batch_size': FLAGS.batch_size,
            'model_scope': FLAGS.model_scope,
            'num_classes': FLAGS.num_classes,
            'negative_ratio': FLAGS.negative_ratio,
            'positive_threshold': FLAGS.positive_threshold,
            'neg_threshold': FLAGS.neg_threshold,
            'weight_decay': FLAGS.weight_decay,
            'momentum': FLAGS.momentum,
            'learning_rate': FLAGS.learning_rate,
            'end_learning_rate': FLAGS.end_learning_rate,
            'decay_boundaries': parse_comma_list(FLAGS.decay_boundaries),
            'lr_decay_factors': parse_comma_list(FLAGS.lr_decay_factors),
            # evaluation
            'select_threshold': FLAGS.select_threshold,
            'min_size': FLAGS.min_size,
            'nms_threshold': FLAGS.nms_threshold,
            'nms_topk': FLAGS.nms_topk,
            'keep_topk': FLAGS.keep_topk,
            'eval_metric_fn_key': "coco_detection_metrics",
            'max_nms_detections' : 100,
            # visualize
            'max_examples_to_draw': FLAGS.max_examples_to_draw,
            'max_boxes_to_draw': FLAGS.max_boxes_to_draw,
            'min_score_thresh': FLAGS.min_score_thresh,
            # distillation
            'distillation': False
        })

    # log tensor
    train_tensors_to_log = {
        'lr': 'learning_rate',
        'ce': 'cross_entropy_loss',
        'loc': 'location_loss',
        'loss': 'total_loss',
        'l2': 'l2_loss',
        'acc': 'post_forward/cls_accuracy',
        'num_positives':'post_forward/num_positives',
        'num_negatives_selected':'post_forward/num_negatives_select'
    }
    train_logging_hook = tf.train.LoggingTensorHook(tensors=train_tensors_to_log, every_n_iter=FLAGS.log_every_n_steps,
                                              formatter = lambda dicts: (', '.join(['%s=%.6f' % (k, v) for k, v in dicts.items()])))
    eval_tensors_to_log = {
        'ce': 'cross_entropy_loss',
        'loc': 'location_loss',
        'loss': 'total_loss',
        'l2': 'l2_loss',
        'acc': 'post_forward/cls_accuracy',
        'num_positives': 'post_forward/num_positives',
        'num_negatives_selected': 'post_forward/num_negatives_select',
        'num_detections_after_nms': 'num_detections_after_nms'
    }
    eval_logging_hook = tf.train.LoggingTensorHook(tensors=eval_tensors_to_log, every_n_iter=FLAGS.log_every_n_steps,
                                                   formatter=lambda dicts: (', '.join(['%s=%.6f' % (k, v) for k, v in dicts.items()])))

    train_input_pattern = '/home/dxfang/dataset/tfrecords/pascal_voc/train-000*'
    eval_input_pattern = '/home/dxfang/dataset/tfrecords/pascal_voc/eval-000*'

    print('Starting a training cycle.')
    train_spec = tf.estimator.TrainSpec(
        input_fn=input_pipeline(file_pattern=train_input_pattern, is_training=True, batch_size=FLAGS.batch_size),
        max_steps=FLAGS.max_number_of_steps,
        hooks= [train_logging_hook]
    )
    eval_spec = tf.estimator.EvalSpec(
        input_fn=input_pipeline(file_pattern=eval_input_pattern, is_training=False, batch_size=FLAGS.batch_size),
        hooks=[eval_logging_hook]
    )

    tf.estimator.train_and_evaluate(ssd_detector,train_spec,eval_spec)

    # ssd_detector.evaluate(input_fn=input_pipeline(file_pattern=eval_input_pattern,
    #                                               is_training=False,
    #                                               batch_size=FLAGS.batch_size),
    #                       steps=20,
    #                       hooks=[eval_logging_hook])

    # distillation
    # class_for_use = range(1,FLAGS.step1_classes+1)
    # ssd_detector.train(input_fn=dist_input_fn(class_list=None,file_pattern=train_input_pattern,
    #                                            is_training=True,
    #                                            batch_size=FLAGS.batch_size),
    #                    hooks=[train_logging_hook],
    #                    max_steps=FLAGS.max_number_of_steps)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()