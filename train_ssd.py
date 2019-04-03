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
import sys
import time

import numpy as np
import tensorflow as tf

from models import ssd_model_fn

from utils import scaffolds

# hardware related configuration
from inputs import input_pipeline

tf.app.flags.DEFINE_integer(
    'num_readers', 2,
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
    'data_dir', '/data/kingdom/obj_detection/dataset/tfrecords/pascal_voc/',
    'The directory where the dataset input data is stored.')
tf.app.flags.DEFINE_integer(
    'num_classes', 3, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_integer(
    'log_every_n_steps',50,
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
    'max_number_of_steps', 200000,
    'The max number of steps to use for training.')
tf.app.flags.DEFINE_integer(
    'batch_size',16,
    'Batch size for training and evaluation.')
tf.app.flags.DEFINE_string(
    'data_format', 'channels_last',  # 'channels_first' or 'channels_last'
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
    'decay_boundaries', '500,100000',
    'Learning rate decay boundaries by global_step (comma-separated list).')
tf.app.flags.DEFINE_string(
    'lr_decay_factors', '0.1,1, 0.1',
    'The values of learning_rate decay factor for each segment between boundaries (comma-separated list).')

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
    'nms_topk', 200, 'Number of total object per class to keep after NMS.')
tf.app.flags.DEFINE_integer(
    'keep_topk', 400, 'Number of total object to keep for each class before nms.')

# visualization realted configuration
tf.app.flags.DEFINE_integer(
    'max_examples_to_draw', 30, 'Number of image to draw while eval.')
tf.app.flags.DEFINE_integer(
    'max_boxes_to_draw',50, 'Number of bbox to draw per image while eval.')
tf.app.flags.DEFINE_float(
    'min_score_thresh',0.3, 'min score of bbox to draw')

# distillation configuration
tf.app.flags.DEFINE_integer(
    'step1_classes',10,'Number of classes use for training first task'
)
tf.app.flags.DEFINE_integer(
    'step2_classes',10,'Number of classes use for training second task'
)

# '2019-03-21-13-32-46_w_pretrained_wo_bn'
# 'pretrained_ssd'
# '2019-03-29-13-31-17_pretrained_SEnet'
# '2019-03-31-19-25-57_pretrained_w_bn_SEnet'
save_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
model_dir_string = os.path.join('./logs','2019-04-02-00-53-50_pretrained')
tf.app.flags.DEFINE_string(
    'model_dir', model_dir_string,
    'The directory where the model will be stored.')

# checkpoint related configuration
tf.app.flags.DEFINE_string(
    'checkpoint_path', './logs/pretrained_vgg',
    'The path to a checkpoint from which to fine-tune.')

FLAGS = tf.app.flags.FLAGS

def parse_comma_list(args):
    return [float(s.strip()) for s in args.split(',')]

def main(_):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    # gpu config
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    # multi gpu training strategy
    distribute_strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=2)
    # Set up a RunConfig to only save checkpoints once per training cycle.
    run_config = tf.estimator.RunConfig().replace(
        save_checkpoints_secs=FLAGS.save_checkpoints_secs).replace(
        save_checkpoints_steps=5000).replace(
        save_summary_steps=FLAGS.save_summary_steps).replace(
        keep_checkpoint_max=5).replace(
        tf_random_seed=FLAGS.tf_random_seed).replace(
        log_step_count_steps=FLAGS.log_every_n_steps)\
        .replace(
        train_distribute=distribute_strategy
    )
    estimator_params = {
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
        # batch normalization
        'backbone_batch_normal': False,
        'additional_batch_normal': False,
        'bn_detection_head': False,
        # init_fn
        'model_dir': FLAGS.model_dir,
        'checkpoint_path': FLAGS.checkpoint_path,
        'checkpoint_model_scope': FLAGS.checkpoint_model_scope,
        'checkpoint_exclude_scopes': FLAGS.checkpoint_exclude_scopes,
        'ignore_missing_vars': FLAGS.ignore_missing_vars,
        # evaluation
        'select_threshold': FLAGS.select_threshold,
        'min_size': FLAGS.min_size,
        'nms_threshold': FLAGS.nms_threshold,
        'nms_topk': FLAGS.nms_topk,
        'keep_topk': FLAGS.keep_topk,
        'eval_metric_fn_key': "coco_detection_metrics",
        # 'eval_metric_fn_key': "pascal_voc_detection_metrics",
        'pad_nms_detections': 4000,
        # visualize
        'max_examples_to_draw': FLAGS.max_examples_to_draw,
        'max_boxes_to_draw': FLAGS.max_boxes_to_draw,
        'min_score_thresh': FLAGS.min_score_thresh,

        # distillation
        'distillation': False,
        'is_tack_A': False,
        'is_tack_B': False
    }

    ssd_detector = tf.estimator.Estimator(
        model_fn=ssd_model_fn, model_dir=FLAGS.model_dir, config=run_config,
        params=estimator_params,
        # warm_start_from=FLAGS.checkpoint_path
        warm_start_from=None
        )

    # log tensor
    train_tensors_to_log = {
        'lr': 'learning_rate',
        'ce': 'cross_entropy_loss',
        'loc': 'location_loss',
        'loss': 'total_loss',
        'l2': 'l2_loss',
        'acc': 'cls_accuracy',
        'num_positives': 'hard_example_mining/num_positives',
        'num_negatives_selected': 'hard_example_mining/num_negatives_select'
    }
    train_logging_hook = tf.train.LoggingTensorHook(tensors=train_tensors_to_log, every_n_iter=FLAGS.log_every_n_steps,
                                              formatter = lambda dicts: (', '.join(['%s=%.6f' % (k, v) for k, v in dicts.items()])))
    eval_tensors_to_log = {
        'ce': 'cross_entropy_loss',
        'loc': 'location_loss',
        'loss': 'total_loss',
        'l2': 'l2_loss',
        'acc': 'cls_accuracy',
        'num_positives': 'hard_example_mining/num_positives',
        'num_negatives_selected': 'hard_example_mining/num_negatives_select',
        'num_detections_after_nms': 'evaluation_scope/num_detections_after_nms'
    }
    eval_logging_hook = tf.train.LoggingTensorHook(tensors=eval_tensors_to_log, every_n_iter=FLAGS.log_every_n_steps,
                                                   formatter=lambda dicts: (', '.join(['%s=%.6f' % (k, v) for k, v in dicts.items()])))

    # train_input_pattern = '/data/kingdom/obj_detection/dataset/tfrecords/pascal_voc/train-000*'
    # eval_input_pattern ='/data/kingdom/obj_detection/dataset/tfrecords/pascal_voc/val-000*'
    train_input_pattern = '/data/kingdom/obj_detection/dataset/tfrecords/widerface_ccpd/train/*'
    eval_input_pattern =[ "/data/kingdom/obj_detection/dataset/tfrecords/widerface_ccpd/eval/wider_face/*",
    "/data/kingdom/obj_detection/dataset/tfrecords/widerface_ccpd/eval/ccpd/*"]

    print('Starting a training cycle.')
    task_A_list = list(range(1,11))
    task_B_list = list(range(11,21))
    task_A_list = None
    train_spec = tf.estimator.TrainSpec(
        input_fn=input_pipeline(class_list=task_A_list,file_pattern=train_input_pattern, is_training=True, batch_size=FLAGS.batch_size,data_format=FLAGS.data_format),
        max_steps=FLAGS.max_number_of_steps,
        hooks= [train_logging_hook],
    )
    eval_spec = tf.estimator.EvalSpec(
        input_fn=input_pipeline(class_list=task_A_list,file_pattern=eval_input_pattern, is_training=False, batch_size=FLAGS.batch_size,data_format=FLAGS.data_format,num_readers=1),
        hooks=[eval_logging_hook],
        throttle_secs=600
    )

    tf.estimator.train_and_evaluate(ssd_detector,train_spec,eval_spec)

    # ssd_detector.train(
    #     input_fn=input_pipeline(class_list=None,
    #                             file_pattern=train_input_pattern,
    #                             is_training=True,
    #                             data_format=FLAGS.data_format,
    #                             batch_size=FLAGS.batch_size),
    #     steps=1000,
    #     hooks=[train_logging_hook]
    # )


    # ssd_detector.evaluate(input_fn=input_pipeline(file_pattern=eval_input_pattern,
    #                                               is_training=False,
    #                                               batch_size=FLAGS.batch_size,
    #                                               data_format=FLAGS.data_format,
    #                                               num_readers=1),
    #                       hooks=[eval_logging_hook],
    #                       )



    predict = False
    if predict:
        print('Starting a predict cycle.')
        predict_dir = os.path.join(FLAGS.model_dir, 'predict')
        if not os.path.isdir(predict_dir):
            os.mkdir(predict_dir)
        pred_results = ssd_detector.predict(
            input_fn=input_pipeline(file_pattern=eval_input_pattern, is_training=False, batch_size=1,
                                    data_format=FLAGS.data_format),
        )
        det_results = list(pred_results)
        # print(list(det_results))

        # [{'bboxes_1': array([[0.        , 0.        , 0.28459054, 0.5679505 ], [0.3158835 , 0.34792888, 0.7312541 , 1.        ]], dtype=float32), 'scores_17': array([0.01333667, 0.01152573], dtype=float32), 'filename': b'000703.jpg', 'shape': array([334, 500,   3])}]
        for class_ind in range(1, FLAGS.num_classes):
            with open(os.path.join(predict_dir, 'results_{}.txt'.format(class_ind)), 'wt') as f:
                for image_ind, pred in enumerate(det_results):
                    filename = pred['filename']
                    shape = pred['shape']
                    scores = pred['scores_{}'.format(class_ind)]
                    bboxes = pred['bboxes_{}'.format(class_ind)]

                    bboxes[:, 0] = (bboxes[:, 0] * shape[0]).astype(np.int32, copy=False) + 1
                    bboxes[:, 1] = (bboxes[:, 1] * shape[1]).astype(np.int32, copy=False) + 1
                    bboxes[:, 2] = (bboxes[:, 2] * shape[0]).astype(np.int32, copy=False) + 1
                    bboxes[:, 3] = (bboxes[:, 3] * shape[1]).astype(np.int32, copy=False) + 1

                    valid_mask = np.logical_and((bboxes[:, 2] - bboxes[:, 0] > 0), (bboxes[:, 3] - bboxes[:, 1] > 0))

                    for det_ind in range(valid_mask.shape[0]):
                        if not valid_mask[det_ind]:
                            continue
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(filename.decode('utf8')[:-4], scores[det_ind],
                                       bboxes[det_ind, 1], bboxes[det_ind, 0],
                                       bboxes[det_ind, 3], bboxes[det_ind, 2]))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()