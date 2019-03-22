# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Common utility functions for evaluation."""
import collections
import os
import time

import numpy as np
import tensorflow as tf

# from object_detection.core import box_list
# from object_detection.core import box_list_ops
# from object_detection.core import keypoint_ops
# from object_detection.core import standard_fields as fields
from metrics import coco_evaluation
# from object_detection.utils import label_map_util
from utils import object_detection_evaluation

slim = tf.contrib.slim

# A dictionary of metric names to classes that implement the metric. The classes
# in the dictionary must implement
# utils.object_detection_evaluation.DetectionEvaluator interface.
EVAL_METRICS_CLASS_DICT = {
    'coco_detection_metrics':
        coco_evaluation.CocoDetectionEvaluator,
    # 'coco_mask_metrics':
    #     coco_evaluation.CocoMaskEvaluator,
    # 'oid_challenge_detection_metrics':
    #     object_detection_evaluation.OpenImagesDetectionChallengeEvaluator,
    'pascal_voc_detection_metrics':
        object_detection_evaluation.PascalDetectionEvaluator,
    # 'weighted_pascal_voc_detection_metrics':
    #     object_detection_evaluation.WeightedPascalDetectionEvaluator,
    # 'pascal_voc_instance_segmentation_metrics':
    #     object_detection_evaluation.PascalInstanceSegmentationEvaluator,
    # 'weighted_pascal_voc_instance_segmentation_metrics':
    #     object_detection_evaluation.WeightedPascalInstanceSegmentationEvaluator,
    # 'oid_V2_detection_metrics':
    #     object_detection_evaluation.OpenImagesDetectionEvaluator,
}

EVAL_DEFAULT_METRIC = 'coco_detection_metrics'


def write_metrics(metrics, global_step, summary_dir):
    """Write metrics to a summary directory.

    Args:
      metrics: A dictionary containing metric names and values.
      global_step: Global step at which the metrics are computed.
      summary_dir: Directory to write tensorflow summaries to.
    """
    tf.logging.info('Writing metrics to tf summary.')
    summary_writer = tf.summary.FileWriterCache.get(summary_dir)
    for key in sorted(metrics):
        summary = tf.Summary(value=[
            tf.Summary.Value(tag=key, simple_value=metrics[key]),
        ])
        summary_writer.add_summary(summary, global_step)
        tf.logging.info('%s: %f', key, metrics[key])
    tf.logging.info('Metrics written to tf summary.')

def get_evaluators(categories, eval_metric_fn_key="coco_detection_metrics"):
    if eval_metric_fn_key== "coco_detection_metrics":
        kwargs_dict = {
            'include_metrics_per_category' : True,
            'all_metrics_per_category' : True
          }
    else:
        kwargs_dict = {}
    evaluator = EVAL_METRICS_CLASS_DICT[eval_metric_fn_key](categories, **kwargs_dict)
    return evaluator
