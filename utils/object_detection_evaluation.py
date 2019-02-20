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
"""object_detection_evaluation module.

ObjectDetectionEvaluation is a class which manages ground truth information of a
object detection dataset, and computes frequently used detection metrics such as
Precision, Recall, CorLoc of the provided detection results.
It supports the following operations:
1) Add ground truth information of images sequentially.
2) Add detection result of images sequentially.
3) Evaluate detection metrics on already inserted detection results.
4) Write evaluation result into a pickle file for future processing or
   visualization.

Note: This module operates on numpy boxes and box lists.
"""

from abc import ABCMeta
from abc import abstractmethod
import collections
import logging
import unicodedata
import numpy as np
import tensorflow as tf


class DetectionEvaluator(object):
  """Interface for object detection evalution classes.

  Example usage of the Evaluator:
  ------------------------------
  evaluator = DetectionEvaluator(categories)

  # Detections and groundtruth for image 1.
  evaluator.add_single_groundtruth_image_info(...)
  evaluator.add_single_detected_image_info(...)

  # Detections and groundtruth for image 2.
  evaluator.add_single_groundtruth_image_info(...)
  evaluator.add_single_detected_image_info(...)

  metrics_dict = evaluator.evaluate()
  """
  __metaclass__ = ABCMeta

  def __init__(self, categories):
    """Constructor.

    Args:
      categories: A list of dicts, each of which has the following keys -
        'id': (required) an integer id uniquely identifying this category.
        'name': (required) string representing category name e.g., 'cat', 'dog'.
    """
    self._categories = categories

  @abstractmethod
  def add_single_ground_truth_image_info(self, image_id, groundtruth_dict):
    """Adds groundtruth for a single image to be used for evaluation.

    Args:
      image_id: A unique string/integer identifier for the image.
      groundtruth_dict: A dictionary of groundtruth numpy arrays required
        for evaluations.
    """
    pass

  @abstractmethod
  def add_single_detected_image_info(self, image_id, detections_dict):
    """Adds detections for a single image to be used for evaluation.

    Args:
      image_id: A unique string/integer identifier for the image.
      detections_dict: A dictionary of detection numpy arrays required
        for evaluation.
    """
    pass

  def get_estimator_eval_metric_ops(self, eval_dict):
    """Returns dict of metrics to use with `tf.estimator.EstimatorSpec`.

    Note that this must only be implemented if performing evaluation with a
    `tf.estimator.Estimator`.

    Args:
      eval_dict: A dictionary that holds tensors for evaluating an object
        detection model, returned from
        eval_util.result_dict_for_single_example().

    Returns:
      A dictionary of metric names to tuple of value_op and update_op that can
      be used as eval metric ops in `tf.estimator.EstimatorSpec`.
    """
    pass

  @abstractmethod
  def evaluate(self):
    """Evaluates detections and returns a dictionary of metrics."""
    pass

  @abstractmethod
  def clear(self):
    """Clears the state to prepare for a fresh evaluation."""
    pass
