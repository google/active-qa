# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utility functions for evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


def compute_f1_single(prediction, ground_truth):
  """Computes F1 score for a single prediction/ground_truth pair.

  This function computes the token-level intersection between the predicted and
  ground-truth answers, consistent with the SQuAD evaluation script.

  This is different from another common way of computing F1 (e.g. used by BiDAF)
  that uses the intersection between predicted answer span and ground-truth
  spans. The main difference is that this method gives credit to correct partial
  answers that don't match any full answer span, while the other one wouldn't.

  Args:
    prediction: predicted string.
    ground_truth: ground truth string.

  Returns:
    f1: Token-wise F1 score between the two input strings.
  """
  assert isinstance(prediction, unicode)
  assert isinstance(ground_truth, unicode)

  prediction_tokens = prediction.split()
  ground_truth_tokens = ground_truth.split()
  common = collections.Counter(prediction_tokens) & collections.Counter(
      ground_truth_tokens)
  num_same = sum(common.values())
  if num_same == 0:
    return 0.0
  precision = num_same / len(prediction_tokens)
  recall = num_same / len(ground_truth_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1


def compute_f1(prediction, gold_answers):
  """Computes F1 score for a single prediction and a list of gold answers.

  See 'compute_f1_single' for details. Expects all input strings to be unicode.

  Args:
    prediction: predicted string.
    gold_answers: a list of ground truth strings.

  Returns:
    f1: Maximum of the token-wise F1 score between the prediction and each gold
        answer.
  """
  if not gold_answers:
    return 0.0
  else:
    return max(
        [compute_f1_single(prediction, answer) for answer in gold_answers])
