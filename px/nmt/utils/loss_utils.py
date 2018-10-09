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

"""Utils related to calculating losses."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def mask_padding(tensor, sequence_length):
  """Mask out the padding values.

  Args:
    tensor: A tensor of shape [T, B].
    sequence_length: A tensor of size [B], storing the non padded length for
      every entry in the batch.

  Returns:
    A tensor of shape [T, B], having value 0 for all padded positions.
  """
  mask = tf.sequence_mask(
      tf.to_int32(sequence_length), tf.to_int32(tf.shape(tensor)[0]))
  return tensor * tf.transpose(tf.to_float(mask), [1, 0])


def cross_entropy_sequence_loss(logits, targets, sequence_length, weights=None):
  """Calculates the per-example cross-entropy loss for a sequence of logits.

  Masks out all losses passed the sequence length.

  Args:
    logits: Logits of shape [T, B, vocab_size]
    targets: Target classes of shape [T, B]
    sequence_length: An int32 tensor of shape [B] corresponding
      to the length of each input
    weights: Instance weights of shape [B]

  Returns:
    A tensor of shape [T, B] that contains the loss per example, per time step.
  """
  with tf.name_scope('cross_entropy_sequence_loss'):
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=targets)
    losses = mask_padding(losses, sequence_length)
    if weights is not None:
      losses *= weights
    return losses


def entropy_regularization(logits, sequence_length):
  """Compute entropy regularization.

  Masks out all values passed the sequence length.

  Args:
    logits: Logits of shape [T, B, vocab_size]
    sequence_length: An int32 tensor of shape [B] corresponding
      to the length of each input

  Returns:
    A tensor of shape [T, B] that contains the entropy regularization factor
    per example, per time step.
  """
  with tf.name_scope('entropy_regularization'):
    probs = tf.nn.softmax(logits)
    entropy_reg = -tf.reduce_sum(probs * tf.log(probs + 1e-32), -1)
    return mask_padding(entropy_reg, sequence_length)
