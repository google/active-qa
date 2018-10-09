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

"""Convert the context string into a vector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from third_party.nmt.utils import misc_utils as utils

from px.nmt import model_helper

utils.check_tensorflow_version()


def feed(context_vector, encoder_outputs, encoder_state, hparams):
  """Feed the context vector into to model.

  Args:
    context_vector: A context vector of [batch, vector_size]
    encoder_outputs: The source encoder outputs.
      Will be passed into the attention.
    encoder_state: The source encoder final hidden state.
      Will be passed into decoder initial state.
    hparams: Hyperparameters configurations.
  Returns:
    encoder outputs ans encoder state that have been fed with context.
  Raises:
    ValueError: if context_feed value is not defined.
  """

  # string append. Do nothing
  if hparams.context_feed == "append":
    return encoder_outputs, encoder_state

  # feed the context into the decoder initial hidden state
  elif hparams.context_feed == "decoder_hidden_state":
    if hparams.context_vector == "last_state":
      encoder_state = context_vector
    else:
      encoder_state = ((tf.contrib.rnn.LSTMStateTuple(
          context_vector, context_vector),) * len(encoder_state))

  # feed the context into the encoder output
  elif hparams.context_feed == "encoder_output":
    if hparams.context_vector != "bilstm_full":
      context_vector = tf.expand_dims(context_vector, 0)

    encoder_outputs = tf.concat([context_vector, encoder_outputs], 0)

  else:
    raise ValueError("Unknown context_feed mode: {}"
                     .format(hparams.context_feed))

  return encoder_outputs, encoder_state


def get_context_vector(mode, iterator, hparams, vector_size=None):
  """Convert the context string into a vector.

  Args:
    mode: Must be tf.contrib.learn.ModeKeys.TRAIN,
      tf.contrib.learn.ModeKeys.EVAL, or tf.contrib.learn.ModeKeys.INFER.
    iterator: A BatchedInput iterator.
    hparams: Hyperparameters configurations.
    vector_size: context vector size. Will be hparams.num_units if undefined.
  Returns:
    A context vector tensor of size [batch_size, vector_size].
  Raises:
    ValueError: if context_vector value is not defined.
  """
  if hparams.context_vector == "append":
    return None
  if vector_size is None:
    vector_size = hparams.num_units

  # maxpooling over all encoder's outputs (https://arxiv.org/abs/1709.04348).
  if hparams.context_vector == "bilstm_pool":
    encoder_outputs, _ = _build_lstm_encoder(mode, iterator, hparams)
    # maxpool over time axis
    context_vector = tf.reduce_max(encoder_outputs, 0)

  # get all encoder outputs
  elif hparams.context_vector == "bilstm_full":
    encoder_outputs, _ = _build_lstm_encoder(mode, iterator, hparams)
    return encoder_outputs

  # get the last encoder output
  elif hparams.context_vector == "bilstm_last":
    encoder_outputs, _ = _build_lstm_encoder(mode, iterator, hparams)
    # get the last encoder output
    context_vector = get_last_encoder_output(encoder_outputs,
                                             iterator.context_sequence_length)

  # 4-layers CNN, then pool over all layers (https://arxiv.org/abs/1709.04348).
  elif hparams.context_vector == "cnn":
    context_vector = get_cnn_vector(mode, iterator, hparams)

  # get the last LSTM hidden state.
  elif hparams.context_vector == "last_state":
    _, encoder_state = _build_lstm_encoder(mode, iterator, hparams)
    return encoder_state

  else:
    raise ValueError("Unknown context_vector mode: {}"
                     .format(hparams.context_vector))

  # resize the context vector to the desired length
  resizer = tf.get_variable(
      "context_resizer",
      shape=(context_vector.get_shape()[1], vector_size),
      dtype=tf.float32)
  context_vector = tf.tanh(tf.matmul(context_vector, resizer))
  return context_vector


def get_embeddings(hparams, iterator):
  """Look up embedding, encoder_emb_inp: [max_time, batch_size, num_units]."""
  source = iterator.context
  # Make shape [max_time, batch_size].
  source = tf.transpose(source)

  embedding_context = tf.get_variable(
      "embedding_context", [hparams.src_vocab_size, hparams.num_units],
      tf.float32)
  encoder_emb_inp = tf.nn.embedding_lookup(embedding_context, source)
  return encoder_emb_inp


def get_cnn_vector(mode, iterator, hparams, kernels=[3, 3, 3, 3]):
  with tf.variable_scope("context_cnn_encoder"):
    conv = get_embeddings(hparams, iterator)

    # Set axis into [batch_size, max_time, num_units] to simplify CNN operations
    conv = tf.transpose(conv, [1, 0, 2])

    maxpools = []
    for layer, kernel_size in enumerate(kernels):
      conv = conv1d(conv, [kernel_size, conv.shape[2].value, hparams.num_units],
                    layer)
      # maxpool on time axis
      maxpools.append(tf.reduce_max(conv, 1))

  # flatten the unit axis
  maxpools = tf.concat(maxpools, -1)
  return maxpools


def conv1d(tensor, filter_shape, layer):
  weight = tf.get_variable("W_{}".format(layer), filter_shape)
  bias = tf.get_variable("b_{}".format(layer), [filter_shape[2]])
  conv = tf.nn.conv1d(tensor, weight, stride=1, padding="SAME", name="conv")
  return tf.nn.relu(tf.nn.bias_add(conv, bias), name="relu")


def get_last_encoder_output(encoder_outputs, sequence_length):
  # Make shape [batch_size, max_time, num_units].
  encoder_outputs = tf.transpose(encoder_outputs, [1, 0, 2])
  batch_range = tf.range(tf.shape(encoder_outputs)[0])
  indices = tf.stack([batch_range, sequence_length - 1], axis=1)
  return tf.gather_nd(encoder_outputs, indices)


def _build_lstm_encoder(mode, iterator, hparams):
  """Build an encoder."""
  num_layers = hparams.num_encoder_layers
  num_residual_layers = hparams.num_residual_layers

  with tf.variable_scope("context_rnn_encoder") as scope:
    dtype = scope.dtype
    encoder_emb_inp = get_embeddings(hparams, iterator)

    num_bi_layers = int(num_layers / 2)
    num_bi_residual_layers = int(num_residual_layers / 2)

    # Shape of encoder_outputs if time majoris  True:
    #   [max_time, batch_size, num_units]
    # Shape of encoder_outputs if time major is False:
    #   [batch_size, max_time, num_units]
    encoder_outputs, bi_encoder_state = (
        _build_bidirectional_rnn(
            inputs=encoder_emb_inp,
            sequence_length=iterator.context_sequence_length,
            dtype=dtype,
            hparams=hparams,
            mode=mode,
            num_bi_layers=num_bi_layers,
            num_bi_residual_layers=num_bi_residual_layers))

    if num_bi_layers == 1:
      encoder_state = bi_encoder_state
    else:
      # alternatively concat forward and backward states
      encoder_state = []
      for layer_id in range(num_bi_layers):
        encoder_state.append(bi_encoder_state[0][layer_id])  # forward
        encoder_state.append(bi_encoder_state[1][layer_id])  # backward
      encoder_state = tuple(encoder_state)
  return encoder_outputs, encoder_state


def _build_bidirectional_rnn(inputs,
                             sequence_length,
                             dtype,
                             hparams,
                             mode,
                             num_bi_layers,
                             num_bi_residual_layers,
                             base_gpu=0):
  """Create and call biddirectional RNN cells.

  Args:
    num_residual_layers: Number of residual layers from top to bottom. For
      example, if `num_bi_layers=4` and `num_residual_layers=2`, the last 2 RNN
      layers in each RNN cell will be wrapped with `ResidualWrapper`.
    base_gpu: The gpu device id to use for the first forward RNN layer. The
      i-th forward RNN layer will use `(base_gpu + i) % num_gpus` as its
      device id. The `base_gpu` for backward RNN cell is `(base_gpu +
      num_bi_layers)`.

  Returns:
    The concatenated bidirectional output and the bidirectional RNN cell"s
    state.
  """
  # Construct forward and backward cells
  fw_cell = _build_encoder_cell(
      hparams, mode, num_bi_layers, num_bi_residual_layers, base_gpu=base_gpu)
  bw_cell = _build_encoder_cell(
      hparams,
      mode,
      num_bi_layers,
      num_bi_residual_layers,
      base_gpu=(base_gpu + num_bi_layers))

  bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
      fw_cell,
      bw_cell,
      inputs,
      dtype=dtype,
      sequence_length=sequence_length,
      time_major=True)

  return tf.concat(bi_outputs, -1), bi_state


def _build_encoder_cell(hparams,
                        mode,
                        num_layers,
                        num_residual_layers,
                        base_gpu=0):
  """Build a multi-layer RNN cell that can be used by encoder."""

  return model_helper.create_rnn_cell(
      unit_type=hparams.unit_type,
      num_units=hparams.num_units,
      num_layers=num_layers,
      num_residual_layers=num_residual_layers,
      forget_bias=hparams.forget_bias,
      dropout=hparams.dropout,
      num_gpus=hparams.num_gpus,
      mode=mode,
      base_gpu=base_gpu)
