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

"""Utility functions for building models."""

from __future__ import print_function

import collections
import os
import six
import time

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import lookup_ops
from third_party.nmt.utils import misc_utils as utils

from px.nmt.utils import iterator_utils
from px.nmt.utils import vocab_utils

__all__ = [
    "get_initializer", "get_device_str", "create_train_model",
    "create_train_model_for_server", "create_eval_model", "create_infer_model",
    "create_emb_for_encoder_and_decoder", "create_rnn_cell", "gradient_clip",
    "create_or_load_model", "load_model", "avg_checkpoints",
    "compute_perplexity"
]

# If a vocab size is greater than this value, put the embedding on cpu instead
VOCAB_SIZE_THRESHOLD_CPU = 50000


def get_initializer(init_op, seed=None, init_weight=None):
  """Create an initializer. init_weight is only for uniform."""
  if init_op == "uniform":
    assert init_weight
    return tf.random_uniform_initializer(-init_weight, init_weight, seed=seed)
  elif init_op == "glorot_normal":
    return tf.keras.initializers.glorot_normal(seed=seed)
  elif init_op == "glorot_uniform":
    return tf.keras.initializers.glorot_uniform(seed=seed)
  else:
    raise ValueError("Unknown init_op %s" % init_op)


def get_device_str(device_id, num_gpus):
  """Return a device string for multi-GPU setup."""
  if num_gpus == 0:
    return "/cpu:0"
  device_str_output = "/gpu:%d" % (device_id % num_gpus)
  return device_str_output


class ExtraArgs(
    collections.namedtuple(
        "ExtraArgs",
        ("single_cell_fn", "model_device_fn", "attention_mechanism_fn"))):
  pass


class TrainModel(
    collections.namedtuple(
        "TrainModel",
        ("graph", "model", "iterator", "src_placeholder", "tgt_placeholder",
         "annot_placeholder", "skip_count_placeholder"))):
  pass


def create_train_model(model_creator,
                       hparams,
                       scope=None,
                       num_workers=1,
                       jobid=0,
                       graph=None,
                       extra_args=None,
                       trie=None,
                       use_placeholders=False):
  """Create train graph, model, and iterator."""
  src_vocab_file = hparams.src_vocab_file
  tgt_vocab_file = hparams.tgt_vocab_file

  if not graph:
    graph = tf.Graph()

  with graph.as_default(), tf.container(scope or "train"):
    src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
        src_vocab_file, tgt_vocab_file, hparams.share_vocab)
    reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_file(
        tgt_vocab_file, default_value=vocab_utils.UNK)

    annot_placeholder = None
    src_placeholder = None
    tgt_placeholder = None
    annot_dataset = None
    ctx_dataset = None
    if use_placeholders:
      src_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
      src_dataset = tf.data.Dataset.from_tensor_slices(src_placeholder)

      tgt_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
      tgt_dataset = tf.data.Dataset.from_tensor_slices(tgt_placeholder)

      if hparams.use_rl:
        annot_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        annot_dataset = tf.data.Dataset.from_tensor_slices(annot_placeholder)
    else:
      src_file = "%s.%s" % (hparams.train_prefix, hparams.src)
      tgt_file = "%s.%s" % (hparams.train_prefix, hparams.tgt)
      ctx_file = None
      if hparams.ctx is not None:
        ctx_file = "%s.%s" % (hparams.train_prefix, hparams.ctx)

      src_dataset = tf.data.TextLineDataset(src_file)
      tgt_dataset = tf.data.TextLineDataset(tgt_file)

      if hparams.train_annotations is not None:
        annot_dataset = tf.data.TextLineDataset(hparams.train_annotations)

      if ctx_file is not None:
        ctx_dataset = tf.data.TextLineDataset(ctx_file)

    skip_count_placeholder = tf.placeholder(shape=(), dtype=tf.int64)

    iterator = iterator_utils.get_iterator(
        hparams=hparams,
        src_dataset=src_dataset,
        tgt_dataset=tgt_dataset,
        src_vocab_table=src_vocab_table,
        tgt_vocab_table=tgt_vocab_table,
        ctx_dataset=ctx_dataset,
        annot_dataset=annot_dataset,
        batch_size=hparams.batch_size,
        sos=hparams.sos,
        eos=hparams.eos,
        random_seed=hparams.random_seed,
        num_buckets=hparams.num_buckets,
        src_max_len=hparams.src_max_len,
        tgt_max_len=hparams.tgt_max_len,
        skip_count=skip_count_placeholder,
        num_shards=num_workers,
        shard_index=jobid)

    # Note: One can set model_device_fn to
    # `tf.train.replica_device_setter(ps_tasks)` for distributed training.
    model_device_fn = None
    if extra_args:
      model_device_fn = extra_args.model_device_fn
    with tf.device(model_device_fn):
      model = model_creator(
          hparams=hparams,
          iterator=iterator,
          mode=tf.contrib.learn.ModeKeys.TRAIN,
          source_vocab_table=src_vocab_table,
          target_vocab_table=tgt_vocab_table,
          reverse_target_vocab_table=reverse_tgt_vocab_table,
          scope=scope,
          extra_args=extra_args,
          trie=trie)

  return TrainModel(
      graph=graph,
      model=model,
      iterator=iterator,
      src_placeholder=src_placeholder,
      tgt_placeholder=tgt_placeholder,
      annot_placeholder=annot_placeholder,
      skip_count_placeholder=skip_count_placeholder)


class TrainModelForServer(
    collections.namedtuple(
        "TrainModelForServer",
        ("graph", "model", "iterator", "src_placeholder", "tgt_placeholder",
         "wgt_placeholder", "batch_size_placeholder", "skip_count_placeholder"))
):
  pass


def create_train_model_for_server(model_creator,
                                  hparams,
                                  scope=None,
                                  num_workers=1,
                                  jobid=0,
                                  graph=None,
                                  extra_args=None,
                                  trie=None):
  """Create graph, model, and iterator when running the NMT in server mode.

  This is different from the standard training model, because the input arrives
  via RPC and thus has to be fed using placeholders."""
  assert hparams.num_buckets == 1, "No bucketing when in server mode."

  src_vocab_file = hparams.src_vocab_file
  tgt_vocab_file = hparams.tgt_vocab_file

  if not graph:
    graph = tf.Graph()

  with graph.as_default(), tf.container(scope or "train"):
    src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
        src_vocab_file, tgt_vocab_file, hparams.share_vocab)
    reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_file(
        tgt_vocab_file, default_value=vocab_utils.UNK)
    skip_count_placeholder = tf.placeholder(shape=(), dtype=tf.int64)

    src_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
    src_dataset = tf.data.Dataset.from_tensor_slices(src_placeholder)

    tgt_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
    tgt_dataset = tf.data.Dataset.from_tensor_slices(tgt_placeholder)

    wgt_placeholder = tf.placeholder(shape=[None], dtype=tf.float32)
    wgt_dataset = tf.data.Dataset.from_tensor_slices(wgt_placeholder)

    ctx_placeholder = None
    if hparams.ctx is not None:
      ctx_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
    ctx_dataset = None
    if ctx_placeholder is not None:
      ctx_dataset = tf.data.Dataset.from_tensor_slices(ctx_placeholder)

    batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)

    iterator = iterator_utils.get_iterator(
        hparams=hparams,
        src_dataset=src_dataset,
        tgt_dataset=tgt_dataset,
        src_vocab_table=src_vocab_table,
        tgt_vocab_table=tgt_vocab_table,
        wgt_dataset=wgt_dataset,
        ctx_dataset=ctx_dataset,
        annot_dataset=None,
        batch_size=batch_size_placeholder,
        sos=hparams.sos,
        eos=hparams.eos,
        random_seed=hparams.random_seed,
        num_buckets=hparams.num_buckets,
        src_max_len=hparams.src_max_len,
        tgt_max_len=hparams.tgt_max_len,
        skip_count=skip_count_placeholder,
        num_shards=num_workers,
        shard_index=jobid)

    # Note: One can set model_device_fn to
    # `tf.train.replica_device_setter(ps_tasks)` for distributed training.
    model_device_fn = None
    if extra_args:
      model_device_fn = extra_args.model_device_fn
    with tf.device(model_device_fn):
      model = model_creator(
          hparams=hparams,
          iterator=iterator,
          mode=tf.contrib.learn.ModeKeys.TRAIN,
          source_vocab_table=src_vocab_table,
          target_vocab_table=tgt_vocab_table,
          reverse_target_vocab_table=reverse_tgt_vocab_table,
          scope=scope,
          extra_args=extra_args,
          trie=trie)

  return TrainModelForServer(
      graph=graph,
      model=model,
      iterator=iterator,
      src_placeholder=src_placeholder,
      tgt_placeholder=tgt_placeholder,
      wgt_placeholder=wgt_placeholder,
      batch_size_placeholder=batch_size_placeholder,
      skip_count_placeholder=skip_count_placeholder)


class EvalModel(
    collections.namedtuple(
        "EvalModel",
        ("graph", "model", "src_file_placeholder", "tgt_file_placeholder",
         "ctx_file_placeholder", "annot_file_placeholder", "iterator"))):
  pass


def create_eval_model(model_creator,
                      hparams,
                      scope=None,
                      graph=None,
                      extra_args=None,
                      trie=None):
  """Create train graph, model, src/tgt file holders, and iterator."""
  src_vocab_file = hparams.src_vocab_file
  tgt_vocab_file = hparams.tgt_vocab_file

  if not graph:
    graph = tf.Graph()

  with graph.as_default(), tf.container(scope or "eval"):
    src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
        src_vocab_file, tgt_vocab_file, hparams.share_vocab)
    reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_file(
        tgt_vocab_file, default_value=vocab_utils.UNK)
    src_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
    tgt_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)

    ctx_file_placeholder = None
    if hparams.ctx is not None:
      ctx_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)

    annot_file_placeholder = None
    if hparams.dev_annotations is not None:
      annot_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)

    src_dataset = tf.data.TextLineDataset(src_file_placeholder)
    tgt_dataset = tf.data.TextLineDataset(tgt_file_placeholder)

    ctx_dataset = None
    if ctx_file_placeholder is not None:
      ctx_dataset = tf.data.TextLineDataset(ctx_file_placeholder)

    annot_dataset = None
    if annot_file_placeholder is not None:
      annot_dataset = tf.data.TextLineDataset(annot_file_placeholder)

    iterator = iterator_utils.get_iterator(
        hparams=hparams,
        src_dataset=src_dataset,
        tgt_dataset=tgt_dataset,
        src_vocab_table=src_vocab_table,
        tgt_vocab_table=tgt_vocab_table,
        batch_size=hparams.batch_size,
        sos=hparams.sos,
        eos=hparams.eos,
        ctx_dataset=ctx_dataset,
        annot_dataset=annot_dataset,
        random_seed=hparams.random_seed,
        num_buckets=hparams.num_buckets,
        src_max_len=hparams.src_max_len_infer,
        tgt_max_len=hparams.tgt_max_len_infer)
    model = model_creator(
        hparams=hparams,
        iterator=iterator,
        mode=tf.contrib.learn.ModeKeys.EVAL,
        source_vocab_table=src_vocab_table,
        target_vocab_table=tgt_vocab_table,
        reverse_target_vocab_table=reverse_tgt_vocab_table,
        scope=scope,
        extra_args=extra_args,
        trie=trie)

  return EvalModel(
      graph=graph,
      model=model,
      src_file_placeholder=src_file_placeholder,
      tgt_file_placeholder=tgt_file_placeholder,
      ctx_file_placeholder=ctx_file_placeholder,
      annot_file_placeholder=annot_file_placeholder,
      iterator=iterator)


class InferModel(
    collections.namedtuple(
        "InferModel", ("graph", "model", "src_placeholder", "annot_placeholder",
                       "ctx_placeholder", "trie_exclude_placeholder",
                       "batch_size_placeholder", "iterator"))):
  pass


def create_infer_model(model_creator,
                       hparams,
                       scope=None,
                       graph=None,
                       extra_args=None,
                       trie=None):
  """Create inference model."""
  src_vocab_file = hparams.src_vocab_file
  tgt_vocab_file = hparams.tgt_vocab_file

  if not graph:
    graph = tf.Graph()

  with graph.as_default(), tf.container(scope or "infer"):
    src_vocab_table, tgt_vocab_table = vocab_utils.create_vocab_tables(
        src_vocab_file, tgt_vocab_file, hparams.share_vocab)
    reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_file(
        tgt_vocab_file, default_value=vocab_utils.UNK)

    src_placeholder = tf.placeholder(shape=[None], dtype=tf.string)

    ctx_placeholder = None
    if hparams.ctx is not None:
      ctx_placeholder = tf.placeholder(shape=[None], dtype=tf.string)

    annot_placeholder = None
    if hparams.use_rl:
      annot_placeholder = tf.placeholder(shape=[None], dtype=tf.string)

    batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)

    src_dataset = tf.data.Dataset.from_tensor_slices(src_placeholder)

    ctx_dataset = None
    if ctx_placeholder is not None:
      ctx_dataset = tf.data.Dataset.from_tensor_slices(ctx_placeholder)

    annot_dataset = None
    if annot_placeholder is not None:
      annot_dataset = tf.data.Dataset.from_tensor_slices(annot_placeholder)

    trie_exclude_placeholder = None
    trie_exclude_dataset = None
    if hparams.infer_mode.startswith("trie_"):
      trie_exclude_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
      trie_exclude_dataset = tf.data.Dataset.from_tensor_slices(
          trie_exclude_placeholder)

    iterator = iterator_utils.get_infer_iterator(
        hparams=hparams,
        src_dataset=src_dataset,
        src_vocab_table=src_vocab_table,
        tgt_vocab_table=tgt_vocab_table,
        ctx_dataset=ctx_dataset,
        annot_dataset=annot_dataset,
        trie_exclude_dataset=trie_exclude_dataset,
        batch_size=batch_size_placeholder,
        eos=hparams.eos,
        src_max_len=hparams.src_max_len_infer)
    model = model_creator(
        hparams=hparams,
        iterator=iterator,
        mode=tf.contrib.learn.ModeKeys.INFER,
        source_vocab_table=src_vocab_table,
        target_vocab_table=tgt_vocab_table,
        reverse_target_vocab_table=reverse_tgt_vocab_table,
        scope=scope,
        extra_args=extra_args,
        trie=trie)
  return InferModel(
      graph=graph,
      model=model,
      src_placeholder=src_placeholder,
      annot_placeholder=annot_placeholder,
      trie_exclude_placeholder=trie_exclude_placeholder,
      ctx_placeholder=ctx_placeholder,
      batch_size_placeholder=batch_size_placeholder,
      iterator=iterator)


def _get_embed_device(vocab_size):
  """Decide on which device to place an embed matrix given its vocab size."""
  if vocab_size > VOCAB_SIZE_THRESHOLD_CPU:
    return "/cpu:0"
  else:
    return "/gpu:0"


def _create_pretrained_emb_from_txt(vocab_file,
                                    embed_file,
                                    num_trainable_tokens=3,
                                    dtype=tf.float32,
                                    scope=None):
  """Load pretrain embeding from embed_file, and return an embedding matrix.

  Args:
    embed_file: Path to a Glove formatted embedding txt file.
    num_trainable_tokens: Make the first n tokens in the vocab file as trainable
      variables. Default is 3, which is "<unk>", "<s>" and "</s>".
  """
  vocab, _ = vocab_utils.load_vocab(vocab_file)
  trainable_tokens = vocab[:num_trainable_tokens]

  utils.print_out("# Using pretrained embedding: %s." % embed_file)
  utils.print_out("  with trainable tokens: ")

  emb_dict, emb_size = vocab_utils.load_embed_txt(embed_file)
  for token in trainable_tokens:
    utils.print_out("    %s" % token)
    if token not in emb_dict:
      emb_dict[token] = [0.0] * emb_size

  emb_mat = np.array(
      [emb_dict[token] for token in vocab], dtype=dtype.as_numpy_dtype())
  emb_mat = tf.constant(emb_mat)
  emb_mat_const = tf.slice(emb_mat, [num_trainable_tokens, 0], [-1, -1])
  with tf.variable_scope(scope or "pretrain_embeddings", dtype=dtype) as scope:
    with tf.device(_get_embed_device(num_trainable_tokens)):
      emb_mat_var = tf.get_variable("emb_mat_var",
                                    [num_trainable_tokens, emb_size])
  return tf.concat([emb_mat_var, emb_mat_const], 0)


def _create_or_load_embed(embed_name, vocab_file, embed_file, vocab_size,
                          embed_size, dtype):
  """Create a new or load an existing embedding matrix."""
  if vocab_file and embed_file:
    embedding = _create_pretrained_emb_from_txt(vocab_file, embed_file)
  else:
    with tf.device(_get_embed_device(vocab_size)):
      embedding = tf.get_variable(embed_name, [vocab_size, embed_size], dtype)
  return embedding


def create_emb_for_encoder_and_decoder(share_vocab,
                                       src_vocab_size,
                                       tgt_vocab_size,
                                       src_embed_size,
                                       tgt_embed_size,
                                       dtype=tf.float32,
                                       num_partitions=0,
                                       src_vocab_file=None,
                                       tgt_vocab_file=None,
                                       src_embed_file=None,
                                       tgt_embed_file=None,
                                       scope=None):
  """Create embedding matrix for both encoder and decoder.

  Args:
    share_vocab: A boolean. Whether to share embedding matrix for both
      encoder and decoder.
    src_vocab_size: An integer. The source vocab size.
    tgt_vocab_size: An integer. The target vocab size.
    src_embed_size: An integer. The embedding dimension for the encoder's
      embedding.
    tgt_embed_size: An integer. The embedding dimension for the decoder's
      embedding.
    dtype: dtype of the embedding matrix. Default to float32.
    num_partitions: number of partitions used for the embedding vars.
    scope: VariableScope for the created subgraph. Default to "embedding".

  Returns:
    embedding_encoder: Encoder's embedding matrix.
    embedding_decoder: Decoder's embedding matrix.

  Raises:
    ValueError: if use share_vocab but source and target have different vocab
      size.
  """

  if num_partitions <= 1:
    partitioner = None
  else:
    # Note: num_partitions > 1 is required for distributed training due to
    # embedding_lookup tries to colocate single partition-ed embedding variable
    # with lookup ops. This may cause embedding variables being placed on worker
    # jobs.
    partitioner = tf.fixed_size_partitioner(num_partitions)

  if (src_embed_file or tgt_embed_file) and partitioner:
    raise ValueError(
        "Can't set num_partitions > 1 when using pretrained embedding")

  with tf.variable_scope(
      scope or "embeddings",
      dtype=dtype,
      partitioner=partitioner,
      reuse=tf.AUTO_REUSE) as scope:
    # Share embedding
    if share_vocab:
      if src_vocab_size != tgt_vocab_size:
        raise ValueError("Share embedding but different src/tgt vocab sizes"
                         " %d vs. %d" % (src_vocab_size, tgt_vocab_size))
      assert src_embed_size == tgt_embed_size
      utils.print_out("# Use the same embedding for source and target")
      vocab_file = src_vocab_file or tgt_vocab_file
      embed_file = src_embed_file or tgt_embed_file

      embedding_encoder = _create_or_load_embed("embedding_share", vocab_file,
                                                embed_file, src_vocab_size,
                                                src_embed_size, dtype)
      embedding_decoder = embedding_encoder
    else:
      with tf.variable_scope(
          "encoder", partitioner=partitioner, reuse=tf.AUTO_REUSE):
        embedding_encoder = _create_or_load_embed(
            "embedding_encoder", src_vocab_file, src_embed_file, src_vocab_size,
            src_embed_size, dtype)

      with tf.variable_scope(
          "decoder", partitioner=partitioner, reuse=tf.AUTO_REUSE):
        embedding_decoder = _create_or_load_embed(
            "embedding_decoder", tgt_vocab_file, tgt_embed_file, tgt_vocab_size,
            tgt_embed_size, dtype)

  return embedding_encoder, embedding_decoder


def _single_cell(unit_type,
                 num_units,
                 forget_bias,
                 dropout,
                 mode,
                 residual_connection=False,
                 device_str=None,
                 residual_fn=None):
  """Create an instance of a single RNN cell."""
  # dropout (= 1 - keep_prob) is set to 0 during eval and infer
  dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0

  # Cell Type
  if unit_type == "lstm":
    utils.print_out("  LSTM, forget_bias=%g" % forget_bias, new_line=False)
    single_cell = tf.contrib.rnn.BasicLSTMCell(
        num_units, forget_bias=forget_bias)
  elif unit_type == "gru":
    utils.print_out("  GRU", new_line=False)
    single_cell = tf.contrib.rnn.GRUCell(num_units)
  elif unit_type == "layer_norm_lstm":
    utils.print_out(
        "  Layer Normalized LSTM, forget_bias=%g" % forget_bias, new_line=False)
    single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
        num_units, forget_bias=forget_bias, layer_norm=True)
  elif unit_type == "nas":
    utils.print_out("  NASCell", new_line=False)
    single_cell = tf.contrib.rnn.NASCell(num_units)
  else:
    raise ValueError("Unknown unit type %s!" % unit_type)

  # Dropout (= 1 - keep_prob)
  if dropout > 0.0:
    single_cell = tf.contrib.rnn.DropoutWrapper(
        cell=single_cell, input_keep_prob=(1.0 - dropout))
    utils.print_out(
        "  %s, dropout=%g " % (type(single_cell).__name__, dropout),
        new_line=False)

  # Residual
  if residual_connection:
    single_cell = tf.contrib.rnn.ResidualWrapper(
        single_cell, residual_fn=residual_fn)
    utils.print_out("  %s" % type(single_cell).__name__, new_line=False)

  # Device Wrapper
  if device_str:
    single_cell = tf.contrib.rnn.DeviceWrapper(single_cell, device_str)
    utils.print_out(
        "  %s, device=%s" % (type(single_cell).__name__, device_str),
        new_line=False)

  return single_cell


def _cell_list(unit_type,
               num_units,
               num_layers,
               num_residual_layers,
               forget_bias,
               dropout,
               mode,
               num_gpus,
               base_gpu=0,
               single_cell_fn=None,
               residual_fn=None):
  """Create a list of RNN cells."""
  if not single_cell_fn:
    single_cell_fn = _single_cell

  # Multi-GPU
  cell_list = []
  for i in range(num_layers):
    utils.print_out("  cell %d" % i, new_line=False)
    single_cell = single_cell_fn(
        unit_type=unit_type,
        num_units=num_units,
        forget_bias=forget_bias,
        dropout=dropout,
        mode=mode,
        residual_connection=(i >= num_layers - num_residual_layers),
        device_str=get_device_str(i + base_gpu, num_gpus),
        residual_fn=residual_fn)
    utils.print_out("")
    cell_list.append(single_cell)

  return cell_list


def create_rnn_cell(unit_type,
                    num_units,
                    num_layers,
                    num_residual_layers,
                    forget_bias,
                    dropout,
                    mode,
                    num_gpus,
                    base_gpu=0,
                    single_cell_fn=None):
  """Create multi-layer RNN cell.

  Args:
    unit_type: string representing the unit type, i.e. "lstm".
    num_units: the depth of each unit.
    num_layers: number of cells.
    num_residual_layers: Number of residual layers from top to bottom. For
      example, if `num_layers=4` and `num_residual_layers=2`, the last 2 RNN
      cells in the returned list will be wrapped with `ResidualWrapper`.
    forget_bias: the initial forget bias of the RNNCell(s).
    dropout: floating point value between 0.0 and 1.0:
      the probability of dropout.  this is ignored if `mode != TRAIN`.
    mode: either tf.contrib.learn.TRAIN/EVAL/INFER
    num_gpus: The number of gpus to use when performing round-robin
      placement of layers.
    base_gpu: The gpu device id to use for the first RNN cell in the
      returned list. The i-th RNN cell will use `(base_gpu + i) % num_gpus`
      as its device id.
    single_cell_fn: allow for adding customized cell.
      When not specified, we default to model_helper._single_cell
  Returns:
    An `RNNCell` instance.
  """
  cell_list = _cell_list(
      unit_type=unit_type,
      num_units=num_units,
      num_layers=num_layers,
      num_residual_layers=num_residual_layers,
      forget_bias=forget_bias,
      dropout=dropout,
      mode=mode,
      num_gpus=num_gpus,
      base_gpu=base_gpu,
      single_cell_fn=single_cell_fn)

  if len(cell_list) == 1:  # Single layer.
    return cell_list[0]
  else:  # Multi layers
    return tf.contrib.rnn.MultiRNNCell(cell_list)


def gradient_clip(gradients, max_gradient_norm):
  """Clipping gradients of a model."""
  clipped_gradients, gradient_norm = tf.clip_by_global_norm(
      gradients, max_gradient_norm)
  gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
  gradient_norm_summary.append(
      tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))

  return clipped_gradients, gradient_norm_summary, gradient_norm


def load_model(model, ckpt, session, name):
  start_time = time.time()
  model.saver.restore(session, ckpt)
  session.run(tf.tables_initializer())
  utils.print_out("  loaded %s model parameters from %s, time %.2fs" %
                  (name, ckpt, time.time() - start_time))
  return model


def avg_checkpoints(model_dir, num_last_checkpoints, global_step,
                    global_step_name):
  """Average the last N checkpoints in the model_dir."""
  checkpoint_state = tf.train.get_checkpoint_state(model_dir)
  if not checkpoint_state:
    utils.print_out("# No checkpoint file found in directory: %s" % model_dir)
    return None

  # Checkpoints are ordered from oldest to newest.
  checkpoints = (
      checkpoint_state.all_model_checkpoint_paths[-num_last_checkpoints:])

  if len(checkpoints) < num_last_checkpoints:
    utils.print_out(
        "# Skipping averaging checkpoints because not enough checkpoints is "
        "avaliable.")
    return None

  avg_model_dir = os.path.join(model_dir, "avg_checkpoints")
  if not tf.gfile.Exists(avg_model_dir):
    utils.print_out(
        "# Creating new directory %s for saving averaged checkpoints." %
        avg_model_dir)
    tf.gfile.MakeDirs(avg_model_dir)

  utils.print_out("# Reading and averaging variables in checkpoints:")
  var_list = tf.contrib.framework.list_variables(checkpoints[0])
  var_values, var_dtypes = {}, {}
  for (name, shape) in var_list:
    if name != global_step_name:
      var_values[name] = np.zeros(shape)

  for checkpoint in checkpoints:
    utils.print_out("    %s" % checkpoint)
    reader = tf.contrib.framework.load_checkpoint(checkpoint)
    for name in var_values:
      tensor = reader.get_tensor(name)
      var_dtypes[name] = tensor.dtype
      var_values[name] += tensor

  for name in var_values:
    var_values[name] /= len(checkpoints)

  # Build a graph with same variables in the checkpoints, and save the averaged
  # variables into the avg_model_dir.
  with tf.Graph().as_default():
    tf_vars = [
        tf.get_variable(v, shape=var_values[v].shape, dtype=var_dtypes[name])
        for v in var_values
    ]

    placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
    assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
    global_step_var = tf.Variable(
        global_step, name=global_step_name, trainable=False)
    saver = tf.train.Saver(tf.all_variables())

    with tf.Session() as sess:
      sess.run(tf.initialize_all_variables())
      for p, assign_op, (name, value) in zip(placeholders, assign_ops,
                                             six.iteritems(var_values)):
        sess.run(assign_op, {p: value})

      # Use the built saver to save the averaged checkpoint. Only keep 1
      # checkpoint and the best checkpoint will be moved to avg_best_metric_dir.
      saver.save(sess, os.path.join(avg_model_dir, "translate.ckpt"))

  return avg_model_dir


def get_global_step(model, session):
  return model.global_step.eval(session=session)


def create_or_load_model(model, model_dir, session, name):
  """Create translation model and initialize or load parameters in session."""
  latest_ckpt = tf.train.latest_checkpoint(model_dir)
  if latest_ckpt:
    model = load_model(model, latest_ckpt, session, name)
  else:
    start_time = time.time()
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    utils.print_out("  created %s model with fresh parameters, time %.2fs" %
                    (name, time.time() - start_time))

  global_step = model.global_step.eval(session=session)
  return model, global_step


def compute_perplexity(hparams, model, sess, name):
  """Compute perplexity of the output of the model.

  Args:
    hparams: holds the parameters.
    model: model for compute perplexity.
    sess: tensorflow session to use.
    name: name of the batch.

  Returns:
    The perplexity of the eval outputs.
  """
  total_loss = 0
  total_predict_count = 0
  start_time = time.time()
  step = 0
  start_time_step = time.time()
  while True:
    try:
      loss, _, _, predict_count, batch_size = model.eval(sess)
      total_loss += loss * batch_size
      total_predict_count += predict_count
      if step % hparams.steps_per_stats == 0:
        # print_time does not print decimal places for time.
        utils.print_out("  computing perplexity %s, step %d, time %.3f" %
                        (name, step, time.time() - start_time_step))
      step += 1
      start_time_step = time.time()
    except tf.errors.OutOfRangeError:
      break

  perplexity = utils.safe_exp(total_loss / total_predict_count)
  utils.print_time("  eval %s: perplexity %.2f" % (name, perplexity),
                   start_time)
  return perplexity
