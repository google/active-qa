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
"""For loading data into NMT models."""
from __future__ import print_function

import collections

import tensorflow as tf

__all__ = ["BatchedInput", "get_iterator", "get_infer_iterator"]


# heavily relies on simple but long lambda functions.
# pylint: disable=g-long-lambda
# NOTE(ebrevdo): When we subclass this, instances' __dict__ becomes empty.
class BatchedInput(
    collections.namedtuple(
        "BatchedInput", ("initializer", "source_string", "source",
                         "target_input", "target_output", "weights", "context",
                         "annotation", "trie_exclude", "source_sequence_length",
                         "target_sequence_length", "context_sequence_length"))):
  pass


def get_infer_iterator(hparams,
                       src_dataset,
                       src_vocab_table,
                       batch_size,
                       eos,
                       ctx_dataset=None,
                       annot_dataset=None,
                       trie_exclude_dataset=None,
                       src_max_len=None,
                       tgt_vocab_table=None):
  src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)

  # even in append mode, this will be only the source, without attached context.
  src_string_dataset = src_dataset

  # Create a fake dataset for weights.
  wgt_dataset = src_dataset.map(lambda src: tf.constant(1.0))

  # Create a fake context dataset.
  if ctx_dataset is None:
    ctx_dataset = src_dataset.map(lambda src: tf.constant("no context"))

  # Create a fake annotations dataset.
  if annot_dataset is None:
    annot_dataset = src_dataset.map(lambda src: tf.constant("1\t1"))

  # Create a fake trie exclude dataset.
  if trie_exclude_dataset is None:
    trie_exclude_dataset = src_dataset.map(lambda src: tf.constant(""))

  if tgt_vocab_table is None:
    tgt_vocab_table = src_vocab_table

  if hparams.context_feed == "append":
    src_dataset = tf.data.Dataset.zip((ctx_dataset, src_dataset)).map(
        lambda ctx, src: ctx + " " + hparams.context_delimiter + " " + src)

  src_dataset = src_dataset.map(lambda src: tf.string_split([src]).values)
  ctx_dataset = ctx_dataset.map(lambda ctx: tf.string_split([ctx]).values)
  annot_dataset = annot_dataset.map(
      # We only need the first column, which contains the doc id that will be
      # later passed to the environment.
      lambda annot: tf.string_split([annot], delimiter="\t").values[0])
  trie_exclude_dataset = trie_exclude_dataset.map(
      lambda tex: tf.string_split([tex]).values)

  if src_max_len:
    src_dataset = src_dataset.map(lambda src: src[:src_max_len])
  # Convert the word strings to ids.
  src_dataset = src_dataset.map(
      lambda src: tf.cast(src_vocab_table.lookup(src), tf.int32))
  ctx_dataset = ctx_dataset.map(
      lambda ctx: tf.cast(src_vocab_table.lookup(ctx), tf.int32))
  trie_exclude_dataset = trie_exclude_dataset.map(
      lambda tex: tf.cast(tgt_vocab_table.lookup(tex), tf.int32))

  # Append context with <eos> so the length will be at least 1.
  ctx_dataset = ctx_dataset.map(lambda ctx: tf.concat((ctx, [src_eos_id]), 0))
  trie_exclude_dataset = trie_exclude_dataset.map(
      lambda tex: tf.reduce_join(tf.as_string(tex), separator=" "))

  src_dataset = tf.data.Dataset.zip(
      (src_string_dataset, src_dataset, wgt_dataset, ctx_dataset, annot_dataset,
       trie_exclude_dataset))
  # Add in the annotations and word counts.
  src_dataset = src_dataset.map(
      lambda src_string, src, wgt, ctx, annot, tex: (
          src_string,
          src,
          wgt,
          ctx,
          annot,
          tex,
          tf.size(src),
          tf.size(ctx))
  )

  def batching_func(x):
    return x.padded_batch(
        batch_size,
        # The entry is the source line rows;
        # this has unknown-length vectors. The last entry is
        # the source row size; this is a scalar.
        padded_shapes=(
            tf.TensorShape([]),  # src_string
            tf.TensorShape([None]),  # src
            tf.TensorShape([]),  # wgt
            tf.TensorShape([None]),  # ctx
            tf.TensorShape([]),  # annot
            tf.TensorShape([]),  # tex
            tf.TensorShape([]),  # src_len
            tf.TensorShape([])),  # ctx_len
        # Pad the source and context sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        padding_values=(
            "",  # src_string
            src_eos_id,  # src
            1.0,  # wgt
            src_eos_id,  # ctx
            "",  # annot --unused
            "",  # tex --unused
            0,  # src_len -- unused
            0))  # ctx_len -- unused

  batched_dataset = batching_func(src_dataset)
  batched_iter = batched_dataset.make_initializable_iterator()
  (src_raw, src_ids, weights, ctx_ids, annot_strs, tex_strs, src_seq_len,
   ctx_seq_len) = batched_iter.get_next()
  return BatchedInput(
      initializer=batched_iter.initializer,
      source_string=src_raw,
      source=src_ids,
      target_input=None,
      target_output=None,
      weights=weights,
      context=ctx_ids,
      annotation=annot_strs,
      trie_exclude=tex_strs,
      source_sequence_length=src_seq_len,
      target_sequence_length=None,
      context_sequence_length=ctx_seq_len)


def get_iterator(hparams,
                 src_dataset,
                 tgt_dataset,
                 src_vocab_table,
                 tgt_vocab_table,
                 batch_size,
                 sos,
                 eos,
                 random_seed,
                 num_buckets,
                 wgt_dataset=None,
                 ctx_dataset=None,
                 annot_dataset=None,
                 src_max_len=None,
                 tgt_max_len=None,
                 num_parallel_calls=4,
                 output_buffer_size=None,
                 skip_count=None,
                 num_shards=1,
                 shard_index=0,
                 reshuffle_each_iteration=True):
  if not output_buffer_size:
    output_buffer_size = batch_size * 1000
  src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
  tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

  # Create a fake weight dataset.
  if wgt_dataset is None:
    wgt_dataset = src_dataset.map(lambda src: tf.constant(1.0))

  # Create a fake context dataset.
  if ctx_dataset is None:
    ctx_dataset = src_dataset.map(lambda src: tf.constant("no context"))

  # Create a fake annotations dataset.
  if annot_dataset is None:
    annot_dataset = src_dataset.map(lambda src: tf.constant("1\t1"))

  # Create a fake trie exclude dataset.
  trie_exclude_dataset = src_dataset.map(lambda src: tf.constant(""))

  if hparams.context_feed == "append":
    src_dataset = tf.data.Dataset.zip((ctx_dataset, src_dataset)).map(
        lambda ctx, src: ctx + " " + hparams.context_delimiter + " " + src)

  src_tgt_dataset = tf.data.Dataset.zip(
      (src_dataset, tgt_dataset, wgt_dataset, ctx_dataset, annot_dataset,
       trie_exclude_dataset))

  src_tgt_dataset = src_tgt_dataset.shard(num_shards, shard_index)

  if skip_count is not None:
    src_tgt_dataset = src_tgt_dataset.skip(skip_count)

  src_tgt_dataset = src_tgt_dataset.shuffle(output_buffer_size, random_seed,
                                            reshuffle_each_iteration)

  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt, wgt, ctx, annot, tex: (
          src,
          tf.string_split([src]).values,
          tf.string_split([tgt]).values,
          wgt,
          tf.string_split([ctx]).values,
          tf.string_split([annot], delimiter="\t").values[0],
          tex),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  # Filter zero length input sequences.
  src_tgt_dataset = src_tgt_dataset.filter(
      lambda src_string, src, tgt, wgt, ctx, annot, tex: (
          tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0)))

  # In non-server mode we truncate the targets before adding the sentence-end
  # token, possibly resulting in length tgt_max_len+1.
  if not hparams.server_mode and tgt_max_len:
    # Limit target length.
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src_string, src, tgt, wgt, ctx, annot, tex: \
        (src_string, src, tgt[:tgt_max_len], wgt, ctx, annot, tex),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
  # Also append <eos> to the context.
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src_string, src, tgt, wgt, ctx, annot, tex: (
          src_string,
          src,
          tf.concat(([sos], tgt), 0),
          tf.concat((tgt, [eos]), 0),
          wgt,
          tf.concat((ctx, [eos]), 0),
          annot,
          tex),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  if src_max_len:
    # This may introduce an inconsistency between src_string and the tokenized
    # source which is cut off.
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src_string, src, tgt_in, tgt_out, wgt, ctx, annot, tex: \
        (src_string, src[:src_max_len], tgt_in, tgt_out, wgt, ctx, annot, tex),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  # In server mode we truncate the targets after adding the sentence-end token,
  # possibly removing the sentence-end token.
  # This is done to mimic the behaviour in join reinforcement learning mode
  # (hparams.use_rl=True) where we train based on what is produced via
  # inference. Inference is stopped if no sentence end token was produced after
  # a number of steps.
  if hparams.server_mode and tgt_max_len:
    # Limit target length.
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src_string, src, tgt_in, tgt_out, wgt, ctx, annot, tex: \
        (src_string, src, tgt_in[:tgt_max_len], tgt_out[:tgt_max_len], wgt, \
         ctx, annot, tex),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  # Convert the word strings to ids.  Word strings that are not in the
  # vocab get the lookup table's default_value integer.
  # Source vocab is also used in context
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src_string, src, tgt_in, tgt_out, wgt, ctx, annot, tex: (
          src_string,
          tf.cast(src_vocab_table.lookup(src), tf.int32),
          tf.cast(tgt_vocab_table.lookup(tgt_in), tf.int32),
          tf.cast(tgt_vocab_table.lookup(tgt_out), tf.int32),
          wgt,
          tf.cast(src_vocab_table.lookup(ctx), tf.int32),
          annot, tex),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  # Add in sequence lengths.
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src_string, src, tgt_in, tgt_out, wgt, ctx, annot, tex: (
          src_string, src, tgt_in, tgt_out, wgt, ctx, annot, tex,
          tf.size(src), tf.size(tgt_in), tf.size(ctx)),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
  def batching_func(x):
    return x.padded_batch(
        batch_size,
        # The first three entries are the source and target line rows;
        # these have unknown-length vectors.  The last two entries are
        # the source and target row sizes; these are scalars.
        padded_shapes=(
            tf.TensorShape([]),  # src_string
            tf.TensorShape([None]),  # src
            tf.TensorShape([None]),  # tgt_input
            tf.TensorShape([None]),  # tgt_output
            tf.TensorShape([]),  # wgt
            tf.TensorShape([None]),  # ctx
            tf.TensorShape([]),  # annot
            tf.TensorShape([]),  # tex
            tf.TensorShape([]),  # src_len
            tf.TensorShape([]),  # tgt_len
            tf.TensorShape([])),  # ctx_len
        # Pad the source and target sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        padding_values=(
            "",  # src_string
            src_eos_id,  # src
            tgt_eos_id,  # tgt_input
            tgt_eos_id,  # tgt_output
            0.0,  # wgt
            src_eos_id,  # ctx
            "",  # annot -- unused
            "",  # tex -- unused
            0,  # src_len -- unused
            0,  # tgt_len -- unused
            0))  # ctx_len -- unused

  if num_buckets > 1:

    def key_func(unused_1, unused_2, unused_3, unused_4, unused_5, unused_6,
                 unused_7, unused_8, src_len, tgt_len, unused_9):
      # Calculate bucket_width by maximum source sequence length.
      # Pairs with length [0, bucket_width) go to bucket 0, length
      # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
      # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
      if src_max_len:
        bucket_width = (src_max_len + num_buckets - 1) // num_buckets
      else:
        bucket_width = 10

      # Bucket sentence pairs by the length of their source sentence and target
      # sentence.
      bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
      return tf.to_int64(tf.minimum(num_buckets, bucket_id))

    def reduce_func(unused_key, windowed_data):
      return batching_func(windowed_data)

    batched_dataset = src_tgt_dataset.apply(
        tf.contrib.data.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=batch_size))
  else:
    batched_dataset = batching_func(src_tgt_dataset)
  batched_iter = batched_dataset.make_initializable_iterator()
  (src_string, src_ids, tgt_input_ids, tgt_output_ids, weights, ctx_ids,
   annot_strs, tex_strs, src_seq_len, tgt_seq_len,
   ctx_seq_len) = batched_iter.get_next()

  return BatchedInput(
      initializer=batched_iter.initializer,
      source_string=src_string,
      source=src_ids,
      target_input=tgt_input_ids,
      target_output=tgt_output_ids,
      weights=weights,
      context=ctx_ids,
      annotation=annot_strs,
      trie_exclude=tex_strs,
      source_sequence_length=src_seq_len,
      target_sequence_length=tgt_seq_len,
      context_sequence_length=ctx_seq_len)
