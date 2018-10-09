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

"""Common utility functions for tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops import lookup_ops

from px.nmt.utils import iterator_utils


def create_test_hparams(unit_type="lstm",
                        encoder_type="uni",
                        num_layers=4,
                        attention="",
                        attention_architecture=None,
                        use_residual=False,
                        inference_indices=None,
                        init_op="uniform",
                        beam_width=0,
                        num_translations_per_input=1,
                        ctx=None,
                        context_vector="",
                        context_feed="",
                        train_annotations=None,
                        infer_mode="greedy"):
  """Create training and inference test hparams."""
  num_residual_layers = 0
  if use_residual:
    num_residual_layers = 2

  return tf.contrib.training.HParams(
      # Networks
      num_units=5,
      num_encoder_layers=num_layers,
      num_decoder_layers=num_layers,
      dropout=0.5,
      unit_type=unit_type,
      encoder_type=encoder_type,
      residual=use_residual,
      num_residual_layers=num_residual_layers,
      num_embeddings_partitions=0,

      # Attention mechanisms
      attention=attention,
      attention_architecture=attention_architecture,
      output_attention=True,
      pass_hidden_state=True,

      # Train
      optimizer="sgd",
      init_op=init_op,
      init_weight=0.1,
      max_gradient_norm=5.0,
      max_emb_gradient_norm=None,
      learning_rate=1.0,
      warmup_steps=0,
      warmup_scheme="t2t",
      num_train_steps=1,
      decay_scheme="",
      colocate_gradients_with_ops=True,
      batch_size=128,
      num_buckets=5,

      # Infer
      tgt_max_len_infer=100,
      infer_batch_size=32,
      beam_width=beam_width,
      length_penalty_weight=0.0,
      num_translations_per_input=num_translations_per_input,
      infer_mode=infer_mode,

      # Misc
      forget_bias=0.0,
      num_gpus=1,
      share_vocab=False,
      random_seed=3,
      num_keep_ckpts=5,

      # Vocab
      src_vocab_size=5,
      tgt_vocab_size=5,
      eos="eos",
      sos="sos",
      src_vocab_file="",
      tgt_vocab_file="",
      src_embed_file="",
      tgt_embed_file="",

      # For inference.py test
      subword_option="bpe",
      src="src",
      tgt="tgt",
      ctx=ctx,
      context_vector=context_vector,
      context_feed=context_feed,
      train_annotations=train_annotations,
      src_max_len=400,
      tgt_eos_id=0,
      tgt_vocab=["eos", "test1", "test2", "test3", "test4", "test5"],
      src_max_len_infer=None,
      inference_indices=inference_indices,
      metrics=["bleu"],
      steps_per_stats=10,

      # reformulator
      use_rl=False,
      entropy_regularization_weight=0.0,
      server_mode=False,
      optimize_ngrams_len=0)


def create_test_iterator(hparams, mode, trie_excludes=None):
  """Create test iterator."""
  src_vocab_table = lookup_ops.index_table_from_tensor(
      tf.constant([hparams.eos, "a", "b", "c", "d"]))
  tgt_vocab_mapping = tf.constant([hparams.sos, hparams.eos, "a", "b", "c"])
  tgt_vocab_table = lookup_ops.index_table_from_tensor(tgt_vocab_mapping)

  reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_tensor(
      tgt_vocab_mapping)

  src_dataset = tf.data.Dataset.from_tensor_slices(
      tf.constant(["a a b b c", "a b b"]))

  ctx_dataset = tf.data.Dataset.from_tensor_slices(
      tf.constant(["c b c b a", "b c b a"]))

  trie_excludes = trie_excludes or []
  trie_excludes = " {} ".format(hparams.eos).join(trie_excludes)
  tex_dataset = tf.data.Dataset.from_tensor_slices(
      tf.constant([trie_excludes, trie_excludes]))

  if mode != tf.contrib.learn.ModeKeys.INFER:
    tgt_dataset = tf.data.Dataset.from_tensor_slices(
        tf.constant(["a b c b c", "a b c b"]))
    return (iterator_utils.get_iterator(
        hparams=hparams,
        src_dataset=src_dataset,
        tgt_dataset=tgt_dataset,
        ctx_dataset=ctx_dataset,
        annot_dataset=None,
        src_vocab_table=src_vocab_table,
        tgt_vocab_table=tgt_vocab_table,
        batch_size=hparams.batch_size,
        sos=hparams.sos,
        eos=hparams.eos,
        random_seed=hparams.random_seed,
        num_buckets=hparams.num_buckets), src_vocab_table, tgt_vocab_table,
            reverse_tgt_vocab_table)
  else:
    return (iterator_utils.get_infer_iterator(
        hparams=hparams,
        src_dataset=src_dataset,
        ctx_dataset=ctx_dataset,
        annot_dataset=None,
        trie_exclude_dataset=tex_dataset,
        src_vocab_table=src_vocab_table,
        tgt_vocab_table=tgt_vocab_table,
        eos=hparams.eos,
        batch_size=hparams.batch_size), src_vocab_table, tgt_vocab_table,
            reverse_tgt_vocab_table)
