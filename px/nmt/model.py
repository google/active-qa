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

"""Basic sequence-to-sequence model with dynamic RNN support."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import inspect
from enum import Enum


import numpy as np
import tensorflow as tf

from px.nmt import environment_client
from px.nmt import context_encoder
from px.nmt import model_helper
from px.nmt.utils import diverse_decoder_utils
from px.nmt.utils import iterator_utils
from px.nmt.utils import loss_utils
from px.nmt.utils import trie_decoder_utils
from px.nmt.optimistic_restore_saver import OptimisticRestoreSaver

from tensorflow.python.layers import core as layers_core
from third_party.nmt.utils import misc_utils as utils

utils.check_tensorflow_version()

__all__ = ['BaseModel', 'Model']


class DecoderType(Enum):
  TRAINING = 0
  SAMPLE = 1
  GREEDY = 2
  BEAM_SEARCH = 3
  TRIE_BEAM_SEARCH = 4
  TRIE_SAMPLE = 5
  TRIE_GREEDY = 6
  DIVERSE_BEAM_SEARCH = 7


class BaseModel(object):
  """Sequence-to-sequence base class.
  """

  def __init__(self,
               hparams,
               mode,
               iterator,
               source_vocab_table,
               target_vocab_table,
               reverse_target_vocab_table=None,
               scope=None,
               extra_args=None,
               trie=None):
    """Create the model.

    Args:
      hparams: Hyperparameter configurations.
      mode: TRAIN | EVAL | INFER
      iterator: Dataset Iterator that feeds data.
      source_vocab_table: Lookup table mapping source words to ids.
      target_vocab_table: Lookup table mapping target words to ids.
      reverse_target_vocab_table: Lookup table mapping ids to target words. Only
        required in INFER mode. Defaults to None.
      scope: scope of the model.
      extra_args: model_helper.ExtraArgs, for passing customizable functions.
      trie: pygtrie.Trie to decode into

    """
    assert isinstance(iterator, iterator_utils.BatchedInput)
    self.iterator = iterator
    self.mode = mode
    self.src_vocab_table = source_vocab_table
    self.tgt_vocab_table = target_vocab_table

    self.src_vocab_size = hparams.src_vocab_size
    self.tgt_vocab_size = hparams.tgt_vocab_size
    self.num_gpus = hparams.num_gpus
    self.reverse_target_vocab_table = reverse_target_vocab_table
    # extra_args: to make it flexible for adding external customizable code
    self.single_cell_fn = None
    if extra_args:
      self.single_cell_fn = extra_args.single_cell_fn

    # Set num layers
    self.num_encoder_layers = hparams.num_encoder_layers
    self.num_decoder_layers = hparams.num_decoder_layers
    assert self.num_encoder_layers
    assert self.num_decoder_layers

    self.trie = trie

    # Set num residual layers
    if hasattr(hparams, 'num_residual_layers'):  # compatible common_test_utils
      self.num_encoder_residual_layers = hparams.num_residual_layers
      self.num_decoder_residual_layers = hparams.num_residual_layers
    else:
      self.num_encoder_residual_layers = hparams.num_encoder_residual_layers
      self.num_decoder_residual_layers = hparams.num_decoder_residual_layers

    # Initializer
    initializer = model_helper.get_initializer(
        hparams.init_op, hparams.random_seed, hparams.init_weight)
    tf.get_variable_scope().set_initializer(initializer)

    # Embeddings
    self.init_embeddings(hparams, scope)
    self.batch_size = tf.size(self.iterator.source_sequence_length)

    # Projection
    with tf.variable_scope(scope or 'build_network', reuse=tf.AUTO_REUSE):
      with tf.variable_scope('decoder/output_projection'):
        self.output_layer = layers_core.Dense(
            hparams.tgt_vocab_size, use_bias=False, name='output_projection')

    if hparams.use_rl:
      # Create environment function
      self._environment_reward_fn = (
          environment_client.make_environment_reward_fn(
              hparams.environment_server, mode=hparams.environment_mode))

    ## Train graph
    res = self.build_graph(hparams, scope=scope)

    (self.loss, self.rewards, self.logits, self.final_context_state,
     self.sample_id, self.sample_words, self.sample_strings,
     train_summaries) = res
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      self.word_count = tf.reduce_sum(
          self.iterator.source_sequence_length) + tf.reduce_sum(
              self.iterator.target_sequence_length)

    if self.mode != tf.contrib.learn.ModeKeys.INFER:
      ## Count the number of predicted words for compute ppl.
      self.predict_count = tf.reduce_sum(self.iterator.target_sequence_length)

    self.global_step = tf.Variable(0, trainable=False)
    params = tf.trainable_variables()

    # Gradients and SGD update operation for training the model.
    # Arrage for the embedding vars to appear at the beginning.
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      self.learning_rate = tf.constant(hparams.learning_rate)
      # warm-up
      self.learning_rate = self._get_learning_rate_warmup(hparams)
      # decay
      self.learning_rate = self._get_learning_rate_decay(hparams)

      # Optimizer
      if hparams.optimizer == 'sgd':
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      elif hparams.optimizer == 'adam':
        opt = tf.train.AdamOptimizer(self.learning_rate)
      elif hparams.optimizer == 'rmsprop':
        opt = tf.train.RMSPropOptimizer(self.learning_rate)
      elif hparams.optimizer == 'adadelta':
        opt = tf.train.AdadeltaOptimizer(self.learning_rate)

      # Gradients
      gradients = tf.gradients(
          self.loss,
          params,
          colocate_gradients_with_ops=hparams.colocate_gradients_with_ops)

      (clipped_gradients, gradients_norm_summary,
       gradients_norm) = model_helper.gradient_clip(
           gradients, max_gradient_norm=hparams.max_gradient_norm)
      self.gradients_norm = gradients_norm

      self.update = opt.apply_gradients(
          zip(clipped_gradients, params), global_step=self.global_step)

      # Summary
      self.train_summary = tf.summary.merge([
          tf.summary.scalar('lr', self.learning_rate),
          tf.summary.scalar('train_loss', self.loss),
      ] + train_summaries + gradients_norm_summary)

    if self.mode == tf.contrib.learn.ModeKeys.INFER:
      self.infer_summary = self._get_infer_summary(hparams)

    # Saver
    self.saver = OptimisticRestoreSaver(
        max_to_keep=hparams.num_keep_ckpts, init_uninitialized_variables=True)

    # Print trainable variables
    utils.print_out('# Trainable variables')
    for param in params:
      utils.print_out('  {}, {}, {}'.format(param.name, param.get_shape(),
                                            param.op.device))

  def _get_learning_rate_warmup(self, hparams):
    """Get learning rate warmup."""
    warmup_steps = hparams.warmup_steps
    warmup_scheme = hparams.warmup_scheme
    utils.print_out(
        '  learning_rate={}, warmup_steps={}, warmup_scheme={}'.format(
            hparams.learning_rate, warmup_steps, warmup_scheme))

    # Apply inverse decay if global steps less than warmup steps.
    # Inspired by https://arxiv.org/pdf/1706.03762.pdf (Section 5.3)
    # When step < warmup_steps,
    #   learing_rate *= warmup_factor ** (warmup_steps - step)
    if warmup_scheme == 't2t':
      # 0.01^(1/warmup_steps): we start with a lr, 100 times smaller
      warmup_factor = tf.exp(tf.log(0.01) / warmup_steps)
      inv_decay = warmup_factor**(tf.to_float(warmup_steps - self.global_step))
    else:
      raise ValueError('Unknown warmup scheme {}'.format(warmup_scheme))

    return tf.cond(
        self.global_step < hparams.warmup_steps,
        lambda: inv_decay * self.learning_rate,
        lambda: self.learning_rate,
        name='learning_rate_warump_cond')

  def _get_learning_rate_decay(self, hparams):
    """Get learning rate decay."""
    if hparams.decay_scheme == 'luong10':
      start_decay_step = int(hparams.num_train_steps / 2)
      remain_steps = hparams.num_train_steps - start_decay_step
      decay_steps = int(remain_steps / 10)  # decay 10 times
      decay_factor = 0.5
    elif hparams.decay_scheme == 'luong234':
      start_decay_step = int(hparams.num_train_steps * 2 / 3)
      remain_steps = hparams.num_train_steps - start_decay_step
      decay_steps = int(remain_steps / 4)  # decay 4 times
      decay_factor = 0.5
    elif not hparams.decay_scheme:  # no decay
      start_decay_step = hparams.num_train_steps
      decay_steps = 0
      decay_factor = 1.0
    elif hparams.decay_scheme:
      raise ValueError('Unknown decay scheme %s' % hparams.decay_scheme)
    utils.print_out('  decay_scheme=%s, start_decay_step=%d, decay_steps %d, '
                    'decay_factor %g' % (hparams.decay_scheme, start_decay_step,
                                         decay_steps, decay_factor))

    return tf.cond(
        self.global_step < start_decay_step,
        lambda: self.learning_rate,
        lambda: tf.train.exponential_decay(
            self.learning_rate,
            (self.global_step - start_decay_step),
            decay_steps, decay_factor, staircase=True),
        name='learning_rate_decay_cond')

  def init_embeddings(self, hparams, scope):
    """Init embeddings."""
    self.embedding_encoder, self.embedding_decoder = (
        model_helper.create_emb_for_encoder_and_decoder(
            share_vocab=hparams.share_vocab,
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            src_embed_size=hparams.num_units,
            tgt_embed_size=hparams.num_units,
            num_partitions=hparams.num_embeddings_partitions,
            src_vocab_file=hparams.src_vocab_file,
            tgt_vocab_file=hparams.tgt_vocab_file,
            src_embed_file=hparams.src_embed_file,
            tgt_embed_file=hparams.tgt_embed_file,
            scope=scope,
        ))

  def train(self, sess):
    assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
    return sess.run([
        self.update, self.loss, self.rewards, self.predict_count,
        self.train_summary, self.global_step, self.word_count, self.batch_size,
        self.gradients_norm, self.learning_rate, self.sample_strings
    ])

  def eval(self, sess):
    assert self.mode == tf.contrib.learn.ModeKeys.EVAL
    return sess.run([
        self.loss, self.rewards, self.sample_words, self.predict_count,
        self.batch_size
    ])

  def build_graph(self, hparams, scope=None):
    """Subclass must implement this method.

    Creates a sequence-to-sequence model with dynamic RNN decoder API.
    Args:
      hparams: Hyperparameter configurations.
      scope: VariableScope for the created subgraph; default "dynamic_seq2seq".

    Returns:
      A tuple of the form (logits, loss, final_context_state),
      where:
        logits: float32 Tensor [batch_size x num_decoder_symbols].
        loss: the total loss / batch_size.
        final_context_state: The final state of decoder RNN.

    Raises:
      ValueError: if encoder_type differs from mono and bi, or
        attention_option is not (luong | scaled_luong |
        bahdanau | normed_bahdanau).
    """
    utils.print_out('# creating {} graph ...'.format(self.mode))
    dtype = tf.float32
    train_summaries = []

    with tf.variable_scope(
        scope or 'dynamic_seq2seq', dtype=dtype, reuse=tf.AUTO_REUSE):
      ## Context
      if hparams.ctx is not None:
        vector_size = hparams.num_units
        if (hparams.encoder_type == 'bi' and
            hparams.context_feed == 'encoder_output'):
          vector_size *= 2

        context_vector = context_encoder.get_context_vector(
            self.mode, self.iterator, hparams, vector_size=vector_size)

      ## Encoder
      encoder_outputs, encoder_state = self._build_encoder(hparams)

      ## Feed the Context
      if hparams.ctx is not None:
        encoder_outputs, encoder_state = context_encoder.feed(
            context_vector, encoder_outputs, encoder_state, hparams)

      ## Decoder
      logits, sample_ids, final_context_state = self._build_decoder(
          encoder_outputs, encoder_state, hparams)

      sample_words = self.reverse_target_vocab_table.lookup(
          tf.to_int64(sample_ids))

      # Make output shape = [batch_size, time] or [beam_width, batch_size, time]
      # when using beam search.
      sample_ids = tf.transpose(sample_ids)
      sample_words = tf.transpose(sample_words)
      sample_strings = tf.py_func(self.tokens_to_strings,
                                  [sample_words, hparams.eos], (tf.string),
                                  'TokensToStrings')

      if hparams.server_mode:
        rewards = self.iterator.weights
      elif hparams.use_rl:
        # Compute rewards when in TRAIN or EVAL mode
        iterator = self.iterator
        doc_ids = iterator.annotation

        rewards, _ = self.compute_rewards(
            questions=sample_strings, doc_ids=doc_ids)

        train_summaries.append(
            tf.summary.scalar('train_avg_reward', tf.reduce_mean(rewards)))
      else:
        # tf does only accepts tensors as returned values
        rewards = tf.constant([1.0])

      ## Loss
      if self.mode != tf.contrib.learn.ModeKeys.INFER:

        with tf.device(
            model_helper.get_device_str(self.num_encoder_layers - 1,
                                        self.num_gpus)):
          # encoder_outputs.shape = [max_time, batch_size, embeddings_dim]
          question_embeddings = tf.reduce_mean(encoder_outputs, 0)
          loss = self._compute_loss(
              hparams=hparams,
              logits=logits,
              sample_ids=sample_ids,
              sample_words=sample_words,
              rewards=rewards,
              question_embeddings=question_embeddings,
              train_summaries=train_summaries)
      else:
        loss = None

      return (loss, rewards, logits, final_context_state, sample_ids,
              sample_words, sample_strings, train_summaries)

  @abc.abstractmethod
  def _build_encoder(self, hparams):
    """Subclass must implement this.

    Build and run an RNN encoder.

    Args:
      hparams: Hyperparameters configurations.

    Returns:
      A tuple of encoder_outputs and encoder_state.
    """
    pass

  def _build_encoder_cell(self,
                          hparams,
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
        mode=self.mode,
        base_gpu=base_gpu,
        single_cell_fn=self.single_cell_fn)

  def _get_infer_maximum_iterations(self, hparams, source_sequence_length):
    """Maximum decoding steps at inference time."""
    if hparams.tgt_max_len_infer:
      maximum_iterations = hparams.tgt_max_len_infer
      utils.print_out('  decoding maximum_iterations %d' % maximum_iterations)
    else:
      decoding_length_factor = 2.0
      max_encoder_length = tf.reduce_max(source_sequence_length)
      maximum_iterations = tf.to_int32(
          tf.round(tf.to_float(max_encoder_length) * decoding_length_factor))
    return maximum_iterations

  def _build_decoder(self, encoder_outputs, encoder_state, hparams):
    """Build and run a RNN decoder with a final projection layer.

    Args:
      encoder_outputs: The outputs of encoder for every time step.
      encoder_state: The final state of the encoder.
      hparams: The Hyperparameters configurations.

    Returns:
      A tuple of final logits and final decoder state:
        logits: size [time, batch_size, vocab_size].

    Raises:
      ValueError: In case of incompatible hparams.
    """
    tgt_sos_id = tf.cast(
        self.tgt_vocab_table.lookup(tf.constant(hparams.sos)), tf.int32)
    tgt_eos_id = tf.cast(
        self.tgt_vocab_table.lookup(tf.constant(hparams.eos)), tf.int32)

    iterator = self.iterator
    source_sequence_length = iterator.source_sequence_length
    # add 1 into the source length
    if hparams.context_feed == 'encoder_output':
      if hparams.context_vector == 'bilstm_full':
        context_length = self.get_max_time(self.iterator.context)
        source_sequence_length = tf.add(context_length, source_sequence_length)
      else:
        source_sequence_length = tf.add(1, source_sequence_length)

    # maximum_iteration: The maximum decoding steps.
    maximum_iterations = self._get_infer_maximum_iterations(
        hparams, iterator.source_sequence_length)

    ## Decoder.
    with tf.variable_scope('decoder') as decoder_scope:
      cell, decoder_initial_state = self._build_decoder_cell(
          hparams, encoder_outputs, encoder_state, source_sequence_length)

      start_tokens = tf.fill([self.batch_size], tgt_sos_id)
      end_token = tgt_eos_id

      # decoder_type:
      # Train and SL: Training (0)
      # Eval and SL: Training (0)
      # Train and RL: Sample (1)
      # Eval and RL: Greedy (2)
      # INFER and no beam: Greedy (2)
      # INFER and beam: BeamSearch (3)

      if self.mode == tf.contrib.learn.ModeKeys.TRAIN and not hparams.use_rl:
        decoder_type = DecoderType.TRAINING
      elif self.mode == tf.contrib.learn.ModeKeys.EVAL and not hparams.use_rl:
        decoder_type = DecoderType.TRAINING
      elif self.mode == tf.contrib.learn.ModeKeys.TRAIN and hparams.use_rl:
        if hparams.infer_mode.startswith('trie_'):
          decoder_type = DecoderType.TRIE_SAMPLE
        else:
          decoder_type = DecoderType.SAMPLE
      elif self.mode == tf.contrib.learn.ModeKeys.EVAL and hparams.use_rl:
        if hparams.infer_mode.startswith('trie_'):
          decoder_type = DecoderType.TRIE_GREEDY
        else:
          decoder_type = DecoderType.GREEDY
      elif self.mode == tf.contrib.learn.ModeKeys.INFER:
        if hparams.infer_mode == 'greedy':
          if hparams.beam_width != 0:
            raise ValueError('Greedy decoding requires beam_width == 0')
          decoder_type = DecoderType.GREEDY
        elif hparams.infer_mode == 'trie_greedy':
          if hparams.beam_width != 0:
            raise ValueError('Trie greedy decoding requires beam_width == 0')
          decoder_type = DecoderType.TRIE_GREEDY
        elif hparams.infer_mode == 'sample':
          if hparams.beam_width != 0:
            raise ValueError('Sampling requires beam_width == 0')
          decoder_type = DecoderType.SAMPLE
        elif hparams.infer_mode == 'trie_sample':
          if hparams.beam_width != 0:
            raise ValueError('Trie sampling requires beam_width == 0')
          decoder_type = DecoderType.TRIE_SAMPLE
        elif hparams.infer_mode == 'beam_search':
          if hparams.beam_width <= 0:
            raise ValueError('Beam search requires beam_width > 0')
          decoder_type = DecoderType.BEAM_SEARCH
        elif hparams.infer_mode == 'trie_beam_search':
          if hparams.beam_width <= 0:
            raise ValueError('Trie beam search requires beam_width > 0')
          decoder_type = DecoderType.TRIE_BEAM_SEARCH
        elif hparams.infer_mode == 'diverse_beam_search':
          if hparams.beam_width <= 0:
            raise ValueError('Diverse beam search requires beam_width > 0')
          decoder_type = DecoderType.DIVERSE_BEAM_SEARCH
        else:
          raise ValueError('Invalid infer_mode: {}.'.format(hparams.infer_mode))
      else:
        raise ValueError(
            'No decoder found for the given combination of mode ({})'
            'use_rl ({}) and beam_width ({}).'.format(self.mode, self.use_rl,
                                                      hparams.beam_width))

      if decoder_type == DecoderType.TRAINING:
        # TrainingHelper
        # decoder_emp_inp: [max_time, batch_size, num_units]
        target_input = iterator.target_input
        # Make shape [max_time, batch_size].
        target_input = tf.transpose(target_input)
        decoder_emb_inp = tf.nn.embedding_lookup(self.embedding_decoder,
                                                 target_input)
        helper = tf.contrib.seq2seq.TrainingHelper(
            decoder_emb_inp, iterator.target_sequence_length, time_major=True)
        maximum_iterations = None
        output_layer = None

      elif decoder_type in [DecoderType.SAMPLE, DecoderType.TRIE_SAMPLE]:
        # Sample
        if hparams.sampling_temperature > 0:
          helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
              self.embedding_decoder,
              start_tokens,
              end_token,
              softmax_temperature=hparams.sampling_temperature,
              seed=hparams.random_seed)
        else:
          helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
              self.embedding_decoder, start_tokens, end_token)
        output_layer = self.output_layer

      elif decoder_type in [DecoderType.GREEDY, DecoderType.TRIE_GREEDY]:
        # Greedy
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            self.embedding_decoder, start_tokens, end_token)
        output_layer = self.output_layer

      if decoder_type in [
          DecoderType.TRAINING, DecoderType.SAMPLE, DecoderType.GREEDY
      ]:
        # Decoder
        my_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell, helper, decoder_initial_state,
            output_layer=output_layer)  # applied per timestep

      elif decoder_type in [DecoderType.TRIE_GREEDY, DecoderType.TRIE_SAMPLE]:

        assert self.trie is not None

        my_decoder = trie_decoder_utils.TrieSamplerDecoder(
            trie=self.trie,
            cell=cell,
            helper=helper,
            initial_state=decoder_initial_state,
            trie_exclude=self.iterator.trie_exclude,
            output_layer=output_layer)

      elif decoder_type == DecoderType.DIVERSE_BEAM_SEARCH:
        # Diverse Beam Search decoder
        my_decoder = diverse_decoder_utils.DiverseBeamSearchDecoder(
            maximum_iterations=maximum_iterations,
            decoder_scope=decoder_scope,
            decoding_iterations=hparams.diverse_beam_search_iterations,
            cell=cell,
            embedding=self.embedding_decoder,
            start_tokens=start_tokens,
            end_token=end_token,
            initial_state=decoder_initial_state,
            beam_width=hparams.beam_width,
            output_layer=self.output_layer,
            length_penalty_weight=hparams.length_penalty_weight)

      elif decoder_type == DecoderType.BEAM_SEARCH:
        # Beam Search decoder
        my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell=cell,
            embedding=self.embedding_decoder,
            start_tokens=start_tokens,
            end_token=end_token,
            initial_state=decoder_initial_state,
            beam_width=hparams.beam_width,
            output_layer=self.output_layer,
            length_penalty_weight=hparams.length_penalty_weight)

      elif decoder_type == DecoderType.TRIE_BEAM_SEARCH:
        assert self.trie is not None

        my_decoder = trie_decoder_utils.TrieBeamSearchDecoder(
            cell=cell,
            embedding=self.embedding_decoder,
            start_tokens=start_tokens,
            end_token=end_token,
            initial_state=decoder_initial_state,
            beam_width=hparams.beam_width,
            output_layer=self.output_layer,
            length_penalty_weight=hparams.length_penalty_weight,
            trie=self.trie,
            trie_exclude=self.iterator.trie_exclude)

      # Dynamic decoding
      outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
          my_decoder,
          maximum_iterations=maximum_iterations,
          output_time_major=True,
          swap_memory=True,
          scope=decoder_scope)

      if decoder_type == DecoderType.TRAINING:
        sample_id = outputs.sample_id

        # Note: there's a subtle difference here between train and inference.
        # We could have set output_layer when create my_decoder
        #   and shared more code between train and inference.
        # We chose to apply the output_layer to all timesteps for speed:
        #   10% improvements for small models & 20% for larger ones.
        # If memory is a concern, we should apply output_layer per timestep.
        logits = self.output_layer(outputs.rnn_output)

      elif decoder_type in [
          DecoderType.GREEDY, DecoderType.SAMPLE, DecoderType.TRIE_GREEDY,
          DecoderType.TRIE_SAMPLE
      ]:
        logits = outputs.rnn_output
        sample_id = outputs.sample_id

      elif decoder_type in [
          DecoderType.BEAM_SEARCH,
          DecoderType.TRIE_BEAM_SEARCH,
          DecoderType.DIVERSE_BEAM_SEARCH,
      ]:
        logits = tf.no_op()
        sample_id = outputs.predicted_ids

    return logits, sample_id, final_context_state

  def get_sequence_length(self, hparams, tokens):
    """Given an array of tokens, returns the length of each one.

    Args:
      hparams: hyperparameters object.
      tokens: a numpy array of shape=(batch_size, max_seq_len)

    Returns:
      sequence_length: a numpy array of shape=(batch_size)
    """
    batch_size = tokens.get_shape()[0].value

    def get_sequence_length_py(tokens):
      lengths = []
      for line in tokens:
        line = [token.decode('utf-8') for token in line]
        if hparams.eos in line:
          lengths.append(line.index(hparams.eos) + 1)
        else:
          lengths.append(len(line))
      return np.array(lengths, dtype=np.int32)

    sequence_length = tf.py_func(get_sequence_length_py, [tokens], tf.int32,
                                 'GetSequenceLength')
    sequence_length.set_shape([batch_size])
    return sequence_length

  def tokens_to_strings(self, all_tokens, eos):
    """Convert a numpy matrix of tokens to an array of strings

    Args:
      all_tokens: tokens as a numpy array of shape [batch_size, max_seq_len]
      eos: a string representing the end-of-sequence token.

    Returns:
      strings: a numpy array of strings of shape [batch_size]
    """

    strings = []
    for tokens in all_tokens:
      try:
        tokens = [token.decode('utf-8') for token in tokens]
        if eos in tokens:
          length = tokens.index(eos)
        else:
          length = len(tokens) + 1
        string = u''.join(tokens[:length])
      except UnicodeEncodeError as e:
        tf.logging.error(e)
        tf.logging.error(repr(tokens))
      except UnicodeDecodeError as e:
        tf.logging.error(e)
        tf.logging.error(repr(tokens))
      except TypeError as e:
        tf.logging.error(e)
        tf.logging.error(repr(tokens))

      string = string.replace(u'\u2581', u' ').lstrip()

      strings.append(string)
    return np.array(strings, dtype=np.object)

  def compute_rewards(self, questions, doc_ids):
    """Compute rewards by calling a environment server.

    Args:
      questions: an array of shape=(batch_size,) containing the question
          strings.
      doc_ids: an array of shape=(batch_size,) containing the doc ids that will
          be used by the environment server to answer the question.

    Returns:
       rewards: an array of shape=(batch_size) containing the rewards (normally,
           f1 scores of each question.
       answer: an array of shape(batch_size) containing the answer strings for
           the questions.
    """

    batch_size = questions.get_shape()[0].value

    rewards, _, answers = tf.py_func(
        self._environment_reward_fn, [questions, doc_ids],
        (tf.float32, tf.float32, tf.string), 'CallEnvironment')

    rewards.set_shape([batch_size])
    answers.set_shape([batch_size])

    return rewards, answers

  def get_max_time(self, tensor):
    return tensor.shape[0].value or tf.shape(tensor)[0]

  @abc.abstractmethod
  def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                          source_sequence_length):
    """Subclass must implement this.

    Args:
      hparams: Hyperparameters configurations.
      encoder_outputs: The outputs of encoder for every time step.
      encoder_state: The final state of the encoder.
      source_sequence_length: sequence length of encoder_outputs.

    Returns:
      A tuple of a multi-layer RNN cell used by decoder
        and the initial state of the decoder RNN.
    """
    pass

  def init_weights(self, shape):
    """Weight initialization using a normal distribution."""
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

  def _value_network(self, question_embeddings):
    """A feedforward neural net that takes as input the original question and
    outputs a prediction for the reward.

    Args:
      question_embeddings: original question embeddings of shape=
        [batch_size,embeddings_dimension]

    Returns:
      The logits of shape=[batch_size]

    """
    layer_1 = tf.add(tf.matmul(question_embeddings, self.pw_1), self.pb_1)
    layer_1 = tf.tanh(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, self.pw_2), self.pb_2)
    layer_2 = tf.sigmoid(layer_2)
    logits = tf.reshape(layer_2, shape=[-1])

    return logits

  def _compute_loss(self, hparams, logits, sample_ids, sample_words, rewards,
                    question_embeddings, train_summaries):
    """Compute optimization loss."""
    if hparams.use_rl:
      target_output = sample_ids
      sequence_length = self.get_sequence_length(hparams, sample_words)
      losses, advantages = self._compute_loss_offset_and_advantages(
          rewards=rewards,
          baseline_type=hparams.baseline_type,
          replication_factor=hparams.replication_factor,
          question_embeddings=question_embeddings,
          train_summaries=train_summaries)
    else:
      target_output = self.iterator.target_output
      sequence_length = self.iterator.target_sequence_length
      losses = 0.0
      advantages = None

    if hparams.server_mode:
      advantages = rewards

    # Make shape [max_time, batch_size].
    target_output = tf.transpose(target_output, name='transpose_targets')
    losses += loss_utils.cross_entropy_sequence_loss(
        logits=logits,
        targets=target_output,
        sequence_length=sequence_length,
        weights=advantages)

    if hparams.entropy_regularization_weight > 0:
      entropy_reg = loss_utils.entropy_regularization(
          logits=logits, sequence_length=sequence_length
      ) * hparams.entropy_regularization_weight
      losses -= entropy_reg
      train_summaries.append(
          tf.summary.histogram('Entropy_Regularization',
                               tf.to_float(entropy_reg)))

    average_length = tf.reduce_mean(tf.to_float(sequence_length))
    train_summaries.append(tf.summary.scalar('AverageLength', average_length))

    # In RL training, normalize the loss by sequence length, otherwise by batch.
    if hparams.use_rl or hparams.server_mode:
      losses /= tf.to_float(sequence_length + 1)
      loss = tf.reduce_sum(losses)
    else:
      loss = tf.reduce_sum(losses) / tf.to_float(self.batch_size)
    return loss

  def _compute_loss_offset_and_advantages(self,
                                          rewards,
                                          baseline_type=2,
                                          replication_factor=1,
                                          question_embeddings=None,
                                          train_summaries=None):
    """Compute the baseline and apply it to the rewards.

    Args:
      rewards: A tensor of shape [B].
      baseline_type: int that selects the baseline to apply; possible values
        0: Computes the baseline as average reward per source.
        1: Computes the baseline using a value network.
        2: This baseline is just the average reward over the whole batch. So
           this baseline may combine averages of different sources (depending on
           replication).
        Otherwise: ignore baseline and return rewards as they are.
      replication_factor: int; how many copies of each input to expect.
      question_embeddings: A tensor of embeddings used for the value network.
      train_summaries: this is passed around to collect summaries for
                       TensorBoard.

    Returns:
      A tuple consisting of:
        losses: The losses of the value network (a tensor of shape [B]) when the
                value network is used; otherwise 0.0.
        rewards: A tensor of shape [B] that contains rewards with baseline
                 subtracted.
    """
    losses = 0.0
    if baseline_type == 0:
      baseline = tf.reshape(rewards, [-1, replication_factor])
      baseline = tf.reduce_mean(baseline, 1, keep_dims=True)
      baseline = tf.tile(baseline, [1, replication_factor])
      baseline = tf.reshape(baseline, [-1])
    elif baseline_type == 1:
      # Initialize the weights of the value network
      n_dim = question_embeddings.get_shape()[1].value
      self.pw_1 = self.init_weights((n_dim, n_dim))
      self.pb_1 = self.init_weights((n_dim,))
      self.pw_2 = self.init_weights((n_dim, 1))
      self.pb_2 = self.init_weights((1,))

      # The gradient here is stopped because we do not want a degenerated
      # cooperation between policy and value networks in which they always
      # produce predicted and real rewards that are close to zero.
      # We might want to remove this in the future but leaving it here
      # normally leads to a more stable training.
      question_embeddings = tf.stop_gradient(question_embeddings)
      baseline = self._value_network(question_embeddings)

      losses_baseline = tf.pow(rewards - baseline, 2)
      losses += losses_baseline
      if train_summaries:
        train_summaries.append(
            tf.summary.histogram('Losses_Baseline',
                                 tf.to_float(losses_baseline)))
    elif baseline_type == 2:
      # baseline has shape [].
      baseline = tf.reduce_mean(rewards)
    else:
      baseline = 0.0

    return losses, rewards - baseline

  def _get_infer_summary(self, hparams):
    return tf.no_op()

  def infer(self, sess):
    assert self.mode == tf.contrib.learn.ModeKeys.INFER

    return sess.run([
        self.logits, self.infer_summary, self.sample_id, self.sample_words,
        self.rewards
    ])


class Model(BaseModel):
  """Sequence-to-sequence dynamic model.

  This class implements a multi-layer recurrent neural network as encoder,
  and a multi-layer recurrent neural network decoder.
  """

  def _build_encoder(self, hparams):
    """Build an encoder."""
    num_layers = self.num_encoder_layers
    num_residual_layers = self.num_encoder_residual_layers

    iterator = self.iterator

    source = iterator.source
    # Make shape [max_time, batch_size].
    source = tf.transpose(source)

    with tf.variable_scope('encoder') as scope:
      dtype = scope.dtype
      # Look up embedding, emp_inp: [max_time, batch_size, num_units]
      encoder_emb_inp = tf.nn.embedding_lookup(self.embedding_encoder, source)

      # encoder_outputs: [max_time, batch_size, num_units]
      if hparams.encoder_type == 'uni':
        utils.print_out('  num_layers={}, num_residual_layers={}'.format(
            num_layers, num_residual_layers))
        cell = self._build_encoder_cell(hparams, num_layers,
                                        num_residual_layers)

        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            cell,
            encoder_emb_inp,
            dtype=dtype,
            sequence_length=iterator.source_sequence_length,
            time_major=True,
            swap_memory=True)
      elif hparams.encoder_type == 'bi':
        num_bi_layers = int(num_layers / 2)
        num_bi_residual_layers = int(num_residual_layers / 2)
        utils.print_out('  num_bi_layers={}, num_bi_residual_layers={}'.format(
            num_bi_layers, num_bi_residual_layers))

        encoder_outputs, bi_encoder_state = (
            self._build_bidirectional_rnn(
                inputs=encoder_emb_inp,
                sequence_length=iterator.source_sequence_length,
                dtype=dtype,
                hparams=hparams,
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
      else:
        raise ValueError('Unknown encoder_type {}'.format(hparams.encoder_type))
    return encoder_outputs, encoder_state

  def _build_bidirectional_rnn(self,
                               inputs,
                               sequence_length,
                               dtype,
                               hparams,
                               num_bi_layers,
                               num_bi_residual_layers,
                               base_gpu=0):
    """Create and call biddirectional RNN cells.

    Args:
      num_residual_layers: Number of residual layers from top to bottom. For
        example, if `num_bi_layers=4` and `num_residual_layers=2`, the last 2
        RNN layers in each RNN cell will be wrapped with `ResidualWrapper`.
      base_gpu: The gpu device id to use for the first forward RNN layer. The
        i-th forward RNN layer will use `(base_gpu + i) % num_gpus` as its
        device id. The `base_gpu` for backward RNN cell is `(base_gpu +
        num_bi_layers)`.

    Returns:
      The concatenated bidirectional output and the bidirectional RNN cell"s
      state.
    """
    # Construct forward and backward cells
    fw_cell = self._build_encoder_cell(
        hparams, num_bi_layers, num_bi_residual_layers, base_gpu=base_gpu)
    bw_cell = self._build_encoder_cell(
        hparams,
        num_bi_layers,
        num_bi_residual_layers,
        base_gpu=(base_gpu + num_bi_layers))

    bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
        fw_cell,
        bw_cell,
        inputs,
        dtype=dtype,
        sequence_length=sequence_length,
        time_major=True,
        swap_memory=True)

    return tf.concat(bi_outputs, -1), bi_state

  def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                          source_sequence_length):
    """Build an RNN cell that can be used by decoder."""
    # We only make use of encoder_outputs in attention-based models
    if hparams.attention:
      raise ValueError('BasicModel does not support attention.')

    cell = model_helper.create_rnn_cell(
        unit_type=hparams.unit_type,
        num_units=hparams.num_units,
        num_layers=self.num_decoder_layers,
        num_residual_layers=self.num_decoder_residual_layers,
        forget_bias=hparams.forget_bias,
        dropout=hparams.dropout,
        num_gpus=self.num_gpus,
        mode=self.mode,
        single_cell_fn=self.single_cell_fn)

    # For beam search, we need to replicate encoder infos beam_width times
    if self.mode == tf.contrib.learn.ModeKeys.INFER and hparams.beam_width > 0:
      decoder_initial_state = tf.contrib.seq2seq.tile_batch(
          encoder_state, multiplier=hparams.beam_width)
    else:
      decoder_initial_state = encoder_state

    return cell, decoder_initial_state
