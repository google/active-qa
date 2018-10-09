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
"""Utilities for decoding a seq2seq model into a trie."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import pygtrie
import pickle
import tensorflow as tf


import sentencepiece as sentencepiece_processor

from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.contrib.seq2seq.python.ops import basic_decoder
from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.util import nest

from px.proto import aqa_pb2

__all__ = ['DecoderTrie', 'TrieSamplerDecoder', 'TrieBeamSearchDecoder']


class TrieSamplerState(
    collections.namedtuple('TrieSamplerState',
                           ('cell_state', 'trie_keys', 'trie_exclude'))):
  pass


class TrieSamplerAttentionState(
    collections.namedtuple(
        'TrieSamplerAttentionState',
        ('cell_state', 'attention', 'time', 'alignments', 'alignment_history',
         'attention_state', 'trie_keys', 'trie_exclude')),
    attention_wrapper.AttentionWrapperState):
  pass


class TrieBeamSearchDecoderState(
    collections.namedtuple('TrieBeamSearchDecoderState',
                           ('cell_state', 'log_probs', 'finished', 'lengths',
                            'trie_keys', 'trie_exclude'))):
  pass


class DecoderTrie(pygtrie.StringTrie):
  """A wrapper around a pygtrie.StringTrie."""

  def __init__(self,
               vocab_path,
               eos_token='</s>',
               unk_token='<unk>',
               subword_option='',
               subword_model=None,
               prefix='',
               optimize_ngrams_len=0):
    """Constructor.

    Args:
      vocab_path: Path to the vocab file.
      eos_token: The EOS token (must be in the vocab).
      unk_token: The UNK token (defaults to '<unk>').
      subword_option: Subword splitting option.
      subword_model: Model for subword_option.
      prefix: Prefix that should be removed from the question.
      optimize_ngrams_len: how many ngrams to optimize over at the beginning.
    """
    super(DecoderTrie, self).__init__(separator=' ')

    self.vocab_path = vocab_path
    self._create_vocab()

    assert bool(eos_token), 'must supply a valid EOS token'

    self.eos_token = eos_token
    if eos_token and eos_token in self.vocab:
      self.eos_idx = str(self.vocab.get(eos_token))
    else:
      self.eos_idx = None

    self.unk_token = unk_token
    self.unk_idx = self.vocab.get(unk_token, '0')

    self.prefix = prefix

    self.subword_option = subword_option
    self.subword_model = subword_model
    self._create_sentencepiece_processor()
    self.optimize_ngrams_len = optimize_ngrams_len
    self._create_start_words()

  def _create_vocab(self):
    if not tf.gfile.Exists(self.vocab_path):
      raise ValueError('Provided vocab_path does not exist.')

    tf.logging.info('loading vocab for trie')
    with tf.gfile.Open(self.vocab_path) as vocab_file:
      vocab = dict((token.strip(), str(i))
                   for i, token in enumerate(vocab_file)
                   if len(token.strip()))
    self.vocab = vocab

  def _add_to_start_words(self, key):
    if not self.optimize_ngrams_len:
      return
    key_parts = key.split(' ', self.optimize_ngrams_len)
    self.start_idxs[pygtrie._SENTINEL].add(int(key_parts[0]))
    for i in range(1, self.optimize_ngrams_len):
      if i < len(key_parts):
        ngram = ' '.join(key_parts[:i])
        if ngram not in self.start_idxs:
          self.start_idxs[ngram] = set()
        self.start_idxs[ngram].add(int(key_parts[i]))

  def _create_start_words(self):
    self.start_idxs = {pygtrie._SENTINEL: set()}

    for key in self.iterkeys(shallow=False):
      self._add_to_start_words(key)

  def _create_sentencepiece_processor(self):
    if self.subword_option == 'spm':
      assert self.subword_model is not None, 'need subword_model for spm'
      self.sentpiece = sentencepiece_processor.SentencePieceProcessor()
      self.sentpiece.Load(self.subword_model.encode('utf-8'))
    else:
      self.sentpiece = None

  def __getstate__(self):
    d = dict(self.__dict__)
    del d['vocab']
    del d['sentpiece']
    del d['start_idxs']
    return d

  def __setstate__(self, state):
    self.__dict__.update(state)
    self._create_vocab()
    self._create_sentencepiece_processor()
    self._create_start_words()

  def insert_question(self, question, answers):
    if not question or not answers or len(answers) == 0:
      return
    if question.startswith(self.prefix):
      # removing prefix
      question = question[len(self.prefix):]
    if self.subword_option == 'spm':
      qpieces = self.sentpiece.EncodeAsPieces(question.encode('utf-8'))
    elif self.subword_option == '':
      qpieces = question.split()
    else:
      raise ValueError('subword_option {} unknown'.format(self.subword_option))
    if self.eos_token and len(qpieces) and qpieces[-1] != self.eos_token:
      # we have to make sure the EOS token is at the end otherwise the
      # decoding would not stop
      qpieces.append(self.eos_token)
    idxs = [self.vocab.get(t, self.unk_idx) for t in qpieces]
    key = ' '.join(idxs)
    self[key] = answers

    self._add_to_start_words(key)

  def save_to_file(self, file_path):
    with tf.gfile.Open(file_path, 'wb') as trie_file:
      pickle.dump(self, trie_file)

  @classmethod
  def load_from_file(cls, file_path):
    if not tf.gfile.Exists(file_path):
      raise ValueError('file_path {} does not exist'.format(file_path))
    with tf.gfile.Open(file_path, 'rb') as trie_file:
      trie = pickle.load(trie_file)
    return trie


  def populate_from_text_file(self, text_file_path):
    """Populates the trie from a text file.

    Args:
      text_file_path: Path to the text file.

    Raises:
      ValueError: If the given files do not exist.
    """
    if not tf.gfile.Exists(text_file_path):
      raise ValueError('Provided text_file_path does not exist.')

    # this is mostly for debugging, we'll add the question itself as its answer
    with tf.gfile.Open(text_file_path) as trie_file:
      for line in trie_file:
        l = line.strip()
        self.insert_question(l, [l])

    tf.logging.info('done populating trie')


def _is_attention_state(state):
  return isinstance(state, attention_wrapper.AttentionWrapperState)


def _is_gnmt_state(state):
  # the exact typecheck for tuple here is necessary, because we want to exclude
  # namedtuples
  return type(state) is tuple and len(state) > 0 and _is_attention_state(
      state[0])


class TrieSamplerDecoder(basic_decoder.BasicDecoder):
  """Trie sampling decoder."""

  def __init__(self, trie, trie_exclude, *args, **kwargs):
    super(TrieSamplerDecoder, self).__init__(*args, **kwargs)
    self.trie = trie
    self._trie_exclude = trie_exclude

  def initialize(self, name=None):
    """Initialize the decoder.

    Args:
      name: Name scope for any created operations.

    Returns:
      `(finished, first_inputs, initial_state)`.
    """
    state_tensor = self._initial_state
    while not isinstance(state_tensor, tf.Tensor):
      state_tensor = state_tensor[0]
    batch_size = tf.shape(state_tensor)[0]
    trie_keys = tf.py_func(
        _init_trie_keys_py_func(beam_search=False), [batch_size],
        tf.string,
        stateful=False)
    trie_keys.set_shape((None,))
    initial_state = self._initial_state
    if _is_attention_state(initial_state):
      initial_state = TrieSamplerAttentionState(
          *initial_state, trie_keys=trie_keys, trie_exclude=self._trie_exclude)
    elif _is_gnmt_state(initial_state):
      initial_state = (TrieSamplerAttentionState(
          *initial_state[0],
          trie_keys=trie_keys,
          trie_exclude=self._trie_exclude),) + initial_state[1:]
    else:
      initial_state = TrieSamplerState(
          cell_state=initial_state,
          trie_keys=trie_keys,
          trie_exclude=self._trie_exclude)
    return self._helper.initialize() + (initial_state,)

  def step(self, time, inputs, state, name=None):
    """Perform a decoding step.

    Args:
      time: scalar `int32` tensor.
      inputs: A (structure of) input tensors.
      state: A (structure of) state tensors and TensorArrays.
      name: Name scope for any created operations.

    Returns:
      `(outputs, next_state, next_inputs, finished)`.
    """
    with ops.name_scope(name, 'TrieSamplerDecoderStep', (time, inputs, state)):
      if _is_attention_state(state):
        cell_outputs, cell_state = self._cell(inputs, state)
        state_trie_keys = state.trie_keys
        state_trie_exclude = state.trie_exclude
      elif _is_gnmt_state(state):
        cell_outputs, cell_state = self._cell(inputs, state)
        state_trie_keys = state[0].trie_keys
        state_trie_exclude = state[0].trie_exclude
      else:
        cell_outputs, cell_state = self._cell(inputs, state.cell_state)
        state_trie_keys = state.trie_keys
        state_trie_exclude = state.trie_exclude
      if self._output_layer is not None:
        cell_outputs = self._output_layer(cell_outputs)

      cell_outputs_shape = cell_outputs.get_shape()
      cell_outputs = tf.py_func(
          _trie_scores_py_func(self.trie, beam_search=False),
          [cell_outputs, state_trie_keys, state_trie_exclude],
          tf.float32,
          stateful=False)
      cell_outputs.set_shape(cell_outputs_shape)

      sample_ids = self._helper.sample(
          time=time, outputs=cell_outputs, state=cell_state)
      (finished, next_inputs, next_cell_state) = self._helper.next_inputs(
          time=time,
          outputs=cell_outputs,
          state=cell_state,
          sample_ids=sample_ids)

    trie_keys = tf.py_func(
        _amend_trie_keys_py_func(beam_search=False),
        [state_trie_keys, sample_ids],
        tf.string,
        stateful=False)
    trie_keys.set_shape(state_trie_keys.get_shape())

    if _is_attention_state(next_cell_state):
      next_state = TrieSamplerAttentionState(
          *next_cell_state,
          trie_keys=trie_keys,
          trie_exclude=state_trie_exclude)
    elif _is_gnmt_state(next_cell_state):
      next_state = (TrieSamplerAttentionState(
          *next_cell_state[0],
          trie_keys=trie_keys,
          trie_exclude=state_trie_exclude),) + next_cell_state[1:]
    else:
      next_state = TrieSamplerState(
          cell_state=next_cell_state,
          trie_keys=trie_keys,
          trie_exclude=state_trie_exclude)

    outputs = basic_decoder.BasicDecoderOutput(cell_outputs, sample_ids)
    return outputs, next_state, next_inputs, finished


class TrieBeamSearchDecoder(beam_search_decoder.BeamSearchDecoder):
  """Decoder for doing trie based beam search."""

  def __init__(self, trie, trie_exclude, *args, **kwargs):
    super(TrieBeamSearchDecoder, self).__init__(*args, **kwargs)
    self.trie = trie
    self._trie_exclude = trie_exclude

  def initialize(self, name=None):
    """Initialize the decoder.

    Args:
      name: Name scope for any created operations.

    Returns:
      `(finished, start_inputs, initial_state)`.
    """
    finished, start_inputs = self._finished, self._start_inputs

    log_probs = array_ops.one_hot(  # shape(batch_sz, beam_sz)
        array_ops.zeros([self._batch_size], dtype=dtypes.int32),
        depth=self._beam_width,
        on_value=0.0,
        off_value=-np.Inf,
        dtype=nest.flatten(self._initial_cell_state)[0].dtype)

    initial_state = TrieBeamSearchDecoderState(
        cell_state=self._initial_cell_state,
        log_probs=log_probs,
        finished=finished,
        lengths=array_ops.zeros([self._batch_size, self._beam_width],
                                dtype=dtypes.int64),
        trie_keys=tf.py_func(
            _init_trie_keys_py_func(beam_search=True),
            [self._batch_size, self._beam_width],
            tf.string,
            stateful=False),
        trie_exclude=self._trie_exclude)

    return finished, start_inputs, initial_state

  def step(self, time, inputs, state, name=None):
    """Perform a decoding step.

    Args:
      time: scalar `int32` tensor.
      inputs: A (structure of) input tensors.
      state: A (structure of) state tensors and TensorArrays.
      name: Name scope for any created operations.

    Returns:
      `(outputs, next_state, next_inputs, finished)`.
    """
    batch_size = self._batch_size
    beam_width = self._beam_width
    end_token = self._end_token
    length_penalty_weight = self._length_penalty_weight

    with ops.name_scope(name, 'TrieBeamSearchDecoderStep',
                        (time, inputs, state)):
      cell_state = state.cell_state
      inputs = nest.map_structure(
          lambda inp: self._merge_batch_beams(inp, s=inp.shape[2:]), inputs)
      cell_state = nest.map_structure(self._maybe_merge_batch_beams, cell_state,
                                      self._cell.state_size)
      cell_outputs, next_cell_state = self._cell(inputs, cell_state)
      cell_outputs = nest.map_structure(
          lambda out: self._split_batch_beams(out, out.shape[1:]), cell_outputs)
      next_cell_state = nest.map_structure(
          self._maybe_split_batch_beams, next_cell_state, self._cell.state_size)

      if self._output_layer is not None:
        cell_outputs = self._output_layer(cell_outputs)

      beam_search_output, beam_search_state = _trie_beam_search_step(
          time=time,
          logits=cell_outputs,
          next_cell_state=next_cell_state,
          beam_state=state,
          batch_size=batch_size,
          beam_width=beam_width,
          end_token=end_token,
          length_penalty_weight=length_penalty_weight,
          trie=self.trie)

      finished = beam_search_state.finished
      sample_ids = beam_search_output.predicted_ids
      next_inputs = control_flow_ops.cond(
          math_ops.reduce_all(finished), lambda: self._start_inputs,
          lambda: self._embedding_fn(sample_ids))

    return beam_search_output, beam_search_state, next_inputs, finished


def _trie_beam_search_step(time, logits, next_cell_state, beam_state,
                           batch_size, beam_width, end_token,
                           length_penalty_weight, trie):
  """Performs a single step of Trie Search Decoding.

  Args:
    time: Beam search time step, should start at 0. At time 0 we assume that all
      beams are equal and consider only the first beam for continuations.
    logits: Logits at the current time step. A tensor of shape `[batch_size,
      beam_width, vocab_size]`
    next_cell_state: The next state from the cell, e.g. an instance of
      AttentionWrapperState if the cell is attentional.
    beam_state: Current state of the beam search. An instance of
      `BeamSearchDecoderState`.
    batch_size: The batch size for this input.
    beam_width: Python int.  The size of the beams.
    end_token: The int32 end token.
    length_penalty_weight: Float weight to penalize length. Disabled with 0.0.
    trie: The trie to decode into.

  Returns:
    A new beam state.
  """
  static_batch_size = tensor_util.constant_value(batch_size)

  # Calculate the current lengths of the predictions
  prediction_lengths = beam_state.lengths
  previously_finished = beam_state.finished

  # Calculate the total log probs for the new hypotheses
  # Final Shape: [batch_size, beam_width, vocab_size]
  step_log_probs = nn_ops.log_softmax(logits)
  step_log_probs = _mask_probs(step_log_probs, end_token, previously_finished)
  total_probs = array_ops.expand_dims(beam_state.log_probs, 2) + step_log_probs

  # Calculate the continuation lengths by adding to all continuing beams.
  vocab_size = logits.shape[-1].value or array_ops.shape(logits)[-1]
  lengths_to_add = array_ops.one_hot(
      indices=array_ops.fill([batch_size, beam_width], end_token),
      depth=vocab_size,
      on_value=np.int64(0),
      off_value=np.int64(1),
      dtype=dtypes.int64)
  add_mask = math_ops.to_int64(math_ops.logical_not(previously_finished))
  lengths_to_add *= array_ops.expand_dims(add_mask, 2)
  new_prediction_lengths = (
      lengths_to_add + array_ops.expand_dims(prediction_lengths, 2))

  # Calculate the scores for each beam
  scores = _get_trie_scores(
      log_probs=total_probs,
      sequence_lengths=new_prediction_lengths,
      length_penalty_weight=length_penalty_weight,
      trie_keys=beam_state.trie_keys,
      trie_exclude=beam_state.trie_exclude,
      trie=trie)

  time = ops.convert_to_tensor(time, name='time')
  # During the first time step we only consider the initial beam
  scores_flat = array_ops.reshape(scores, [batch_size, -1])

  # Pick the next beams according to the specified successors function
  next_beam_size = ops.convert_to_tensor(
      beam_width, dtype=dtypes.int32, name='beam_width')
  next_beam_scores, word_indices = nn_ops.top_k(scores_flat, k=next_beam_size)

  next_beam_scores.set_shape([static_batch_size, beam_width])
  word_indices.set_shape([static_batch_size, beam_width])

  # Pick out the probs, beam_ids, and states according to the chosen predictions
  next_beam_probs = _tensor_gather_helper(
      gather_indices=word_indices,
      gather_from=total_probs,
      batch_size=batch_size,
      range_size=beam_width * vocab_size,
      gather_shape=[-1],
      name='next_beam_probs')
  # Note: just doing the following
  #   math_ops.to_int32(word_indices % vocab_size,
  #       name="next_beam_word_ids")
  # would be a lot cleaner but for reasons unclear, that hides the results of
  # the op which prevents capturing it with tfdbg debug ops.
  raw_next_word_ids = math_ops.mod(
      word_indices, vocab_size, name='next_beam_word_ids')
  next_word_ids = math_ops.to_int32(raw_next_word_ids)
  next_beam_ids = math_ops.to_int32(
      word_indices / vocab_size, name='next_beam_parent_ids')

  # Append new ids to current predictions
  previously_finished = _tensor_gather_helper(
      gather_indices=next_beam_ids,
      gather_from=previously_finished,
      batch_size=batch_size,
      range_size=beam_width,
      gather_shape=[-1])
  next_finished = math_ops.logical_or(
      previously_finished,
      math_ops.equal(next_word_ids, end_token),
      name='next_beam_finished')

  # Calculate the length of the next predictions.
  # 1. Finished beams remain unchanged.
  # 2. Beams that are now finished (EOS predicted) have their length
  #    increased by 1.
  # 3. Beams that are not yet finished have their length increased by 1.
  lengths_to_add = math_ops.to_int64(math_ops.logical_not(previously_finished))
  next_prediction_len = _tensor_gather_helper(
      gather_indices=next_beam_ids,
      gather_from=beam_state.lengths,
      batch_size=batch_size,
      range_size=beam_width,
      gather_shape=[-1])
  next_prediction_len += lengths_to_add

  # Pick out the cell_states according to the next_beam_ids. We use a
  # different gather_shape here because the cell_state tensors, i.e.
  # the tensors that would be gathered from, all have dimension
  # greater than two and we need to preserve those dimensions.
  # pylint: disable=g-long-lambda
  next_cell_state = nest.map_structure(
      lambda gather_from: _maybe_tensor_gather_helper(
          gather_indices=next_beam_ids,
          gather_from=gather_from,
          batch_size=batch_size,
          range_size=beam_width,
          gather_shape=[batch_size * beam_width, -1]),
      next_cell_state)
  # pylint: enable=g-long-lambda

  trie_keys = tf.py_func(
      _amend_trie_keys_py_func(beam_search=True),
      [beam_state.trie_keys, next_word_ids, next_beam_ids],
      tf.string,
      stateful=False)
  trie_keys.set_shape(beam_state.trie_keys.get_shape())

  next_state = TrieBeamSearchDecoderState(
      cell_state=next_cell_state,
      log_probs=next_beam_probs,
      lengths=next_prediction_len,
      finished=next_finished,
      trie_keys=trie_keys,
      trie_exclude=beam_state.trie_exclude)

  output = beam_search_decoder.BeamSearchDecoderOutput(
      scores=next_beam_scores,
      predicted_ids=next_word_ids,
      parent_ids=next_beam_ids)

  return output, next_state


def _init_trie_keys_py_func(beam_search=False):
  """Initialize trie keys for decoding.

  Args:
    beam_search: For use in beam search decoding.

  Returns:
    A function that initializes the keys for the decoding state.
  """
  if beam_search:

    def _py_func(batch_size, beam_size):
      trie_keys = np.array([[u'-1'] * beam_size] * batch_size).astype('U')
      return trie_keys
  else:

    def _py_func(batch_size):
      trie_keys = np.array([u'-1'] * batch_size).astype('U')
      return trie_keys

  return _py_func


def _amend_trie_keys_py_func(beam_search=False):
  """Amend the keys after the decoder has made a choice.

  Args:
    beam_search: For use in beam search decoding.

  Returns:
    A function that amends the keys given the chosen next words.
  """
  if beam_search:

    def _py_func(trie_keys, next_word_ids, next_beam_ids):
      idxs = np.tile(
          np.arange(trie_keys.shape[0])[:, np.newaxis], (1, trie_keys.shape[1]))
      trie_keys = trie_keys[idxs.ravel(), next_beam_ids.ravel()].reshape(
          trie_keys.shape)
      trie_keys += u' '
      trie_keys += next_word_ids.astype('U')
      return trie_keys
  else:

    def _py_func(trie_keys, next_word_ids):
      trie_keys += u' '
      trie_keys += next_word_ids.astype('U')
      return trie_keys

  return _py_func


def _clean_trie_key(trie, trie_key):
  parts = trie_key.split()
  if len(parts):
    parts = parts[1:]
  try:
    eos_i = parts.index(trie.eos_idx)
    parts = parts[:eos_i + 1]
  except ValueError:
    pass
  return ' '.join(parts)


def _trie_scores_py_func(trie, beam_search=False):
  """Modify the sampling probabilities according to trie.

  Args:
    trie: The trie to decode into.
    beam_search: For use in beam search decoding.

  Returns:
    A function that modifies the given sampling scores, such that the decoder
    will only decode into the given trie.
  """
  if beam_search:

    def _py_func(log_probs, sequence_lengths, scores, trie_keys, trie_exclude):
      batch_size, beam_size, vocab_size = scores.shape
      masked_scores = np.ma.masked_invalid(scores)
      mask = np.zeros(scores.shape, dtype=np.bool_)
      for batch_idx in range(batch_size):

        trie_exclude_batch_set = _create_trie_exclude_batch_set(
            trie, trie_exclude[batch_idx])

        for beam_idx in range(beam_size):
          trie_key = trie_keys[batch_idx, beam_idx]
          trie_key = _clean_trie_key(trie, trie_key)
          if not len(trie_key):
            trie_key = pygtrie._SENTINEL
          elif not trie.has_node(trie_key):
            # This might happen if the beam is larger than the trie.
            continue

          subtrie_idxs = _get_valid_continuation_idxs(
              trie, trie_exclude_batch_set, trie_key)

          if len(subtrie_idxs):
            mask[batch_idx, beam_idx, np.array(list(subtrie_idxs))] = True

      in_trie_scores = masked_scores[mask]
      out_of_trie_scores = masked_scores[~mask]
      if len(in_trie_scores) == 0 or len(out_of_trie_scores) == 0:
        # no unfinished beam is in trie or no beam is out of trie
        return scores

      min_in = np.min(in_trie_scores)
      max_out = np.max(out_of_trie_scores)
      if np.isneginf(min_in) or np.isneginf(max_out):
        return scores
      trie_diff = max_out - min_in
      if trie_diff > -1.:
        scores[~mask] -= trie_diff + 1.
      return scores
  else:

    def _py_func(logits, trie_keys, trie_exclude):
      batch_size, vocab_size = logits.shape
      scores = np.ones_like(logits) * -np.inf

      def _py_func_inner(batch_idx):
        # for batch_idx in range(batch_size):
        trie_key = trie_keys[batch_idx]
        trie_key = _clean_trie_key(trie, trie_key)
        if not len(trie_key):
          trie_key = pygtrie._SENTINEL
        elif not trie.has_node(trie_key):
          print('WARN: Node not in trie: {}'.format(trie_key))
          return trie_key, batch_idx, set()
          # raise LookupError('Node not in trie: {}'.format(trie_key))
        trie_exclude_batch_set = _create_trie_exclude_batch_set(
            trie, trie_exclude[batch_idx])

        subtrie_idxs = _get_valid_continuation_idxs(
            trie, trie_exclude_batch_set, trie_key)

        return trie_key, batch_idx, subtrie_idxs

      for trie_key, batch_idx, subtrie_idxs in map(_py_func_inner,
                                                   range(batch_size)):
        if not len(subtrie_idxs):
          subtrie_idxs.add(int(trie.eos_idx))
        if len(subtrie_idxs):
          subtrie_idxs = np.array(list(subtrie_idxs))
          scores[batch_idx, subtrie_idxs] = logits[batch_idx, subtrie_idxs]
        else:
          raise LookupError('No continuation found from {}'.format(trie_key))
      return scores

  return _py_func


def _get_valid_continuation_idxs(trie, trie_exclude_batch_set, trie_key):
  # speedup: if we are at the beginning, we use start_idxs
  if trie.optimize_ngrams_len and (trie_key == pygtrie._SENTINEL or len(
      trie_key.split()) < trie.optimize_ngrams_len):
    if trie_key in trie.start_idxs:
      shallow_keys = trie.start_idxs[trie_key]
      return shallow_keys
    else:
      print('WARN: trie_key {} not in start_idxs'.format(trie_key))
  # If there is nothing to exclude we can go shallow.
  subtrie_keys = trie.iterkeys(
      trie_key, shallow=not len(trie_exclude_batch_set))
  subtrie_idxs = set()
  for subtrie_key in subtrie_keys:
    exclude_key = subtrie_key
    if trie_key != pygtrie._SENTINEL:
      subtrie_key = subtrie_key[len(trie_key) + 1:]
    if not len(subtrie_key):
      continue
    if exclude_key in trie_exclude_batch_set:
      continue
    subtrie_idxs.add(int(subtrie_key.split(' ', 1)[0]))
  # Add EOS when previous token is also EOS.
  if trie_key != pygtrie._SENTINEL and len(
      trie_key) and trie_key.split()[-1] == trie.eos_idx:
    subtrie_idxs.add(int(trie.eos_idx))
  return subtrie_idxs


def _create_trie_exclude_batch_set(trie, trie_exclude_batch):
  if trie.eos_idx:
    trie_exclude_batch = [
        t.strip() + ' ' + trie.eos_idx
        for t in (
            ' ' + trie_exclude_batch + ' ').split(' ' + trie.eos_idx + ' ')
        if t.strip()
    ]
  else:
    trie_exclude_batch = [trie_exclude_batch]
  trie_exclude_batch_set = set(trie_exclude_batch)
  return trie_exclude_batch_set


def _get_scores(log_probs, sequence_lengths, length_penalty_weight):
  """Calculates scores for beam search hypotheses.

  Args:
    log_probs: The log probabilities with shape `[batch_size, beam_width,
      vocab_size]`.
    sequence_lengths: The array of sequence lengths.
    length_penalty_weight: Float weight to penalize length. Disabled with 0.0.

  Returns:
    The scores normalized by the length_penalty.
  """
  length_penalty_ = _length_penalty(
      sequence_lengths=sequence_lengths, penalty_factor=length_penalty_weight)
  return log_probs / length_penalty_


def _length_penalty(sequence_lengths, penalty_factor):
  """Calculates the length penalty.

  See https://arxiv.org/abs/1609.08144.

  Returns the length penalty tensor:
  ```
  [(5+sequence_lengths)/6]**penalty_factor
  ```
  where all operations are performed element-wise.

  Args:
    sequence_lengths: `Tensor`, the sequence lengths of each hypotheses.
    penalty_factor: A scalar that weights the length penalty.

  Returns:
    If the penalty is `0`, returns the scalar `1.0`.  Otherwise returns
    the length penalty factor, a tensor with the same shape as
    `sequence_lengths`.
  """
  penalty_factor = ops.convert_to_tensor(penalty_factor, name='penalty_factor')
  penalty_factor.set_shape(())  # penalty should be a scalar.
  static_penalty = tensor_util.constant_value(penalty_factor)
  if static_penalty is not None and static_penalty == 0:
    return 1.0
  return math_ops.div((5. + math_ops.to_float(sequence_lengths))
                      **penalty_factor, (5. + 1.)**penalty_factor)


def _get_trie_scores(log_probs, sequence_lengths, length_penalty_weight,
                     trie_keys, trie_exclude, trie):
  """Calculates scores for beam search hypotheses.

  Args:
    log_probs: The log probabilities with shape `[batch_size, beam_width,
      vocab_size]`.
    sequence_lengths: The array of sequence lengths.
    length_penalty_weight: Float weight to penalize length. Disabled with 0.0.
    trie_keys: Keys (word ids) for the currently running decode pass.
    trie_exclude: A trie containing invalid targets.
    trie: The trie to decode into.

  Returns:
    The scores normalized by the length_penalty.
  """
  scores = _get_scores(log_probs, sequence_lengths, length_penalty_weight)
  scores = tf.py_func(
      _trie_scores_py_func(trie, beam_search=True),
      [log_probs, sequence_lengths, scores, trie_keys, trie_exclude],
      tf.float32,
      stateful=False)
  return scores


def _mask_probs(probs, eos_token, finished):
  """Masks log probabilities.

  The result is that finished beams allocate all probability mass to eos and
  unfinished beams remain unchanged.

  Args:
    probs: Log probabilities of shape `[batch_size, beam_width, vocab_size]`
    eos_token: An int32 id corresponding to the EOS token to allocate
      probability to.
    finished: A boolean tensor of shape `[batch_size, beam_width]` that
      specifies which elements in the beam are finished already.

  Returns:
    A tensor of shape `[batch_size, beam_width, vocab_size]`, where unfinished
    beams stay unchanged and finished beams are replaced with a tensor with all
    probability on the EOS token.
  """
  vocab_size = array_ops.shape(probs)[2]
  # All finished examples are replaced with a vector that has all
  # probability on EOS
  finished_row = array_ops.one_hot(
      eos_token,
      vocab_size,
      dtype=probs.dtype,
      on_value=ops.convert_to_tensor(0., dtype=probs.dtype),
      off_value=probs.dtype.min)
  finished_probs = array_ops.tile(
      array_ops.reshape(finished_row, [1, 1, -1]),
      array_ops.concat([array_ops.shape(finished), [1]], 0))
  finished_mask = array_ops.tile(
      array_ops.expand_dims(finished, 2), [1, 1, vocab_size])

  return array_ops.where(finished_mask, finished_probs, probs)


def _check_maybe(t):
  if t.shape.ndims is None:
    raise ValueError(
        'Expected tensor (%s) to have known rank, but ndims == None.' % t)


def _maybe_tensor_gather_helper(gather_indices, gather_from, batch_size,
                                range_size, gather_shape):
  """Maybe applies _tensor_gather_helper.

  This applies _tensor_gather_helper when the gather_from dims is at least as
  big as the length of gather_shape. This is used in conjunction with nest so
  that we don't apply _tensor_gather_helper to inapplicable values like scalars.

  Args:
    gather_indices: The tensor indices that we use to gather.
    gather_from: The tensor that we are gathering from.
    batch_size: The batch size.
    range_size: The number of values in each range. Likely equal to beam_width.
    gather_shape: What we should reshape gather_from to in order to preserve the
      correct values. An example is when gather_from is the attention from an
      AttentionWrapperState with shape [batch_size, beam_width, attention_size].
      There, we want to preserve the attention_size elements, so gather_shape is
      [batch_size * beam_width, -1]. Then, upon reshape, we still have the
      attention_size as desired.

  Returns:
    output: Gathered tensor of shape tf.shape(gather_from)[:1+len(gather_shape)]
      or the original tensor if its dimensions are too small.
  """
  if isinstance(gather_from, tensor_array_ops.TensorArray):
    return gather_from
  _check_maybe(gather_from)
  if gather_from.shape.ndims >= len(gather_shape):
    return _tensor_gather_helper(
        gather_indices=gather_indices,
        gather_from=gather_from,
        batch_size=batch_size,
        range_size=range_size,
        gather_shape=gather_shape)
  else:
    return gather_from


def _tensor_gather_helper(gather_indices,
                          gather_from,
                          batch_size,
                          range_size,
                          gather_shape,
                          name=None):
  """Helper for gathering the right indices from the tensor.

  This works by reshaping gather_from to gather_shape (e.g. [-1]) and then
  gathering from that according to the gather_indices, which are offset by
  the right amounts in order to preserve the batch order.

  Args:
    gather_indices: The tensor indices that we use to gather.
    gather_from: The tensor that we are gathering from.
    batch_size: The input batch size.
    range_size: The number of values in each range. Likely equal to beam_width.
    gather_shape: What we should reshape gather_from to in order to preserve the
      correct values. An example is when gather_from is the attention from an
      AttentionWrapperState with shape [batch_size, beam_width, attention_size].
      There, we want to preserve the attention_size elements, so gather_shape is
      [batch_size * beam_width, -1]. Then, upon reshape, we still have the
      attention_size as desired.
    name: The tensor name for set of operations. By default this is
      'tensor_gather_helper'. The final output is named 'output'.

  Returns:
    output: Gathered tensor of shape tf.shape(gather_from)[:1+len(gather_shape)]
  """
  with ops.name_scope(name, 'tensor_gather_helper'):
    range_ = array_ops.expand_dims(math_ops.range(batch_size) * range_size, 1)
    gather_indices = array_ops.reshape(gather_indices + range_, [-1])
    output = array_ops.gather(
        array_ops.reshape(gather_from, gather_shape), gather_indices)
    final_shape = array_ops.shape(gather_from)[:1 + len(gather_shape)]
    static_batch_size = tensor_util.constant_value(batch_size)
    final_static_shape = (
        tensor_shape.TensorShape([static_batch_size]).concatenate(
            gather_from.shape[1:1 + len(gather_shape)]))
    output = array_ops.reshape(output, final_shape, name='output')
    output.set_shape(final_static_shape)
    return output
