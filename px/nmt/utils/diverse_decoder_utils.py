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
"""Utilities for decoding a seq2seq model using a diverse decoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf

from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.util import nest

__all__ = ["DiverseBeamSearchDecoder"]


def _check_maybe(t):
  if t.shape.ndims is None:
    raise ValueError(
        "Expected tensor (%s) to have known rank, but ndims == None." % t)


class DiverseBeamSearchDecoder(beam_search_decoder.BeamSearchDecoder):
  """Diverse Beam Search decoder."""

  def __init__(self, decoder_scope, maximum_iterations, decoding_iterations,
               *args, **kwargs):
    """Initialize the DiverseBeamSearchDecoder.

    Args:
      decoder_scope: Scope.
      maximum_iterations: int, Maximum number of decoding iterations.
      decoding_iterations: number of sequential beam search decodings.
      *args: Other argments to apply to the BeamSearchDecoder class.
      **kwargs: Keyword arguments to apply to the BeamSearchDecoder class.
    """
    super(DiverseBeamSearchDecoder, self).__init__(*args, **kwargs)

    self._maximum_iterations = maximum_iterations
    self._decoding_iterations = decoding_iterations
    self._decoding_iterations_remaining = decoding_iterations
    self._decoder_scope = decoder_scope
    self._forbidden_tokens = None

  def finalize(self, outputs, final_state, sequence_lengths):
    """Finalize and return the predicted_ids.

    Args:
      outputs: An instance of BeamSearchDecoderOutput.
      final_state: An instance of BeamSearchDecoderState. Passed through to the
        output.
      sequence_lengths: An `int64` tensor shaped `[batch_size, beam_width]`. The
        sequence lengths determined for each beam during decode. **NOTE** These
        are ignored; the updated sequence lengths are stored in
        `final_state.lengths`.

    Returns:
      outputs: An instance of `FinalBeamSearchDecoderOutput` where the
        predicted_ids are the result of calling _gather_tree.
      final_state: The same input instance of `BeamSearchDecoderState`.
    """
    del sequence_lengths

    self._decoding_iterations_remaining -= 1  # Decrease counter.

    # Get max_sequence_length across all beams for each batch.
    max_sequence_lengths = math_ops.to_int32(
        math_ops.reduce_max(final_state.lengths, axis=1))
    predicted_ids = beam_search_ops.gather_tree(
        outputs.predicted_ids,
        outputs.parent_ids,
        max_sequence_lengths=max_sequence_lengths,
        end_token=self._end_token)
    if self._reorder_tensor_arrays:
      # pylint: disable=g-long-lambda
      # pylint: disable=line-too-long
      final_state = final_state._replace(
          cell_state=nest.map_structure(
              lambda t: self._maybe_sort_array_beams(t, outputs.parent_ids, final_state.lengths),
              final_state.cell_state))
      # pylint: enable=g-long-lambda
      # pylint: enable=line-too-long
    if self._decoding_iterations_remaining >= 1:

      # Transpose to [batch_size, time, beam_width]
      new_forbidden_tokens = tf.transpose(predicted_ids, perm=[1, 0, 2])
      # Reshape to [batch_size, time * beam_width]
      new_forbidden_tokens = tf.reshape(
          new_forbidden_tokens, shape=[tf.shape(new_forbidden_tokens)[0], -1])
      if self._forbidden_tokens is not None:
        self._forbidden_tokens = tf.concat(
            [self._forbidden_tokens, new_forbidden_tokens], axis=1)
      else:
        self._forbidden_tokens = new_forbidden_tokens

      new_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
          self,
          maximum_iterations=self._maximum_iterations,
          output_time_major=True,
          swap_memory=True,
          scope=self._decoder_scope)

      all_scores = tf.concat(
          [outputs.scores, new_outputs.beam_search_decoder_output.scores],
          axis=2)
      all_predicted_ids = tf.concat([
          outputs.predicted_ids,
          new_outputs.beam_search_decoder_output.predicted_ids
      ],
                                    axis=2)
      all_parent_ids = tf.concat([
          outputs.parent_ids, new_outputs.beam_search_decoder_output.parent_ids
      ],
                                 axis=2)
      outputs = beam_search_decoder.BeamSearchDecoderOutput(
          scores=all_scores,
          predicted_ids=all_predicted_ids,
          parent_ids=all_parent_ids)

      # Append eos token ids in case predicted_ids is shorter than new
      # predicted_ids, and vice-versa.
      predicted_ids = pad(
          x=predicted_ids,
          max_size=tf.shape(new_outputs.predicted_ids)[0],
          value=self._end_token)
      new_predicted_ids = pad(
          x=new_outputs.predicted_ids,
          max_size=tf.shape(predicted_ids)[0],
          value=self._end_token)
      predicted_ids = tf.concat([predicted_ids, new_predicted_ids], axis=2)


    outputs = beam_search_decoder.FinalBeamSearchDecoderOutput(
        beam_search_decoder_output=outputs, predicted_ids=predicted_ids)

    return outputs, final_state

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
    coverage_penalty_weight = self._coverage_penalty_weight

    with ops.name_scope(name, "BeamSearchDecoderStep", (time, inputs, state)):
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

      if self._forbidden_tokens is not None:

        def mask_forbidden(arr, forbidden_indices, end_token):
          """Replaces the elements in `arr` with a low constant.

          Args:
            arr: a numpy array of shape [batch_size, beam_width, vocab_size].
            forbidden_indices: a numpy array of shape [batch_size,
              num_forbidden_tokens].
            end_token: a int32 scalar representing the eos token id.

          Returns:
            a numpy array of shape [batch_size, beam_width, vocab_size].
          """
          batch_indices = np.arange(arr.shape[0]).repeat(
              forbidden_indices.shape[1])

          # Remove eos token from indices.
          mask = (forbidden_indices != end_token).flatten()
          batch_indices = batch_indices[mask]
          forbidden_indices = forbidden_indices.flatten()[mask]
          # Set a very low logit value so it is never selected by the decoder.
          arr[batch_indices, :,
              forbidden_indices.reshape((-1))] = np.minimum(
                  arr[batch_indices, :,
                      forbidden_indices.reshape((-1))], -1e7)
          return arr

        # It is faster to mask out the logits in numpy than executing the
        # equivalent, but more complicated, tensorflow operations.
        cell_outputs = tf.py_func(
            func=mask_forbidden,
            inp=[cell_outputs, self._forbidden_tokens, end_token],
            Tout=(tf.float32),
            name="mask_forbidden")

      (beam_search_output, beam_search_state) = _beam_search_step(
          time=time,
          logits=cell_outputs,
          next_cell_state=next_cell_state,
          beam_state=state,
          batch_size=batch_size,
          beam_width=beam_width,
          end_token=end_token,
          length_penalty_weight=length_penalty_weight,
          coverage_penalty_weight=coverage_penalty_weight)

      finished = beam_search_state.finished
      sample_ids = beam_search_output.predicted_ids
      next_inputs = control_flow_ops.cond(
          math_ops.reduce_all(finished), lambda: self._start_inputs,
          lambda: self._embedding_fn(sample_ids))

    return (beam_search_output, beam_search_state, next_inputs, finished)


def pad(x, max_size, value=0.0):
  """Makes the first dimension of x to be at least max_size.

  Args:
    x: a 3-D tensor.
    max_size: an int32 or int64 tensor.
    value: the value that the new elements of x will have.

  Returns:
    The expanded tensor with shape
      [max(x.shape[0], max_size), x.shape[1], x.shape[2]].
  """

  fill = tf.fill(
      dims=[
          tf.maximum(max_size - tf.shape(x)[0], 0),
          tf.shape(x)[1],
          tf.shape(x)[2]
      ],
      value=value)
  return tf.concat([x, fill], axis=0)


def _beam_search_step(time, logits, next_cell_state, beam_state, batch_size,
                      beam_width, end_token, length_penalty_weight,
                      coverage_penalty_weight):
  """Performs a single step of Beam Search Decoding.

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
    coverage_penalty_weight: Float weight to penalize the coverage of source
      sentence. Disabled with 0.0.

  Returns:
    A new beam state.
  """
  static_batch_size = tensor_util.constant_value(batch_size)

  # Calculate the current lengths of the predictions
  prediction_lengths = beam_state.lengths
  previously_finished = beam_state.finished
  not_finished = math_ops.logical_not(previously_finished)

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
  add_mask = math_ops.to_int64(not_finished)
  lengths_to_add *= array_ops.expand_dims(add_mask, 2)
  new_prediction_lengths = (
      lengths_to_add + array_ops.expand_dims(prediction_lengths, 2))

  # Calculate the accumulated attention probabilities if coverage penalty is
  # enabled.
  accumulated_attention_probs = None
  attention_probs = get_attention_probs(next_cell_state,
                                        coverage_penalty_weight)
  if attention_probs is not None:
    attention_probs *= array_ops.expand_dims(math_ops.to_float(not_finished), 2)
    accumulated_attention_probs = (
        beam_state.accumulated_attention_probs + attention_probs)

  # Calculate the scores for each beam
  scores = _get_scores(
      log_probs=total_probs,
      sequence_lengths=new_prediction_lengths,
      length_penalty_weight=length_penalty_weight,
      coverage_penalty_weight=coverage_penalty_weight,
      finished=previously_finished,
      accumulated_attention_probs=accumulated_attention_probs)

  time = ops.convert_to_tensor(time, name="time")
  # During the first time step we only consider the initial beam
  scores_flat = array_ops.reshape(scores, [batch_size, -1])

  # Pick the next beams according to the specified successors function
  next_beam_size = ops.convert_to_tensor(
      beam_width, dtype=dtypes.int32, name="beam_width")
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
      name="next_beam_probs")
  # Note: just doing the following
  #   math_ops.to_int32(word_indices % vocab_size,
  #       name="next_beam_word_ids")
  # would be a lot cleaner but for reasons unclear, that hides the results of
  # the op which prevents capturing it with tfdbg debug ops.
  raw_next_word_ids = math_ops.mod(
      word_indices, vocab_size, name="next_beam_word_ids")
  next_word_ids = math_ops.to_int32(raw_next_word_ids)
  next_beam_ids = math_ops.to_int32(
      word_indices / vocab_size, name="next_beam_parent_ids")

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
      name="next_beam_finished")

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
  next_accumulated_attention_probs = ()
  if accumulated_attention_probs is not None:
    next_accumulated_attention_probs = _tensor_gather_helper(
        gather_indices=next_beam_ids,
        gather_from=accumulated_attention_probs,
        batch_size=batch_size,
        range_size=beam_width,
        gather_shape=[batch_size * beam_width, -1],
        name="next_accumulated_attention_probs")

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

  next_state = beam_search_decoder.BeamSearchDecoderState(
      cell_state=next_cell_state,
      log_probs=next_beam_probs,
      lengths=next_prediction_len,
      finished=next_finished,
      accumulated_attention_probs=next_accumulated_attention_probs)

  output = beam_search_decoder.BeamSearchDecoderOutput(
      scores=next_beam_scores,
      predicted_ids=next_word_ids,
      parent_ids=next_beam_ids)

  return output, next_state


def get_attention_probs(next_cell_state, coverage_penalty_weight):
  """Get attention probabilities from the cell state.

  Args:
    next_cell_state: The next state from the cell, e.g. an instance of
      AttentionWrapperState if the cell is attentional.
    coverage_penalty_weight: Float weight to penalize the coverage of source
      sentence. Disabled with 0.0.

  Returns:
    The attention probabilities with shape `[batch_size, beam_width, max_time]`
    if coverage penalty is enabled. Otherwise, returns None.

  Raises:
    ValueError: If no cell is attentional but coverage penalty is enabled.
  """
  if coverage_penalty_weight == 0.0:
    return None

  # Attention probabilities of each attention layer. Each with shape
  # `[batch_size, beam_width, max_time]`.
  probs_per_attn_layer = []
  if isinstance(next_cell_state, attention_wrapper.AttentionWrapperState):
    probs_per_attn_layer = [attention_probs_from_attn_state(next_cell_state)]
  elif isinstance(next_cell_state, tuple):
    for state in next_cell_state:
      if isinstance(state, attention_wrapper.AttentionWrapperState):
        probs_per_attn_layer.append(attention_probs_from_attn_state(state))

  if not probs_per_attn_layer:
    raise ValueError(
        "coverage_penalty_weight must be 0.0 if no cell is attentional.")

  if len(probs_per_attn_layer) == 1:
    attention_probs = probs_per_attn_layer[0]
  else:
    # Calculate the average attention probabilities from all attention layers.
    attention_probs = [
        array_ops.expand_dims(prob, -1) for prob in probs_per_attn_layer
    ]
    attention_probs = array_ops.concat(attention_probs, -1)
    attention_probs = math_ops.reduce_mean(attention_probs, -1)

  return attention_probs


def _get_scores(log_probs, sequence_lengths, length_penalty_weight,
                coverage_penalty_weight, finished, accumulated_attention_probs):
  """Calculates scores for beam search hypotheses.

  Args:
    log_probs: The log probabilities with shape `[batch_size, beam_width,
      vocab_size]`.
    sequence_lengths: The array of sequence lengths.
    length_penalty_weight: Float weight to penalize length. Disabled with 0.0.
    coverage_penalty_weight: Float weight to penalize the coverage of source
      sentence. Disabled with 0.0.
    finished: A boolean tensor of shape `[batch_size, beam_width]` that
      specifies which elements in the beam are finished already.
    accumulated_attention_probs: Accumulated attention probabilities up to the
      current time step, with shape `[batch_size, beam_width, max_time]` if
      coverage_penalty_weight is not 0.0.

  Returns:
    The scores normalized by the length_penalty and coverage_penalty.

  Raises:
    ValueError: accumulated_attention_probs is None when coverage penalty is
      enabled.
  """
  length_penalty_ = _length_penalty(
      sequence_lengths=sequence_lengths, penalty_factor=length_penalty_weight)
  scores = log_probs / length_penalty_

  coverage_penalty_weight = ops.convert_to_tensor(
      coverage_penalty_weight, name="coverage_penalty_weight")
  if coverage_penalty_weight.shape.ndims != 0:
    raise ValueError("coverage_penalty_weight should be a scalar, "
                     "but saw shape: %s" % coverage_penalty_weight.shape)

  if tensor_util.constant_value(coverage_penalty_weight) == 0.0:
    return scores

  if accumulated_attention_probs is None:
    raise ValueError(
        "accumulated_attention_probs can be None only if coverage penalty is "
        "disabled.")

  # Add source sequence length mask before computing coverage penalty.
  accumulated_attention_probs = array_ops.where(
      math_ops.equal(accumulated_attention_probs, 0.0),
      array_ops.ones_like(accumulated_attention_probs),
      accumulated_attention_probs)

  # coverage penalty =
  #     sum over `max_time` {log(min(accumulated_attention_probs, 1.0))}
  coverage_penalty = math_ops.reduce_sum(
      math_ops.log(math_ops.minimum(accumulated_attention_probs, 1.0)), 2)
  # Apply coverage penalty to finished predictions.
  coverage_penalty *= math_ops.to_float(finished)
  weighted_coverage_penalty = coverage_penalty * coverage_penalty_weight
  # Reshape from [batch_size, beam_width] to [batch_size, beam_width, 1]
  weighted_coverage_penalty = array_ops.expand_dims(weighted_coverage_penalty,
                                                    2)
  return scores + weighted_coverage_penalty


def attention_probs_from_attn_state(attention_state):
  """Calculates the average attention probabilities.

  Args:
    attention_state: An instance of `AttentionWrapperState`.

  Returns:
    The attention probabilities in the given AttentionWrapperState.
    If there're multiple attention mechanisms, return the average value from
    all attention mechanisms.
  """
  # Attention probabilities over time steps, with shape
  # `[batch_size, beam_width, max_time]`.
  attention_probs = attention_state.alignments
  if isinstance(attention_probs, tuple):
    attention_probs = [
        array_ops.expand_dims(prob, -1) for prob in attention_probs
    ]
    attention_probs = array_ops.concat(attention_probs, -1)
    attention_probs = math_ops.reduce_mean(attention_probs, -1)
  return attention_probs


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
  penalty_factor = ops.convert_to_tensor(penalty_factor, name="penalty_factor")
  penalty_factor.set_shape(())  # penalty should be a scalar.
  static_penalty = tensor_util.constant_value(penalty_factor)
  if static_penalty is not None and static_penalty == 0:
    return 1.0
  return math_ops.div((5. + math_ops.to_float(sequence_lengths))
                      **penalty_factor, (5. + 1.)**penalty_factor)


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
  with ops.name_scope(name, "tensor_gather_helper"):
    range_ = array_ops.expand_dims(math_ops.range(batch_size) * range_size, 1)
    gather_indices = array_ops.reshape(gather_indices + range_, [-1])
    output = array_ops.gather(
        array_ops.reshape(gather_from, gather_shape), gather_indices)
    final_shape = array_ops.shape(gather_from)[:1 + len(gather_shape)]
    static_batch_size = tensor_util.constant_value(batch_size)
    final_static_shape = (
        tensor_shape.TensorShape([static_batch_size]).concatenate(
            gather_from.shape[1:1 + len(gather_shape)]))
    output = array_ops.reshape(output, final_shape, name="output")
    output.set_shape(final_static_shape)
    return output
