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

"""Utility functions specifically for NMT."""

from __future__ import print_function

import time
import numpy as np
import tensorflow as tf


from third_party.nmt.utils import evaluation_utils
from third_party.nmt.utils import misc_utils as utils

__all__ = ["decode_and_evaluate", "get_translation"]


def decode_and_evaluate(name,
                        model,
                        sess,
                        trans_file,
                        ref_file,
                        metrics,
                        subword_option,
                        beam_width,
                        tgt_eos,
                        hparams,
                        num_translations_per_input=1,
                        decode=True):
  """Decode a test set and compute a score according to the metrics.

    Args:
      name: name of the set being evaluated.
      model: model
      sess: session
      trans_file: name of the file that the translations will be written to.
      ref_file: ground-truth file to compare against the generated translations.
      metrics: a list of metrics that the model will be evaluated on. Valid
          options are: "f1", "bleu", "rouge", and "accuracy".
      subword_options: either "bpe", "spm", or "".
      beam_width: beam search width.
      tgt_eos: end of sentence token to the target translations.
      hparams: parameters object
      num_translations_per_input: number of translations to be generated per
          input. It is upper-bounded by beam_width
      decode: if True, generate translations using the model. Otherwise, compute
          metrics using the translations in the trans_file.
    Returns:

  """

  all_rewards = []

  # Decode
  if decode:
    utils.print_out("  decoding to output %s." % trans_file)

    start_time = time.time()
    start_time_step = time.time()
    step = 0
    num_sentences = 0
    with tf.gfile.GFile(trans_file, mode="wb") as trans_f:

      num_translations_per_input = max(
          min(num_translations_per_input, beam_width), 1)
      while True:
        try:
          _, _, _, nmt_outputs, rewards = model.infer(sess)

          all_rewards.extend(rewards.flatten().tolist())

          if beam_width == 0:
            nmt_outputs = np.expand_dims(nmt_outputs, 0)

          batch_size = nmt_outputs.shape[1]
          num_sentences += batch_size

          for sent_id in range(batch_size):
            for beam_id in range(num_translations_per_input):
              translation = get_translation(
                  nmt_outputs[beam_id],
                  sent_id,
                  tgt_eos=tgt_eos,
                  subword_option=subword_option)
              trans_f.write((translation + "\n").decode("utf-8"))

          if step % hparams.steps_per_stats == 0:
            # print_time does not print decimal places for time.
            utils.print_out("  external evaluation, step %d, time %.2f" %
                            (step, time.time() - start_time_step))
          step += 1
          start_time_step = time.time()
        except tf.errors.OutOfRangeError:
          utils.print_time(
              "  done, num sentences %d, num translations per input %d" %
              (num_sentences, num_translations_per_input), start_time)
          break

  # Evaluation
  evaluation_scores = {}

  # We treat F1 scores differently because they don't need ground truth
  # sentences and they are expensive to compute due to environment calls.
  if "f1" in metrics:
    f1_score = np.mean(all_rewards)
    evaluation_scores["f1"] = f1_score
    utils.print_out("  f1 %s: %.1f" % (name, f1_score))

  for metric in metrics:
    if metric != "f1" and ref_file:
      if not tf.gfile.Exists(trans_file):
        raise IOException("%s: translation file not found" % trans_file)
      score = evaluation_utils.evaluate(
          ref_file, trans_file, metric, subword_option=subword_option)
      evaluation_scores[metric] = score
      utils.print_out("  %s %s: %.1f" % (metric, name, score))

  return evaluation_scores


def get_translation(nmt_outputs, sent_id, tgt_eos, subword_option):
  """Given batch decoding outputs, select a sentence and turn to text."""
  if tgt_eos:
    tgt_eos = tgt_eos.encode("utf-8")

  # Select a sentence
  output = nmt_outputs[sent_id, :].tolist()

  # If there is an eos symbol in outputs, cut them at that point.
  if tgt_eos and tgt_eos in output:
    output = output[:output.index(tgt_eos)]

  if subword_option == "bpe":  # BPE
    translation = utils.format_bpe_text(output)
  elif subword_option == "spm":  # SPM
    translation = utils.format_spm_text(output)
  else:
    translation = utils.format_text(output)

  return translation
