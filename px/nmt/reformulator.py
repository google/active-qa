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

"""A reformulator model.

Implementation of a reformulator model that takes as input a list of questions
strings and returns a list of lists of reformulation strings.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from collections import namedtuple

import tensorflow as tf

import sentencepiece as sentencepiece_processor
from third_party.nmt.utils import misc_utils

from px.nmt import attention_model
from px.nmt import gnmt_model
from px.nmt import model as nmt_model
from px.nmt import model_helper
from px.nmt import nmt as hparam_utils
from px.nmt.utils import nmt_utils
from px.nmt.utils import trie_decoder_utils
from px.proto import reformulator_pb2

Reformulation = namedtuple(
    "Reformulation", ["reformulation", "tokenized_reformulation", "answers"])


def load_hparams(hparams_file, out_dir):
  """Load hparams from an json file.

  Partial hparams are augmented using default values for parameters that were
  added after the hparams file was written.

  Args:
    hparams_file: Filename of the hparams file.
    out_dir: Directory where the model output will be written.

  Returns:
    hparams: tf.contrib.training.HParams object.
  """
  default_hparams = hparam_utils.get_default_hparams()
  with tf.gfile.GFile(hparams_file) as f:
    hparams_values = json.load(f)
  hparams_values[u"out_dir"] = out_dir
  hparams = tf.contrib.training.HParams(**hparams_values)
  hparam_utils.ensure_compatible_hparams(
      hparams, default_hparams, hparams_path="")
  hparams = hparam_utils.extend_hparams(hparams)
  return hparams


class Reformulator(object):
  """A Reformulator model that reformulates questions.
  """

  def __init__(self, hparams_path, source_prefix, out_dir,
               environment_server_address):
    """Constructor for the reformulator.

    Args:
      hparams_path: Path to json hparams file.
      source_prefix: A prefix that is added to every question before
        translation which should be used for adding tags like <en> <2en>.
        Can be empty or None in which case the prefix is ignored.
      out_dir: Directory where the model output will be written.
      environment_server_address: Address of the environment server.

    Raises:
      ValueError: if model architecture is not known.
    """

    self.hparams = load_hparams(hparams_path, out_dir)
    assert self.hparams.num_buckets == 1, "No bucketing when in server mode."
    assert not self.hparams.server_mode, ("server_mode set to True but not "
                                          "running as server.")

    self.hparams.environment_server = environment_server_address
    if self.hparams.subword_option == "spm":
      self.sentpiece = sentencepiece_processor.SentencePieceProcessor()
      self.sentpiece.Load(self.hparams.subword_model.encode("utf-8"))
    self.source_prefix = source_prefix

    # Create the model
    if not self.hparams.attention:
      model_creator = nmt_model.Model
    elif self.hparams.attention_architecture == "standard":
      model_creator = attention_model.AttentionModel
    elif self.hparams.attention_architecture in ["gnmt", "gnmt_v2"]:
      model_creator = gnmt_model.GNMTModel
    else:
      raise ValueError("Unknown model architecture")

    self.trie = trie_decoder_utils.DecoderTrie(
        vocab_path=self.hparams.tgt_vocab_file,
        eos_token=self.hparams.eos,
        subword_option=self.hparams.subword_option,
        subword_model=self.hparams.get("subword_model"),
        optimize_ngrams_len=self.hparams.optimize_ngrams_len)
    if self.hparams.trie_path is not None and tf.gfile.Exists(
        self.hparams.trie_path):
      self.trie.populate_from_text_file(self.hparams.trie_path)


    combined_graph = tf.Graph()
    self.train_model = model_helper.create_train_model(
        model_creator,
        self.hparams,
        graph=combined_graph,
        trie=self.trie,
        use_placeholders=True)

    # Create different inference models for beam search, sampling and greedy
    # decoding.
    default_infer_mode = self.hparams.infer_mode
    default_beam_width = self.hparams.beam_width
    self.infer_models = {}
    self.hparams.use_rl = False
    self.hparams.infer_mode = "greedy"
    self.hparams.beam_width = 0
    self.infer_models[reformulator_pb2.ReformulatorRequest.
                      GREEDY] = model_helper.create_infer_model(
                          model_creator, self.hparams, graph=combined_graph)

    self.hparams.infer_mode = "sample"
    self.hparams.beam_width = 0
    self.infer_models[reformulator_pb2.ReformulatorRequest.
                      SAMPLING] = model_helper.create_infer_model(
                          model_creator, self.hparams, graph=combined_graph)

    self.hparams.infer_mode = "beam_search"
    self.hparams.beam_width = max(1, default_beam_width)
    self.infer_models[reformulator_pb2.ReformulatorRequest.
                      BEAM_SEARCH] = model_helper.create_infer_model(
                          model_creator, self.hparams, graph=combined_graph)


    self.hparams.infer_mode = "trie_greedy"
    self.hparams.beam_width = 0
    self.infer_models[reformulator_pb2.ReformulatorRequest.
                      TRIE_GREEDY] = model_helper.create_infer_model(
                          model_creator,
                          self.hparams,
                          graph=combined_graph,
                          trie=self.trie)
    self.hparams.infer_mode = default_infer_mode
    self.hparams.beam_width = default_beam_width

    self.hparams.infer_mode = "trie_sample"
    self.hparams.beam_width = 0
    self.infer_models[reformulator_pb2.ReformulatorRequest.
                      TRIE_SAMPLE] = model_helper.create_infer_model(
                          model_creator,
                          self.hparams,
                          graph=combined_graph,
                          trie=self.trie)

    self.hparams.infer_mode = "trie_beam_search"
    self.hparams.beam_width = max(1, default_beam_width)
    self.infer_models[reformulator_pb2.ReformulatorRequest.
                      TRIE_BEAM_SEARCH] = model_helper.create_infer_model(
                          model_creator,
                          self.hparams,
                          graph=combined_graph,
                          trie=self.trie)

    self.hparams.use_rl = True
    self.sess = tf.Session(
        graph=combined_graph, config=misc_utils.get_config_proto())

    with combined_graph.as_default():
      self.sess.run(tf.global_variables_initializer())
      self.sess.run(tf.tables_initializer())
      _, global_step = model_helper.create_or_load_model(
          self.train_model.model, out_dir, self.sess, "train")
      self.last_save_step = global_step

    self.summary_writer = tf.summary.FileWriter(
        os.path.join(out_dir, "train_log"), self.train_model.graph)
    self.checkpoint_path = os.path.join(out_dir, "translate.ckpt")
    self.trie_save_path = os.path.join(out_dir, "trie")

  def tokenize(self, questions, prefix=""):
    """Tokenizes the questions if a sentencepiece model is used.

    In particular, this makes sure the prefix is set correctly.

    Args:
      questions: A list of questions to be tokenized.
      prefix: The expected question prefix.

    Returns:
      A list of tokenized questions.
    """
    if self.hparams.subword_option == "spm":
      tokenized_questions = []
      for question in questions:
        if question.startswith(prefix):
          question = question[len(prefix):].lstrip()
        tokenized_questions.append(" ".join(
            self.sentpiece.EncodeAsPieces(question.encode("utf-8"))))
    else:
      tokenized_questions = questions
    if prefix:
      # Note that no extra space is added between the prefix and the question.
      tokenized_questions = [prefix + q for q in tokenized_questions]
    return tokenized_questions

  def detokenize(self, text):
    """Detokenizes a sentence piece tokenized text.

    Args:
      text: A single string, containing whitespace separated tokens.

    Returns:
      A string where the tokens have been joined into words.
    """
    if self.hparams.subword_option == "spm":
      text = self.sentpiece.DecodePieces(text.split(" "))
    return text.decode("utf-8")

  def reformulate(self, questions, inference_mode, trie_excludes=None):
    """Reformulates questions given original ones.

    Args:
      questions: list of strings containing the questions.
      inference_mode: inference mode from ReformulatorRequest.InferenceMode
          enum.
      trie_excludes: reformulations to exclude from trie decoding.
                     should be a list of lists of strings.

    Returns:
      A list of lists of reformulations of the questions.
    """
    tokenized_questions = self.tokenize(questions, prefix=self.source_prefix)

    infer_model = self.infer_models[inference_mode]
    iterator_feed_dict = {
        infer_model.src_placeholder: tokenized_questions,
        infer_model.batch_size_placeholder: len(tokenized_questions)
    }

    decode_into_trie = inference_mode in [
        reformulator_pb2.ReformulatorRequest.TRIE_BEAM_SEARCH,
        reformulator_pb2.ReformulatorRequest.TRIE_SAMPLE,
        reformulator_pb2.ReformulatorRequest.TRIE_GREEDY
    ]

    if decode_into_trie:
      if not trie_excludes:
        trie_excludes = [[] for _ in questions]

      tokenized_trie_excludes = [
          " {} ".format(self.hparams.eos).join(self.tokenize(e))
          for e in trie_excludes
      ]

      iterator_feed_dict[
          infer_model.trie_exclude_placeholder] = tokenized_trie_excludes

    self.sess.run(
        infer_model.iterator.initializer, feed_dict=iterator_feed_dict)

    # nmt_outputs is a tensor of shape [beam_size, batch size, seq_len]
    _, _, nmt_output_ids, nmt_outputs, _ = infer_model.model.infer(self.sess)

    if nmt_outputs.ndim == 2:
      # Add a dimension when not using beam search. This allows us to use the
      # same code to handle greedy, sampling, and beam search mode.
      nmt_outputs = nmt_outputs[None, :, :]
      nmt_output_ids = nmt_output_ids[None, :, :]

    all_reformulations = []
    for sent_id in range(nmt_outputs.shape[1]):
      reformulations = []
      for beam_id in range(nmt_outputs.shape[0]):
        tokenized_reformulation = nmt_utils.get_translation(
            nmt_outputs[beam_id],
            sent_id=sent_id,
            tgt_eos=self.hparams.eos,
            subword_option=None)
        reformulation = self.detokenize(tokenized_reformulation)

        answers = None
        if decode_into_trie:
          cur_ids = nmt_output_ids[beam_id][sent_id].astype(str).tolist()
          try:
            eos_idx = cur_ids.index(self.trie.eos_idx)
          except ValueError:
            eos_idx = len(cur_ids) - 1

          trie_key = " ".join(cur_ids[:eos_idx + 1])

          answers = self.trie.get(trie_key)

        reformulations.append(
            Reformulation(reformulation, tokenized_reformulation, answers))

      all_reformulations.append(reformulations)

    return all_reformulations

  def train(self, sources, annotations):
    """Trains the reformulator with the given sources.

    Args:
      sources: A list of strings representing the questions.
      annotations: A list of strings representing the document ids.

    Returns:
      Training loss.
      Rewards.
      Rewrites.
    """
    tokenized_sources = self.tokenize(sources, prefix=self.source_prefix)

    iterator_feed_dict = {
        self.train_model.src_placeholder: tokenized_sources,
        self.train_model.tgt_placeholder: tokenized_sources,  # Unused.
        self.train_model.annot_placeholder: annotations,
        self.train_model.skip_count_placeholder: 0
    }

    self.sess.run(
        self.train_model.iterator.initializer, feed_dict=iterator_feed_dict)
    train_result = self.train_model.model.train(self.sess)
    global_step = self.train_model.model.global_step.eval(session=self.sess)
    self.summary_writer.add_summary(train_result[4], global_step)

    # Regularly save a checkpoint.
    if global_step - self.last_save_step >= self.hparams.steps_per_save:
      misc_utils.print_out("Save at step: {}".format(global_step))
      self.last_save_step = global_step
      self.train_model.model.saver.save(
          self.sess, self.checkpoint_path, global_step=global_step)
      if self.trie:
        self.trie.save_to_file(self.trie_save_path +
                               ".{}.pkl".format(global_step))

    rewrites = [rewrite.decode("utf-8") for rewrite in train_result[10]]

    return train_result[1], train_result[2], rewrites
