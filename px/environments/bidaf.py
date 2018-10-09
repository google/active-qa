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
"""Wraps the BiDAF model for use as an environment.

This environment is used for the SQuAD task. The environment uses a BiDAF
model to produce an answer on a specified SQuAD datapoint to a new question
rather than the original.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import nltk
import os
import tensorflow as tf



from third_party.bi_att_flow.basic import read_data as bidaf_data
from third_party.bi_att_flow.basic import cli as bidaf_cli
from third_party.bi_att_flow.basic import evaluator as bidaf_eval
from third_party.bi_att_flow.basic import graph_handler as bidaf_graph
from third_party.bi_att_flow.basic import model as bidaf_model


class BidafEnvironment(object):
  """Environment containing the BiDAF model.

  This environment loads a BiDAF model and preprocessed data for a chosen SQuAD
  dataset. The environment is queried with a pointer to an existing datapoint,
  which contains a preprocessed SQuAD document, and a question. BiDAF is run
  using the given question against the document and the top answer with its
  score is returned.

  Attributes:
    config: BiDAF configuration read from cli.py
    data: Pre-processed SQuAD dataset.
    evaluator: BiDAF evaluation object.
    graph_handler: BiDAF object used to manage the TF graph.
    sess: single Tensorflow session used by the environment.
    model: A BiDAF Model object.
  """

  def __init__(self,
               data_dir,
               shared_path,
               model_dir,
               docid_separator='###',
               debug_mode=False,
               load_test=False,
               load_impossible_questions=False):
    """Constructor loads the BiDAF configuration, model and data.

    Args:
      data_dir: Directory containing preprocessed SQuAD data.
      shared_path: Path to shared data generated at training time.
      model_dir: Directory contining parameters of a pre-trained BiDAF model.
      docid_separator: Separator used to split suffix off the docid string.
      debug_mode: If true logs additional debug information.
      load_test: Whether the test set should be loaded as well.
      load_impossible_questions: Whether info about impossibility of questions
                                 should be loaded.
    """
    self.config = bidaf_cli.get_config()
    self.config.save_dir = model_dir
    self.config.data_dir = data_dir
    self.config.shared_path = shared_path
    self.config.mode = 'forward'
    self.docid_separator = docid_separator
    self.debug_mode = debug_mode

    self.datasets = ['train', 'dev']
    if load_test:
      self.datasets.append('test')

    data_filter = None
    self.data = dict()
    for dataset in self.datasets:
      self.data[dataset] = bidaf_data.read_data(
          self.config, dataset, True, data_filter=data_filter)
    bidaf_data.update_config(self.config, self.data.values())

    models = bidaf_model.get_model(self.config)
    self.evaluator = bidaf_eval.MultiGPUF1Evaluator(self.config, models)
    self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    self.graph_handler = bidaf_graph.GraphHandler(self.config, models[0])
    self.graph_handler.initialize(self.sess)

    nltk_data_path = os.path.join(os.path.expanduser('~'), 'data')
    nltk.data.path.append(nltk_data_path)

    self.impossible_ids = set()
    if load_impossible_questions:
      tf.logging.info('Loading impossible question ids.')
      for dataset in self.datasets:
        self.impossible_ids.update(self._ReadImpossiblities(dataset, data_dir))
    if self.debug_mode:
      tf.logging.info('Loaded {} impossible question ids.'.format(
          len(self.impossible_ids)))

  def _ReadImpossiblities(self, dataset, data_dir):
    """Collects all the docids for impossible questions."""
    data_path = os.path.join(data_dir, '{}-v2.0.json'.format(dataset))
    impossible_ids = []
    with tf.gfile.Open(data_path, 'r') as fh:
      data = json.load(fh)
      for document in data['data']:
        for paragraph in document['paragraphs']:
          for question in paragraph['qas']:
            if question['is_impossible']:
              impossible_ids.append(question['id'])

    if self.debug_mode:
      tf.logging.info('Loaded {} impossible question ids from {}.'.format(
          len(impossible_ids), dataset))
    return impossible_ids

  def _WordTokenize(self, text):
    """Tokenizes the text NLTK for consistency with BiDAF."""
    return [
        token.replace("''", '"').replace('``', '"')
        for token in nltk.word_tokenize(text)
    ]

  def _PreprocessQaData(self, questions, document_ids):
    """Prepares the BiDAF Data object.

    Loads a batch of SQuAD datapoints, identified by their 'ids' field. The
    questions are replaced with those specified in the input.  All datapoints
    must come from the same original dataset (train, dev or test), else the
    shared data will be incorrect. The first id in document_ids is used to
    determine the dataset, a KeyError is thrown if the other ids are not in this
    dataset.

    Args:
      questions: List of strings used to replace the original question.
      document_ids: Identifiers for the SQuAD datapoints to use.

    Returns:
      data: BiDAF Data object containing the desired datapoints only.
      data.shared: The appropriate shared data from the dataset containing
                   the ids in document_ids
      id2questions_dict: A dict mapping docids to original questions and
                         rewrites.

    Raises:
      KeyError: Occurs if it is not the case that all document_ids are present
                in a single preloaded dataset.
    """
    first_docid = document_ids[0].split(self.docid_separator)[0]
    if first_docid in self.data['train'].data['ids']:
      dataset = self.data['train']
    elif first_docid in self.data['dev'].data['ids']:
      dataset = self.data['dev']
    elif 'test' in self.data and first_docid in self.data['test'].data['ids']:
      dataset = self.data['test']
    else:
      raise KeyError('Document id not present: {}'.format(first_docid))
    data_indices = [
        dataset.data['ids'].index(document_ids[i].split(
            self.docid_separator)[0]) for i in range(len(document_ids))
    ]

    data_out = dict()
    # Copies relevant datapoint, retaining the input docids.
    for key in dataset.data.iterkeys():
      if key == 'ids':
        data_out[key] = document_ids
      else:
        data_out[key] = [dataset.data[key][i] for i in data_indices]
    if self.debug_mode:
      for q in data_out['q']:
        tf.logging.info('Original question: {}'.format(
            ' '.join(q).encode('utf-8')))

    # Replaces the question in the datapoint for the rewrite.
    id2questions_dict = dict()
    for i in range(len(questions)):
      id2questions_dict[data_out['ids'][i]] = dict()
      id2questions_dict[data_out['ids'][i]]['original'] = ' '.join(
          data_out['q'][i])
      data_out['q'][i] = self._WordTokenize(questions[i])

      if len(data_out['q'][i]) > self.config.max_ques_size:
        tf.logging.info('Truncated question from {} to {}'.format(
            len(data_out['q'][i]), self.config.max_ques_size))
        data_out['q'][i] = data_out['q'][i][:self.config.max_ques_size]

      id2questions_dict[data_out['ids'][i]]['raw_rewrite'] = questions[i]
      id2questions_dict[data_out['ids'][i]]['rewrite'] = ' '.join(
          data_out['q'][i])
      data_out['cq'][i] = [list(qij) for qij in data_out['q'][i]]

    if self.debug_mode:
      for q in data_out['q']:
        tf.logging.info('New question:      {}'.format(
            ' '.join(q).encode('utf-8')))

    return data_out, dataset.shared, id2questions_dict

  def IsImpossible(self, document_id):
    return document_id in self.impossible_ids

  def GetAnswers(self, questions, document_ids):
    """Computes an answer for a given question from a SQuAD datapoint.

    Runs a BiDAF model on a specified SQuAD datapoint, but using the input
    question in place of the original.

    Args:
      questions: List of strings used to replace the original question.
      document_ids: Identifiers for the SQuAD datapoints to use.

    Returns:
      e.id2answer_dict: A dict containing the answers and their scores.
      e.loss: Scalar training loss for the entire batch.
      id2questions_dict: A dict mapping docids to original questions and
                         rewrites.

    Raises:
      ValueError: If the number of questions and document_ids differ.
      ValueError: If the document_ids are not unique.
    """
    if len(questions) != len(document_ids):
      raise ValueError('Number of questions and document_ids must be equal.')
    if len(document_ids) > len(set(document_ids)):
      raise ValueError('document_ids must be unique.')
    raw_data, shared, id2questions_dict = self._PreprocessQaData(
        questions, document_ids)
    data = bidaf_data.DataSet(raw_data, data_type='', shared=shared)

    num_batches = int(math.ceil(data.num_examples / self.config.batch_size))
    e = None
    for multi_batch in data.get_multi_batches(
        self.config.batch_size, self.config.num_gpus, num_steps=num_batches):
      ei = self.evaluator.get_evaluation(self.sess, multi_batch)
      e = ei if e is None else e + ei
    if self.debug_mode:
      tf.logging.info(e)
      self.graph_handler.dump_answer(e, path=self.config.answer_path)
      self.graph_handler.dump_eval(e, path=self.config.eval_path)
    return e.id2answer_dict, id2questions_dict, e.loss
