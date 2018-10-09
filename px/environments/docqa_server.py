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

r"""gRPC server for the DocQA environment.

Implementation of a gRPC server for the DocQA model. Requests contain
questions and document IDs that identify SQuAD datapoints. The responses
contain answers from the BiDAF environment and associated scores.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from concurrent import futures
import time


from absl import app
from absl import flags
from absl import logging
import grpc

from px.environments import docqa
from px.environments import docqa_squad
from px.proto import aqa_pb2
from px.proto import aqa_pb2_grpc

FLAGS = flags.FLAGS
flags.DEFINE_integer('port', 10000, 'Port to listen on.')
flags.DEFINE_string('precomputed_data_path', '', 'Precomputed data path.')
flags.DEFINE_string('corpus_part', 'train', 'train or dev')
flags.DEFINE_string('docqa_model_dir', '', 'Directory of trained DocQA model.')
flags.DEFINE_string('nltk_dir', '', 'NLTK directory.')
flags.DEFINE_integer('worker_threads', 10,
                     'Number of worker threads running on the server.')
flags.DEFINE_integer('sleep_seconds', 10,
                     'Number of seconds to wait for a termination event.')
flags.DEFINE_bool('load_test', False,
                  'Load test data in addition to dev and train.')
flags.DEFINE_bool('debug_mode', False,
                  'If true, log questions, answers, and scores.')
flags.DEFINE_enum('model_type', 'squad', ['squad', 'triviaqa'], 'Model type.')


class DocqaServer(aqa_pb2_grpc.EnvironmentServerServicer):
  """A gRPC server for the DocQA environment.

  Attributes:
    environment: A DocqaEnvironment object that returns scored answers to
                 questions.
  """

  def __init__(self, *args, **kwargs):
    """"Constructor for the BiDAF server."""
    precomputed_data_path = kwargs.pop('precomputed_data_path', None)
    corpus_dir = kwargs.pop('corpus_dir', None)
    model_dir = kwargs.pop('docqa_model_dir', None)
    nltk_dir = kwargs.pop('nltk_dir', None)
    load_test = kwargs.pop('load_test', False)
    debug_mode = kwargs.pop('debug_mode', False)
    model_type = kwargs.pop('model_type', 'squad')
    corpus_name = kwargs.pop('corpus_name', None)
    corpus_part = kwargs.pop('corpus_part', None)
    self.debug_mode = debug_mode
    if model_type == 'triviaqa':
      self._InitializeEnvironment(
          precomputed_data_path=precomputed_data_path,
          corpus_dir=corpus_dir,
          model_dir=model_dir,
          nltk_dir=nltk_dir,
          load_test=load_test,
          debug_mode=debug_mode)
    elif model_type == 'squad':
      self._InitializeSquadEnvironment(
          corpus_dir=corpus_dir,
          corpus_name=corpus_name,
          corpus_part=corpus_part,
          model_dir=model_dir,
          nltk_dir=nltk_dir)

  def _InitializeEnvironment(self, precomputed_data_path, corpus_dir, model_dir,
                             nltk_dir, load_test, debug_mode):
    """Initilizes the DocQA model environment.

    Args:
      precomputed_data_path: Path to the precomputed data stored in a pickle
          file.
      corpus_dir: Path to corpus directory.
      model_dir: Directory containing parameters of a pre-trained DocQA model.
      nltk_dir: Folder containing the nltk package.
      load_test: If True, loads the test set as well.
      debug_mode: If true, logs additional debug information.
    """
    self._environment = docqa.DocqaEnvironment(
        precomputed_data_path=precomputed_data_path,
        corpus_dir=corpus_dir,
        model_dir=model_dir,
        nltk_dir=nltk_dir,
        load_test=load_test,
        debug_mode=debug_mode)

  def _InitializeSquadEnvironment(self, corpus_dir, corpus_name, corpus_part,
                                  model_dir, nltk_dir):
    """Initilizes the DocQA SquAD model environment.

    Args:
      corpus_dir: Path to corpus directory.
      corpus_name: Name of the corpus, effectively this is a subdirectory in
                   corpus_dir.
      corpus_part: Part of the corpus ("train" or "dev").
      model_dir: Directory containing parameters of a pre-trained DocQA model.
      nltk_dir: Folder containing the nltk package.
    """
    self._environment = docqa_squad.DocqaSquadEnvironment(
        corpus_dir=corpus_dir,
        corpus_name=corpus_name,
        corpus_part=corpus_part,
        model_dir=model_dir,
        nltk_dir=nltk_dir)

  def GetObservations(self, request, context):
    """Returns answers to given questions.

    Passes questions and document ids contained in the request to the Bidaf
    environment and repackages the scored answers coming from the environment
    into the response.

    Args:
      rpc: The rpc object
      request: An EnvironmentRequest containing questions and docids.
      response: An EnvironmentResponse to fill with the resulting answers.
    """
    if self.debug_mode:
      start_time = time.time()

    response = aqa_pb2.EnvironmentResponse()
    if not request.queries:
      context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
      context.set_details('Empty list of queries provided in the request')
      return response

    questions = list()
    document_ids = list()
    for query in request.queries:
      questions.append(query.question)
      document_ids.append(query.id)

    try:
      answers_confidences_scores = self._environment.GetAnswers(
          questions, document_ids)
    except KeyError as e:
      context.set_code(grpc.StatusCode.INTERNAL)
      context.set_details('KeyError: {}'.format(e))
      return response

    # -2 for the entry containing the scores and f1_scores.
    if len(answers_confidences_scores) != len(request.queries):
      context.set_code(grpc.StatusCode.INTERNAL)
      context.set_details('Unexpected number of answers: {} vs. {}'.format(
          len(answers_confidences_scores) - 1, len(request.queries)))
      return response

    for question, document_id, answer_confidence_score in zip(
        questions, document_ids, answers_confidences_scores):
      answer_text, confidence, score = answer_confidence_score
      output_response = response.responses.add()
      output_response.id = document_id
      answer = output_response.answers.add()
      answer.text = answer_text
      answer.scores['confidence'] = confidence
      answer.scores['f1'] = score
      output_response.question = question
      output_response.processed_question = question

    if self.debug_mode:
      logging.info('{} questions processed in {}'.format(
          len(request.queries),
          time.time() - start_time))
    return response


def main(unused_argv):
  logging.info('Loading server...')
  server = grpc.server(
      futures.ThreadPoolExecutor(max_workers=FLAGS.worker_threads))
  aqa_pb2_grpc.add_EnvironmentServerServicer_to_server(
      DocqaServer(
          'active_qa.EnvironmentServer',
          'DocQA environment server',
          precomputed_data_path=FLAGS.precomputed_data_path,
          corpus_dir=FLAGS.corpus_dir,
          corpus_name=FLAGS.corpus_name,
          corpus_part=FLAGS.corpus_part,
          docqa_model_dir=FLAGS.docqa_model_dir,
          nltk_dir=FLAGS.nltk_dir,
          load_test=FLAGS.load_test,
          debug_mode=FLAGS.debug_mode), server)

  port = FLAGS.port
  logging.info('Running server on port {}...'.format(port))

  server.add_insecure_port('[::]:{}'.format(port))


  server.start()


  # Prevent the main thread from exiting.
  try:
    while True:
      time.sleep(FLAGS.sleep_seconds)
  except KeyboardInterrupt:
    server.stop(0)


if __name__ == '__main__':
  app.run(main)
