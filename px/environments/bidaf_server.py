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
r"""gRPC server for the BiDAF environment.

Implementation of a gRPC server for the BiDAF model. Requests contain
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

from px.environments import bidaf
from px.proto import aqa_pb2
from px.proto import aqa_pb2_grpc

FLAGS = flags.FLAGS

flags.DEFINE_integer('port', 10000, 'Port to listen on.')
flags.DEFINE_string('squad_data_dir', '', 'Directory containing squad data.')
flags.DEFINE_string('bidaf_shared_file', '', 'Shared file produced by BiDAF.')
flags.DEFINE_string('bidaf_model_dir', '', 'Directory of trained BiDAF model.')
flags.DEFINE_integer('worker_threads', 10,
                     'Number of worker threads running on the server.')
flags.DEFINE_integer('sleep_seconds', 10,
                     'Number of seconds to wait for a termination event.')
flags.DEFINE_bool('load_test', False,
                  'Load test data in addition to dev and train.')
flags.DEFINE_bool(
    'load_impossible_questions', False, 'For SQuAD v2 impossible '
    'questions can be loaded to return a modified reward.')
flags.DEFINE_bool('debug_mode', False,
                  'If true, log questions, answers, and scores.')

DOCID_SEPARATOR = '###'


class BidafServer(aqa_pb2_grpc.EnvironmentServerServicer):
  """A gRPC server for the BiDAF environment.

  Attributes:
    environment: A BidafEnvironment object that returns scored answers to
                 questions.
  """

  def __init__(self, *args, **kwargs):
    """"Constructor for the BiDAF server."""
    data_dir = kwargs.pop('squad_data_dir', None)
    shared_file = kwargs.pop('bidaf_shared_file', None)
    model_dir = kwargs.pop('bidaf_model_dir', None)
    load_test = kwargs.pop('load_test', False)
    load_impossible_questions = kwargs.pop('load_impossible_questions', False)
    debug_mode = kwargs.pop('debug_mode', False)
    self.debug_mode = debug_mode
    self._InitializeEnvironment(
        data_dir=data_dir,
        shared_file=shared_file,
        model_dir=model_dir,
        load_test=load_test,
        load_impossible_questions=load_impossible_questions,
        debug_mode=debug_mode)

  def _InitializeEnvironment(self, data_dir, shared_file, model_dir, load_test,
                             load_impossible_questions, debug_mode):
    """Initilizes the BiDAF model environment.

    Args:
      data_dir: Directory containing preprocessed SQuAD data.
      shared_file: Path to shared data generated at training time.
      model_dir: Directory contining parameters of a pre-trained BiDAF
                 model.
      load_test: Whether the test set should be loaded as well.
      load_impossible_questions: Whether info about impossibility of questions
                                 should be loaded.
      debug_mode: Whether to log debug information.
    """
    self._environment = bidaf.BidafEnvironment(
        data_dir,
        shared_file,
        model_dir,
        docid_separator=DOCID_SEPARATOR,
        load_test=load_test,
        load_impossible_questions=load_impossible_questions,
        debug_mode=debug_mode)

  def GetObservations(self, request, context):
    """Returns answers to given questions.

    Passes questions and document ids contained in the request to the Bidaf
    environment and repackages the scored answers coming from the environment
    into the response.

    Args:
      request: An EnvironmentRequest containing questions and docids.
      context: The RPC context.

    Returns:
      The EnvironmentResponse filled with the resulting answers.
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
    index = 0
    impossible_ids = set()
    for query in request.queries:
      questions.append(query.question)
      if query.is_impossible:
        impossible_ids.add(query.id)
      # Add an index to each id to make them unique, as required by BiDAF. This
      # augmentation of the docid is for BiDAF internal use and is not visible
      # to the client.
      unique_id = u'{:s}{:s}{:d}'.format(query.id, DOCID_SEPARATOR, index)
      index += 1
      document_ids.append(unique_id)
    if self.debug_mode:
      logging.info('Batch contains %s impossible questions.',
                   len(impossible_ids))

    try:
      answer_dict, questions_dict, _ = self._environment.GetAnswers(
          questions, document_ids)
    except KeyError as e:
      context.set_code(grpc.StatusCode.INTERNAL)
      context.set_details('KeyError: {}'.format(e))
      return response

    # -2 for the entry containing the scores and f1_scores.
    if not len(answer_dict) - 2 == len(request.queries):
      context.set_code(grpc.StatusCode.INTERNAL)
      context.set_details('Unexpected number of answers: {} vs. {}'.format(
          len(answer_dict) - 1, len(request.queries)))
      return response

    for docid in answer_dict.iterkeys():
      if docid == 'scores' or docid == 'f1_scores':
        continue
      output_response = response.responses.add()
      output_response.id = docid.split(DOCID_SEPARATOR)[0]
      answer = output_response.answers.add()
      answer.text = answer_dict[docid]
      answer.scores['environment_confidence'] = answer_dict['scores'][docid]
      output_response.question = questions_dict[docid]['raw_rewrite']
      output_response.processed_question = questions_dict[docid]['rewrite']

      # Set an F1 score of 1.0 for impossible questions if the is_impossible
      # flag was set to true in the request. If is_impossible is set for
      # possible questions an F1 score of 0.0 is returned.
      if output_response.id in impossible_ids:
        if self._environment.IsImpossible(output_response.id):
          answer.scores['f1'] = 1.0
        else:
          answer.scores['f1'] = 0.0
      else:
        answer.scores['f1'] = answer_dict['f1_scores'][docid]

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
      BidafServer(
          'active_qa.EnvironmentServer',
          'BiDAF environment server',
          squad_data_dir=FLAGS.squad_data_dir,
          bidaf_shared_file=FLAGS.bidaf_shared_file,
          bidaf_model_dir=FLAGS.bidaf_model_dir,
          load_test=FLAGS.load_test,
          load_impossible_questions=FLAGS.load_impossible_questions,
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
