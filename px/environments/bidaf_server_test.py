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

"""Tests for the BidafServer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl import logging
import grpc
from grpc.framework.foundation import logging_pool
import mock
import portpicker

import unittest
from tensorflow.python.util.protobuf import compare

from px.environments import bidaf
from px.environments import bidaf_server
from px.proto import aqa_pb2
from px.proto import aqa_pb2_grpc

DOCID_SEPARATOR = '###'

DOCID1 = '123'
DOCID2 = '234'
QUESTION1 = 'A Question 1'
QUESTION2 = 'A Question 2'
PROCESSED1 = 'a question 1'
PROCESSED2 = 'a question 2'
ANSWER1 = 'an answer 1'
ANSWER2 = 'an answer 2'
SCORE1 = 1.0
SCORE2 = 2.0
F1_SCORE1 = 0.1
F1_SCORE2 = 0.2
GOOD_REQUEST = aqa_pb2.EnvironmentRequest()
BAD_REQUEST = aqa_pb2.EnvironmentRequest()
EMPTY_REQUEST = aqa_pb2.EnvironmentRequest()
DUPLICATE_IDS_REQUEST = aqa_pb2.EnvironmentRequest()
GOOD_RESPONSE = aqa_pb2.EnvironmentResponse()
DUPLICATE_IDS_RESPONSE = aqa_pb2.EnvironmentResponse()


def setUpModule():
  query = GOOD_REQUEST.queries.add()
  query.id = DOCID1
  query.question = QUESTION1
  query = GOOD_REQUEST.queries.add()
  query.id = DOCID2
  query.question = QUESTION2

  query = BAD_REQUEST.queries.add()
  query.id = 'bad docid'
  query.question = 'bad question'

  query = DUPLICATE_IDS_REQUEST.queries.add()
  query.id = DOCID1
  query.question = QUESTION1
  query = DUPLICATE_IDS_REQUEST.queries.add()
  query.id = DOCID1
  query.question = QUESTION2

  response = GOOD_RESPONSE.responses.add()
  response.id = DOCID1
  answer = response.answers.add()
  answer.text = ANSWER1
  answer.scores['environment_confidence'] = SCORE1
  answer.scores['f1'] = F1_SCORE1
  response.question = QUESTION1
  response.processed_question = PROCESSED1

  response = GOOD_RESPONSE.responses.add()
  response.id = DOCID2
  answer = response.answers.add()
  answer.text = ANSWER2
  answer.scores['environment_confidence'] = SCORE2
  answer.scores['f1'] = F1_SCORE2
  response.question = QUESTION2
  response.processed_question = PROCESSED2

  response = DUPLICATE_IDS_RESPONSE.responses.add()
  response.id = DOCID1
  answer = response.answers.add()
  answer.text = ANSWER1
  answer.scores['environment_confidence'] = SCORE1
  answer.scores['f1'] = F1_SCORE1
  response.question = QUESTION1
  response.processed_question = PROCESSED1

  response = DUPLICATE_IDS_RESPONSE.responses.add()
  response.id = DOCID1  # Same docid.
  answer = response.answers.add()
  answer.text = ANSWER2
  answer.scores['environment_confidence'] = SCORE2
  answer.scores['f1'] = F1_SCORE2
  response.question = QUESTION2
  response.processed_question = PROCESSED2


class MockBidafEnvironment(bidaf.BidafEnvironment):

  def __init__(self):
    pass

  def GetAnswers(self, questions, document_ids):
    # Returns valid answers to GOOD_REQUEST and DUPLICATE_IDS_REQUEST.
    expected_id1_0 = '{:s}{:s}{:d}'.format(DOCID1, DOCID_SEPARATOR, 0)
    expected_id1_1 = '{:s}{:s}{:d}'.format(DOCID1, DOCID_SEPARATOR, 1)
    expected_id2_1 = '{:s}{:s}{:d}'.format(DOCID2, DOCID_SEPARATOR, 1)
    questions_dict = {}
    answers_dict = collections.OrderedDict()
    logging.info(document_ids)
    if (len(document_ids) == 2 and document_ids[0] == expected_id1_0 and
        document_ids[1] == expected_id2_1):
      answers_dict.update([
          ('scores', {
              document_ids[0]: SCORE1,
              document_ids[1]: SCORE2
          }),
          ('f1_scores', {
              document_ids[0]: F1_SCORE1,
              document_ids[1]: F1_SCORE2
          }),
          (document_ids[0], ANSWER1),
          (document_ids[1], ANSWER2)
      ])
      questions_dict = {
          document_ids[0]: {
              'rewrite': PROCESSED1,
              'raw_rewrite': QUESTION1
          },
          document_ids[1]: {
              'rewrite': PROCESSED2,
              'raw_rewrite': QUESTION2
          }
      }
    elif (len(document_ids) == 2 and document_ids[0] == expected_id1_0 and
          document_ids[1] == expected_id1_1):
      answers_dict.update([
          ('scores', {
              document_ids[0]: SCORE1,
              document_ids[1]: SCORE2
          }),
          ('f1_scores', {
              document_ids[0]: F1_SCORE1,
              document_ids[1]: F1_SCORE2
          }),
          (document_ids[0], ANSWER1),
          (document_ids[1], ANSWER2)
      ])
      questions_dict = {
          document_ids[0]: {
              'rewrite': PROCESSED1,
              'raw_rewrite': QUESTION1
          },
          document_ids[1]: {
              'rewrite': PROCESSED2,
              'raw_rewrite': QUESTION2
          }
      }

    return answers_dict, questions_dict, 0


class BidafServerTest(unittest.TestCase):

  def setUp(self):
    with mock.patch.object(bidaf_server.BidafServer,
                           '_InitializeEnvironment') as mock_method:
      port = portpicker.pick_unused_port()
      server_pool = logging_pool.pool(max_workers=10)
      self._server = grpc.server(server_pool)
      self._server.add_insecure_port('[::]:{}'.format(port))
      servicer = bidaf_server.BidafServer('BidafServer', 'test BiDAF server')
      servicer._environment = MockBidafEnvironment()

      aqa_pb2_grpc.add_EnvironmentServerServicer_to_server(
          servicer, self._server)
      self._server.start()

      channel = grpc.insecure_channel('localhost:%d' % port)
      self._stub = aqa_pb2_grpc.EnvironmentServerStub(channel)

      mock_method.assert_called_once_with(
          model_dir=None,
          data_dir=None,
          debug_mode=False,
          load_test=False,
          load_impossible_questions=False,
          shared_file=None)

  def tearDown(self):
    self._server.stop(None)

  def testValidRequest(self):
    response = self._stub.GetObservations(GOOD_REQUEST)
    self.assertIsInstance(response, aqa_pb2.EnvironmentResponse)
    compare.assertProtoEqual(self, response, GOOD_RESPONSE)

  def testNoQueries(self):
    with self.assertRaises(grpc.RpcError) as call_context:
      self._stub.GetObservations(EMPTY_REQUEST)
    self.assertEqual(call_context.exception.code(),
                     grpc.StatusCode.INVALID_ARGUMENT)
    self.assertEqual(call_context.exception.details(),
                     'Empty list of queries provided in the request')

  def testDuplicateIds(self):
    response = self._stub.GetObservations(DUPLICATE_IDS_REQUEST)
    self.assertIsInstance(response, aqa_pb2.EnvironmentResponse)
    compare.assertProtoEqual(self, response, DUPLICATE_IDS_RESPONSE)

  def testInvalidResponse(self):
    with self.assertRaises(grpc.RpcError) as call_context:
      self._stub.GetObservations(BAD_REQUEST)
    self.assertEqual(call_context.exception.code(), grpc.StatusCode.INTERNAL)
    self.assertEqual(call_context.exception.details(),
                     'Unexpected number of answers: -1 vs. 1')


if __name__ == '__main__':
  unittest.main()
