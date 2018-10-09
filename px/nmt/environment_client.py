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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import OrderedDict
from multiprocessing.pool import ThreadPool

import time


import grpc
import numpy as np
import tensorflow as tf

from px.proto import aqa_pb2
from px.proto import aqa_pb2_grpc


def multi_call_environment(pool, stub, request, timeouts):
  """Make parallel calls to the environment server.

  Args:
    pool: a ThreadPool instance to performe multiple calls. If None, a single
        call will be performed, instead.
    stub: a stub instance
    request: aqa_pb2.EnvironmentRequest instances
    timeouts: a list of timeouts to try.

  Returns:
    responses: aqa_pb2.EnvironmentResponse instances
  """

  flattened_responses = aqa_pb2.EnvironmentResponse()

  if pool is not None:
    all_requests = []

    batch_size = int(np.ceil(len(request.queries) / pool._processes))

    for i in range(0, len(request.queries), batch_size):
      request_i = aqa_pb2.EnvironmentRequest()
      request_i.queries.extend(request.queries[i:i + batch_size])
      all_requests.append(request_i)

    all_responses = pool.map(
        single_call_environment,
        zip(
            len(all_requests) * [stub], all_requests,
            len(all_requests) * [timeouts]))

    for responses in all_responses:
      if responses is not None:
        flattened_responses.responses.extend(responses.responses)
  else:
    responses = single_call_environment((stub, request, timeouts))
    if responses is not None:
      flattened_responses = responses
  return flattened_responses


def single_call_environment(args):
  """Get responses from the environment.

  Args:
    args: a tuple containing:
      - stub: a stub instance
      - request: a aqa_pb2.EnvironmentRequest instance
      - timeouts: a list of timeouts to try.

  Returns:
    responses: aqa_pb2.EnvironmentResponse instances.
  """
  stub, request, timeouts = args
  response = None
  for timeout in timeouts:
    try:
      response = stub.GetObservations(request, timeout=timeout)
      break
    except grpc.RpcError as e:
      tf.logging.warn(e)
      continue  # Try again with the next timeout.

  return response


class LimitedSizeDict(OrderedDict):
  """Specialization of OrderedDict that keeps only the last size_limit added
  items. Used for implementing MRU cache.
  """

  def __init__(self, *args, **kwargs):

    self.size_limit = kwargs.pop('size_limit', None)
    assert self.size_limit >= 0

    OrderedDict.__init__(self, *args, **kwargs)
    self._check_size_limit()

  def __setitem__(self, key, value):
    OrderedDict.__setitem__(self, key, value)
    self._check_size_limit()

  def _check_size_limit(self):
    if self.size_limit is not None:
      while len(self) > self.size_limit:
        # when last=False, FIFO.
        self.popitem(last=False)


def make_cache_key(question, docid):
  """Constructs a cache key using a fixed separator."""
  return question + '###' + docid


def make_environment_reward_fn(environment_server,
                               timeouts=None,
                               mode='squad',
                               use_cache=False,
                               cache_size=-1,
                               env_call_parallelism=1):
  """Creates a function to make environment calls.

  Args:
    environment_server: the address of the environment server.
    timeouts: a list of timeouts (in seconds) to try.
    mode: either "squad" or "sqa" mode.
    use_cache: if True, use cache.
    cache_size: maximum number of items to hold in the cache.
    env_call_parallelism: number of parallel calls.
  Returns:
    A function to queries the environment server.
  """
  channel = grpc.insecure_channel(environment_server)


  grpc.channel_ready_future(channel).result(timeout=30)
  stub = aqa_pb2_grpc.EnvironmentServerStub(channel)

  pool = None
  if env_call_parallelism > 1:
    pool = ThreadPool(processes=env_call_parallelism)

  if timeouts is None:
    timeouts = [120, 300, 300, 300]

  if use_cache:
    cache = LimitedSizeDict(size_limit=cache_size)
  else:
    cache = None

  def environment_reward_fn(questions, doc_ids):
    """Queries the environment server and generates F1 scores using rewritten
    questions.

    Args:
      questions: A numpy array of strings (assuming utf-8 encoding).
                        Dimension [batch_size].
      doc_ids: A numpy array of strings. Dimension [batch_size].

    Returns:
      Numpy array of floats. Dimension [batch_size] containing F1 scores.
      Numpy array of floats. Dimension [batch_size] containing model scores.
      Numpy array of strings. Dimension [batch_size] containing the answers.
    """
    start_time = time.time()
    batch_size = questions.shape[0]
    assert len(doc_ids) == batch_size

    request = aqa_pb2.EnvironmentRequest()
    cache_hit = 0
    for question, docid in zip(questions, doc_ids):

      needs_request = True
      if cache and make_cache_key(question, docid) in cache:
        needs_request = False
        cache_hit += 1

      if needs_request:
        q = request.queries.add()
        q.question = question
        q.id = docid

        if mode == 'search':
          # In search mode, the original question is passed in using the docid
          # Tensor.
          q.original_question = docid

    if mode == 'search':
      scores = [-3.0] * batch_size
    else:
      scores = [0.0] * batch_size
    model_scores = [0.0] * batch_size
    answers = [''] * batch_size

    answered = [False] * batch_size

    if mode == 'search':
      reward_key = 'environment_conf'
    else:
      reward_key = 'f1'

    if cache is not None:
      cache_new = {}

    # only call if there is at least one request
    if request.queries:
      response = multi_call_environment(
          pool=pool, stub=stub, request=request, timeouts=timeouts)
      if len(response.responses) != len(request.queries):
        tf.logging.error('Mismatch in response size {} vs. {}'.format(
            len(response.responses), len(request.queries)))

    for i, (question, docid) in enumerate(zip(questions, doc_ids)):
      if cache is not None:
        key = make_cache_key(question, docid)
        if key in cache:
          score, model_score, answer = cache[key]
          scores[i] = score
          answers[i] = answer
          if mode != 'search':
            model_scores[i] = model_score
          answered[i] = True

      # only look in the responses if it was not in the cache
      if not answered[i]:
        try:
          for response_i in response.responses:
            # Webanswers may not answer the question, in these cases the
            # error message field is populated.
            if mode == 'search' and response_i.error_message != '':
              tf.logging.info(u'RPC error: {}'.format(
                  response_i.error_message).encode('utf-8'))
              continue

            assert len(response_i.answers) == 1
            if response_i.id == docid and response_i.question == question:
              scores[i] = response_i.answers[0].scores[reward_key]
              if mode != 'search':
                model_scores[i] = response_i.answers[0].scores[
                    'environment_confidence']
              answers[i] = response_i.answers[0].text
              answered[i] = True
              if cache is not None:
                key = make_cache_key(question, docid)
                cache_new[key] = (scores[i], model_scores[i], answers[i])
              break
        except UnicodeDecodeError as e:
          tf.logging.error(e)
          tf.logging(response_i.answers[0].text)
        except UnicodeEncodeError as e:
          tf.logging.error(e)
          tf.logging(response_i.answers[0].text)
        except TypeError as e:
          tf.logging.error(e)
          tf.logging(response_i.answers[0].text)

    # update cache with the new items:
    if cache is not None:
      for k, v in cache_new.items():
        cache[k] = v

    # print a sample of 10 replies
    for i in range(len(answered))[:10]:
      try:
        if not answered[i]:
          tf.logging.info(u'Unanswered: {} : {} : {} : {}'.format(
              i, questions[i], doc_ids[i], scores[i]).encode('utf-8'))
        else:
          tf.logging.info(u'Answered: {} : {} : {} : {} : {}'.format(
              i, questions[i], answers[i], doc_ids[i],
              scores[i]).encode('utf-8'))
      except UnicodeEncodeError as e:
        tf.logging.error(e)
        tf.logging.error(questions[i])
      except UnicodeDecodeError as e:
        tf.logging.error(e)
        tf.logging.error(questions[i])
      except TypeError as e:
        tf.logging.error(e)
        tf.logging.error(questions[i])

    if cache is not None:
      tf.logging.info(u'Cache size current/limit: {}/{}'.format(
          len(cache), cache_size))

      tf.logging.info(u'Cache hit/total: {}/{}'.format(cache_hit,
                                                       len(questions)))

    tf.logging.info(u'Time to make {} environment calls: {}'.format(
        len(request.queries),
        time.time() - start_time))

    return (np.array(scores, dtype=np.float32),
            np.array(model_scores, dtype=np.float32),
            np.array(answers, dtype=np.object))

  return environment_reward_fn
