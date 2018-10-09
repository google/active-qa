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

r"""Main to demo the environment server.

Demo to query an environment server with different questions and context
documents and receive back answers.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time


from absl import app
from absl import flags

import numpy as np

import px.nmt.environment_client as environment_client

flags.DEFINE_string('server_address', '', 'Address of the environment server.')
flags.DEFINE_string(
    'mode', 'squad',
    'Operation mode. Valid options are "squad", "sqa" or "search".')
FLAGS = flags.FLAGS


def main(argv=()):
  del argv  # Unused.

  if not FLAGS.server_address:
    sys.exit('A server address must be provided.')

  env_get_answers_fn = environment_client.make_environment_reward_fn(
      FLAGS.server_address, mode=FLAGS.mode)

  while True:
    questions_doc_ids = raw_input(
        'Enter question and doc id (use %%% to separate question and '
        'doc id and use ### to separate multiple questions): ')

    start_time = time.time()
    if not questions_doc_ids:
      continue

    questions = []
    doc_ids = []
    for question_doc_id in questions_doc_ids.split('###'):
      question, doc_id = question_doc_id.split('%%%')
      questions.append(question.strip())
      doc_ids.append(doc_id.strip())

    _, _, answers = env_get_answers_fn(np.array(questions), np.array(doc_ids))
    for question, answer in zip(questions, answers):
      print(u'question: {}\nanswer: {}'.format(
          question.decode('utf8'), answer.decode('utf8')))
    print('{0} questions reformulated in {1:.2f} s'.format(
        len(questions),
        time.time() - start_time))


if __name__ == '__main__':
  app.run(main)
