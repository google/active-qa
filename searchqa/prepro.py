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

"""Converts a dataset in SearchQA text format to SQuAD format.

Does minimal processing of the questions, answers and snippets:

  i)   Removes examples with an empty question.
  ii)  Removes long answers (to avoid answers with long words such as
       'supercalifragilisticexpialidocious'.
  iii) Adds a unique identification number for each question.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import re

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('searchqa_dir', '',
                    'Directory containing the SearchQA datasets')
flags.DEFINE_string('squad_dir', '',
                    'Location where the SQuAD output will be written.')
flags.DEFINE_integer('max_snippets', 10,
                     'If greater than 0, use at most this many snippets.')


def process(sqa_file, squad_path, questions_path, annotation_path, init_id):
  """Converts the given SearchQA txt file to SQuAD format.

  This writes out 3 output files: SQuAD JSON file, questions txt file, and
  annotation txt file.
  """

  questions_file = open(questions_path, 'w+')
  annotation_file = open(annotation_path, 'w+')

  with_answer_count = 0
  with_unigram_answer_count = 0
  without_answer_count = 0
  without_unigram_answer_count = 0
  answer_too_long_count = 0
  squad_dict = {'data': []}
  current_id = init_id
  with open(sqa_file) as f:
    for line in f:
      id_string = str(current_id)

      snippets, question, answer = [x.strip() for x in line.split('|||')]

      # Check if snippets and answer are present. There are some data points
      # with an empty question, and these are just skipped.
      if not question:
        continue
      assert (snippets), (id_string, line)
      assert (answer), (id_string, line)

      snippets = re.findall('<s>(.*?)</s>', snippets)
      context = ''
      if FLAGS.max_snippets > 0:
        context = 'SEP'.join(snippets[:FLAGS.max_snippets])
      else:
        context = 'SEP'.join(snippets)
      context = 'START' + context + 'END'

      # Skip data points with a long answer because it breaks the BiDAF
      # preprocessing.
      if (len(answer)) > 32:
        without_answer_count += 1
        answer_too_long_count += 1
        if len(answer.split()) == 1:
          without_unigram_answer_count += 1
        continue

      answer_for_match = ' ' + re.escape(answer) + ' '
      if not re.search(answer_for_match, context):
        without_answer_count += 1
        if len(answer.split()) == 1:
          without_unigram_answer_count += 1
        continue

      with_answer_count += 1
      if len(answer.split()) == 1:
        with_unigram_answer_count += 1

      answers = [{
          'answer_start': pos.start() + 1,
          'text': answer
      } for pos in re.finditer(answer_for_match, context)]

      qas_dict = {'answers': answers, 'id': id_string, 'question': question}
      squad_dict['data'].append({
          'paragraphs': [{
              'context': context,
              'qas': [qas_dict]
          }]
      })

      questions_file.write('%s\n' % question)
      annotation_file.write('%s\t1\n' % id_string)

      current_id += 1

    print('Processed {}.'.format(sqa_file))
    print('With answer in snippets: {}'.format(with_answer_count))
    print('Unigram answer, with answer in snippets: {}'.format(
        with_unigram_answer_count))
    print('Without answer in snippets or too long: {}'.format(
        without_answer_count))
    print('Unigram answer, without answer in snippets or too long: {}'.format(
        without_unigram_answer_count))
    print('Answer too long: {}'.format(answer_too_long_count))

    json.dump(squad_dict, open(squad_path, 'w'))
    print('Written {}.'.format(squad_path))

    questions_file.close()
    print('Written {}.'.format(questions_path))

    annotation_file.close()
    print('Written {}.'.format(annotation_path))


def main(argv=()):
  del argv  # Unused.

  assert os.path.exists(FLAGS.searchqa_dir)
  assert os.path.exists(FLAGS.squad_dir)

  for (sqa_file, squad_file, questions_file, annotation_file,
       init_id) in [('val.txt', 'dev-v1.1.json', 'dev-questions.txt',
                     'dev-annotation.txt', 0),
                    ('test.txt', 'test-v1.1.json', 'test-questions.txt',
                     'test-annotation.txt', 100000),
                    ('train.txt', 'train-v1.1.json', 'train-questions.txt',
                     'train-annotation.txt', 200000)]:
    sqa_path = os.path.join(FLAGS.searchqa_dir, sqa_file)
    assert os.path.exists(sqa_path)

    squad_path = os.path.join(FLAGS.squad_dir, squad_file)
    questions_path = os.path.join(FLAGS.squad_dir, questions_file)
    annotation_path = os.path.join(FLAGS.squad_dir, annotation_file)
    process(sqa_path, squad_path, questions_path, annotation_path, init_id)


if __name__ == '__main__':
  app.run(main)
