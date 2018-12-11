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
# pylint: disable=g-line-too-long
r"""Main for training the reformulator and the selector.

Additional flags defined in selector_keras.py:
--glove_path: Path to pretrained Glove embeddings.
--save_path: Directory where models will be saved to/loaded from.

"""
# pylint: enable=g-line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import json
import os
import random
import time

from absl import app
from absl import flags
import numpy as np

from px.nmt import environment_client
from px.nmt import reformulator
from px.nmt.utils.logging_utils import safe_string
from px.proto import reformulator_pb2
from px.selector import selector_keras as selector
from px.utils import eval_utils
import tensorflow as tf
from third_party.nmt.utils import misc_utils

flags.DEFINE_string('environment_server_address', '',
                    'Address of the environment server.')
flags.DEFINE_string('train_questions', '',
                    'Path to train questions file (not SPM tokenized).')
flags.DEFINE_string('train_annotations', '', 'Path to train annotation file.')
flags.DEFINE_string('train_data', '', 'Path to train answers file.')
flags.DEFINE_string('dev_questions', '',
                    'Path to dev questions file (not SPM tokenized).')
flags.DEFINE_string('dev_annotations', '', 'Path to dev annotation file.')
flags.DEFINE_string('dev_data', '', 'Path to dev answers file.')
flags.DEFINE_integer('max_dev_examples', 0,
                     'If >0 maximum number of lines read from dev.')
flags.DEFINE_integer('batch_size_train', 64, 'Batch size for training')
flags.DEFINE_integer('batch_size_eval', 64, 'Batch size for eval.')
flags.DEFINE_integer('epochs', 1, 'Number of training epochs.')
flags.DEFINE_integer('num_steps_per_eval', 500,
                     'Number of steps to train for each eval run.')
flags.DEFINE_integer('steps_per_save_selector', 500,
                     'Number of steps to save a checkpoint for the selector.')
flags.DEFINE_string('mode', 'searchqa', 'Which QA dataset to preprocess for.')
flags.DEFINE_string('hparams_path', None, 'Path to the json hparams file.')
flags.DEFINE_string(
    'source_prefix', '<en> <2en> ',
    'This prefix is attached to every source sentence. '
    'Note than no extra space is added after the prefix.')
flags.DEFINE_integer(
    'env_sample_parallelism', 1,
    'How many parallel calls to make to the environment when sampling.')
flags.DEFINE_integer(
    'env_eval_parallelism', 21,
    'How many processes to use to make environment calls during eval.')
flags.DEFINE_bool('enable_reformulator_training', True,
                  'Whether to enable reformulator training.')
flags.DEFINE_bool('enable_selector_training', True,
                  'Whether to enable selector training.')
flags.DEFINE_bool('debug', False, 'Whether to print extra debug statements.')
flags.DEFINE_string('tensorboard_dir', '',
                    'Directory to store log/event files for Tensorboard.')
flags.DEFINE_string('out_dir', None, 'Directory where model output is written.')


FLAGS = flags.FLAGS


def read_data(questions_file,
              annotations_file,
              answers_file,
              preprocessing_mode,
              max_lines=None):
  """Creates questions, annotations, and answers lists from files."""
  with codecs.getreader('utf-8')(tf.gfile.Open(questions_file, 'rb')) as f:
    questions = [line.strip() for line in f if line.strip()]
  with codecs.getreader('utf-8')(tf.gfile.Open(annotations_file, 'rb')) as f:
    annotations = [line.strip() for line in f]

  if preprocessing_mode == 'searchqa':
    questions = [
        q.lower().replace(u'.', u' ').replace(u',', u' , ') for q in questions
    ]
    questions = [' '.join(q.split()) for q in questions]

  annotations = [a.split('\t')[0] for a in annotations]
  assert len(questions) == len(annotations), '{} vs {}'.format(
      len(questions), len(annotations))
  if max_lines:
    annotations = annotations[:max_lines]
    questions = questions[:max_lines]

  with tf.gfile.Open(answers_file) as f:
    json_obj = json.load(f)
    docid_2_answer = {
        docid: answers[0]
        for answers, docid in zip(json_obj['answerss'], json_obj['ids'])
    }

  return questions, annotations, docid_2_answer


def batch(questions, annotations, batch_size):
  """Generator function producing shuffled and batched questions/annotations."""
  data = zip(questions, annotations)
  random.shuffle(data)
  for index in range(0, len(questions), batch_size):
    yield zip(*data[index:index + batch_size])


def query_environment(original_questions, rewrites, annotations, environment_fn,
                      docid_2_answer, token_level_f1_scores):
  """Queries the environment.

  We first prepend original questions to the rewrites. Then we flatten this
  nested list of rewrites, whose lengths are [batch_size, n_rewrites + 1], to a
  list of length [batch_size * (n_rewrites + 1)], which we then use to query the
  environment.
  The flattening is necessary because the environment does not accept nested
  lists as input. Finally, we reshape the returned answers to
  [batch_size, n_rewrites + 1].

  Args:
    original_questions: a list of strings of length [batch_size].
    rewrites: a nested list of strings of lengths [batch_size, n_rewrites].
    annotations: a list of strings of length [batch_size].
    environment_fn: the environment function that will be used to get answers to
      the queries.
    docid_2_answer: a dictionary that maps doc id (annotation) to ground-truth
      answers.
    token_level_f1_scores: a boolean. If True, F1 scores are computed by the
      token-level intersection between the predicted and ground-truth answers.
      If False, F1 scores are from the BiDAF environment, which uses the
      intersection between predicted answer span and ground-truth spans.

  Returns:
    originals_and_rewrites: a nested list of strings of lengths
      [batch_size, n_rewrites + 1].
    answers: a nested list of strings of lengths [batch_size, n_rewrites + 1].
    f1_scores: a float32 array of shape [batch_size, n_rewrites + 1]
      representing the scores for each answer.
  """

  assert len(set(map(len, rewrites))) == 1, (
      'Not all examples have the same number of rewrites: {}'.format(rewrites))
  # Prepend original question to the list of rewrites.
  originals_and_rewrites = np.array([
      [original_question] + list(rewrite_list)
      for original_question, rewrite_list in zip(original_questions, rewrites)
  ])
  # Expand annotations so they have the same shape of rewrites:
  # [batch_size, n_rewrites + 1].
  annotations = np.array([[annotation] * originals_and_rewrites.shape[1]
                          for annotation in annotations]).flatten()

  f1_scores, _, answers = environment_fn(originals_and_rewrites.flatten(),
                                         annotations)

  assert len(annotations) == len(answers)

  if token_level_f1_scores:
    f1_scores = np.array([
        eval_utils.compute_f1_single(
            prediction=answer, ground_truth=docid_2_answer[annotation])
        for annotation, answer in zip(annotations, answers)
    ])

  # Reshape to [batch_size, n_rewrites + 1].
  f1_scores = f1_scores.reshape((len(original_questions), -1))
  answers = answers.reshape((len(original_questions), -1))

  return originals_and_rewrites, answers, f1_scores


def _run_reformulator_eval(questions, annotations, reformulator_instance,
                           environment_fn, batch_size):
  """Runs eval with just the reformulator, using greedy decoding."""
  f1s = []
  for (questions_batch, annotations_batch) in batch(questions, annotations,
                                                    batch_size):
    responses = reformulator_instance.reformulate(
        questions=questions_batch,
        inference_mode=reformulator_pb2.ReformulatorRequest.GREEDY)

    # Discard answers and flatten list.
    reformulations = [r[0].reformulation for r in responses]

    # Get scores from the environment.
    f1_scores, _, _ = environment_fn(
        np.asarray(reformulations), np.asarray(annotations_batch))
    f1s.extend(f1_scores)

  return np.mean(f1s)


def _run_eval_with_selector(questions, annotations, docid_2_answer,
                            reformulator_instance, selector_model, batch_size,
                            environment_fn):
  """Runs a joined eval with the reformulator and selector model."""
  f1s = []
  for batch_id, (questions_batch, annotations_batch) in enumerate(
      batch(questions, annotations, batch_size)):
    responses = reformulator_instance.reformulate(
        questions=questions_batch,
        inference_mode=reformulator_pb2.ReformulatorRequest.BEAM_SEARCH)

    # Discard answers.
    reformulations = [[rf.reformulation for rf in rsp] for rsp in responses]

    question_and_rewrites, answers, scores = query_environment(
        original_questions=questions_batch,
        rewrites=reformulations,
        annotations=annotations_batch,
        environment_fn=environment_fn,
        docid_2_answer=docid_2_answer,
        token_level_f1_scores=True)
    f1s.append(selector_model.eval(question_and_rewrites, answers, scores))
    if FLAGS.debug and batch_id == 0:
      print('Running Eval...')
      print('Questions: {}, Annotation: {}'.format(
          safe_string(questions_batch[0]), safe_string(annotations_batch[0])))
      print('Rewrites: {}'.format(safe_string(reformulations[0])))
      print('Answers and Scores: {}'.format(
          zip(safe_string(answers[0]), scores[0])))

  return np.mean(f1s)


def _correct_searchqa_score(x, dataset):
  """Method to correct for deleted datapoints in the sets.

  Args:
    x: number to correct
    dataset: string that identifies the correction to make

  Returns:
    The rescaled score x.

  Raises:
    ValueError: if dataset is none of train, dev, test.
  """
  if dataset == 'train':
    return x * 90843 / (90843 + 8977)
  elif dataset == 'dev':
    return x * 12635 / (12635 + 1258)
  elif dataset == 'test':
    return x * 24660 / (24660 + 2588)
  else:
    raise ValueError('Unexepected value for dataset: {}'.format(dataset))


def main(argv):
  del argv  # Unused.

  if FLAGS.debug:
    random.seed(0)

  reformulator_instance = reformulator.Reformulator(
      hparams_path=FLAGS.hparams_path,
      source_prefix=FLAGS.source_prefix,
      out_dir=FLAGS.out_dir,
      environment_server_address=FLAGS.environment_server_address)
  environment_fn = environment_client.make_environment_reward_fn(
      FLAGS.environment_server_address,
      mode=FLAGS.mode,
      env_call_parallelism=FLAGS.env_sample_parallelism)

  eval_environment_fn = environment_client.make_environment_reward_fn(
      FLAGS.environment_server_address,
      mode='searchqa',
      env_call_parallelism=FLAGS.env_eval_parallelism)

  # Read data.
  questions, annotations, docid_2_answer = read_data(
      questions_file=FLAGS.train_questions,
      annotations_file=FLAGS.train_annotations,
      answers_file=FLAGS.train_data,
      preprocessing_mode=FLAGS.mode)
  dev_questions, dev_annotations, dev_docid_2_answer = read_data(
      questions_file=FLAGS.dev_questions,
      annotations_file=FLAGS.dev_annotations,
      answers_file=FLAGS.dev_data,
      preprocessing_mode=FLAGS.mode,
      max_lines=FLAGS.max_dev_examples)

  # Summary writer that writes events to a folder. TensorBoard will later read
  # from it.
  summary_writer = tf.summary.FileWriter(
      os.path.join(
          FLAGS.tensorboard_dir,
          'reformulator_and_selector_training_log_' + str(time.time())))

  if FLAGS.enable_selector_training:
    selector_model = selector.Selector()
    last_save_step = 0

  global_step = 0
  for epoch in range(FLAGS.epochs):
    for batch_id, (questions_batch, annotations_batch) in enumerate(
        batch(questions, annotations, FLAGS.batch_size_train)):
      # Run eval every num_steps_per_eval batches.
      if global_step % FLAGS.num_steps_per_eval is 0:
        if FLAGS.debug:
          print('Running eval...')
        eval_start_time = time.time()

        if not FLAGS.enable_selector_training:
          eval_f1_avg = _run_reformulator_eval(
              dev_questions, dev_annotations, reformulator_instance,
              environment_fn, FLAGS.batch_size_eval)
        else:
          eval_f1_avg = _run_eval_with_selector(
              questions=dev_questions,
              annotations=dev_annotations,
              docid_2_answer=dev_docid_2_answer,
              reformulator_instance=reformulator_instance,
              selector_model=selector_model,
              batch_size=FLAGS.batch_size_eval,
              environment_fn=eval_environment_fn)

        # Correct the average F1 score for deleted datapoints in the SearchQA
        # dataset.
        if FLAGS.mode == 'searchqa':
          eval_f1_avg = _correct_searchqa_score(eval_f1_avg, dataset='dev')

        eval_time = time.time() - eval_start_time

        misc_utils.add_summary(
            summary_writer, global_step, tag='eval_f1_avg', value=eval_f1_avg)
        misc_utils.add_summary(
            summary_writer, global_step, tag='eval_time', value=eval_time)

        if FLAGS.debug:
          print('Avg F1 on dev: {}.'.format(eval_f1_avg))
          print('Time to finish eval: {}'.format(eval_time))

      start_time = time.time()
      if FLAGS.debug:
        print('Epoch {}, Batch {}.'.format(epoch, batch_id))
        print('Question: [{}]; Id: {}'.format(questions_batch[0],
                                              annotations_batch[0]))

      # Retrieve rewrites for selector training using beam search.
      if FLAGS.enable_selector_training:
        responses_beam = reformulator_instance.reformulate(
            questions=questions_batch,
            inference_mode=reformulator_pb2.ReformulatorRequest.BEAM_SEARCH)

        # Discard answers.
        reformulations_beam = [
            [rf.reformulation for rf in rsp] for rsp in responses_beam
        ]

      if FLAGS.enable_reformulator_training:
        # Train reformulator model.
        if FLAGS.debug:
          print('Training reformulator...')

        reformulator_loss, f1s, reformulations = reformulator_instance.train(
            sources=questions_batch, annotations=annotations_batch)

        f1_avg = f1s.mean()

        if [] in reformulations:
          if FLAGS.debug:
            print('Found empty rewrites! Skipping this batch.')
          continue

        if FLAGS.debug:
          print('Rewrite: {}'.format(safe_string(reformulations[0])))
          print('Avg F1: {}'.format(f1_avg))
          print('Loss  : {}'.format(reformulator_loss))

        # Write the f1_avg and loss to Tensorboard.
        misc_utils.add_summary(
            summary_writer, global_step, tag='f1_avg', value=f1_avg)
        misc_utils.add_summary(
            summary_writer,
            global_step,
            tag='reformulator_loss',
            value=reformulator_loss)

      # Train selector model.
      if FLAGS.enable_selector_training:
        (selector_questions, selector_answers,
         selector_scores) = query_environment(
             original_questions=questions_batch,
             rewrites=reformulations_beam,
             annotations=annotations_batch,
             environment_fn=eval_environment_fn,
             docid_2_answer=docid_2_answer,
             token_level_f1_scores=False)

        if FLAGS.debug:
          print('Training selector...')

        train_selector_loss, train_selector_accuracy = selector_model.train(
            selector_questions, selector_answers, selector_scores)

        # Regularly save a checkpoint.
        if global_step - last_save_step >= FLAGS.steps_per_save_selector:
          selector_model.save(str(global_step))
          last_save_step = global_step
          print('Selector saved at step: {}'.format(global_step))

        if FLAGS.debug:
          print('Train Accuracy: {}'.format(train_selector_accuracy))
          print('Train Loss    : {}'.format(train_selector_loss))

        # Write the accuracy and loss to Tensorboard.
        misc_utils.add_summary(
            summary_writer,
            global_step,
            tag='train_selector_accuracy',
            value=train_selector_accuracy)
        misc_utils.add_summary(
            summary_writer,
            global_step,
            tag='train_selector_loss',
            value=train_selector_loss)

      iteration_time = time.time() - start_time
      if FLAGS.debug:
        print('Iteration time: {}'.format(iteration_time))

      misc_utils.add_summary(
          summary_writer,
          global_step,
          tag='iteration_time',
          value=iteration_time)

      # Increment the global counter
      global_step += 1


if __name__ == '__main__':
  app.run(main)
