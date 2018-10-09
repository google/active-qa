# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of a selector model using convolutional encoders.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


from absl import flags
from absl import logging
from keras.layers import Convolution1D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import GlobalMaxPooling1D
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing import text
import numpy as np

from tensorflow import gfile

flags.DEFINE_string('glove_path', '', 'Path to pretrained Glove embeddings.')
flags.DEFINE_integer('embedding_dim', 100, 'Embedding dimension.')
flags.DEFINE_integer('max_sequence_length', 10, 'Maximum sequence length.')
flags.DEFINE_string('save_path', '',
                    'Directory where models will be saved to/loaded from.')

FLAGS = flags.FLAGS


class Selector(object):
  """A selector model that selects the best question/answer out of a set."""

  def __init__(self):
    """Constructor for the selector."""
    logging.info('Initializing tokenizer..')

    words, embedding_matrix = self._build_embedding_matrix()
    self.tokenizer = text.Tokenizer(num_words=len(words), lower=False)
    # Tokenizer treats each item in a nested list as a token.
    self.tokenizer.fit_on_texts([[word] for word in words])

    # Preppend a array of zeros to the embeddings matrix that will be used by
    # out-of-vocabulary words.
    embedding_matrix = np.concatenate(
        [np.zeros((1, embedding_matrix.shape[1])), embedding_matrix])

    assert len(words) == len(self.tokenizer.word_index), (
        'embeddings_matrix and tokenizer.word_index do not have the same size:'
        ' {} and {}, respectively'.format(
            len(words), len(self.tokenizer.word_index)))
    assert all([
        self.tokenizer.word_index[word] == i + 1 for i, word in enumerate(words)
    ]), ('embeddings_matrix and tokenizer.word_index are not aligned.')

    self.model = self._build_model(embedding_matrix)

  def load(self, name):
    checkpoint_path_json, checkpoint_path_h5 = self._get_checkpoint_paths(name)
    with gfile.Open(checkpoint_path_json, 'r') as json_file:
      loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    gfile.Copy(checkpoint_path_h5, '/tmp/tmp_model_weights.h5')
    model.load_weights('/tmp/tmp_model_weights.h5')
    logging.info('Loaded model from disk.')
    return model

  def save(self, name):
    checkpoint_path_json, checkpoint_path_h5 = self._get_checkpoint_paths(name)
    model_json = self.model.to_json()
    with gfile.Open(checkpoint_path_json, 'w') as json_file:
      json_file.write(model_json)
    self.model.save_weights('/tmp/tmp_model_weights.h5')
    gfile.Copy('/tmp/tmp_model_weights.h5', checkpoint_path_h5)

  def _get_checkpoint_paths(self, name):
    checkpoint_path_json = os.path.join(FLAGS.save_path,
                                        'model_' + name + '.json')
    checkpoint_path_h5 = os.path.join(FLAGS.save_path, 'model_' + name + '.h5')
    return checkpoint_path_json, checkpoint_path_h5

  def _build_embedding_matrix(self):
    """Builds the embedding matrix for the model.

    Returns:
      words: a list of strings representing the words in the vocabulary.
      embeddings: a float32 array of shape [vocab_size, embeddings_dim].
    """
    logging.info('Loading Glove embeddings.')
    words = []
    embeddings = []
    with gfile.GFile(FLAGS.glove_path) as f:
      for line in f:
        values = line.split()
        words.append(values[0])
        embeddings.append(np.asarray(values[1:], dtype='float32'))

    logging.info('Found %s word vectors.', len(embeddings))
    return words, np.array(embeddings)

  def _build_model(self, embedding_matrix):
    """Builds the model.

    Args:
      embedding_matrix: A float32 array of shape [vocab_size, embedding_dim].

    Returns:
      The model.
    """
    max_feature_length = FLAGS.max_sequence_length

    model_inputs = []
    encoder_outputs = []
    for _ in range(3):
      model_input = Input(shape=(max_feature_length,))
      model_inputs.append(model_input)
      embed = Embedding(
          output_dim=100,
          input_dim=len(embedding_matrix),
          input_length=max_feature_length,
          weights=[embedding_matrix],
          trainable=False)(
              model_input)
      conv = Convolution1D(
          filters=100,
          kernel_size=3,
          padding='valid',
          activation='relu',
          strides=1)(
              embed)
      conv = Dropout(0.4)(conv)
      conv = GlobalMaxPooling1D()(conv)
      encoder_outputs.append(conv)

    merge = Concatenate()(encoder_outputs)
    model_output = Dense(1, activation='sigmoid')(merge)

    model = Model(model_inputs, model_output)
    model.compile(
        loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    logging.info('Model successfully built. Summary: %s', model.summary())
    return model

  def encode_labels(self, labels):
    return np.asarray(labels).astype(np.float)

  def encode_texts(self, texts):
    sequences = self.tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=FLAGS.max_sequence_length)

  def encode_data(self, questions, original_questions, answers, labels):
    return (self.encode_texts(questions), self.encode_texts(original_questions),
            self.encode_texts(answers), self.encode_labels(labels))

  def encode_train(self, question_lists, answer_lists, score_lists):
    """Encodes the input for training purposes.

    The data points consist of:
      - question (original or rewrite)
      - original question
      - answer
      - label
    where the label is the difference between the F1 score of the question and
    the average F1 score of all the questions with the same source.

    Args:
      question_lists: A list of lists of questions. The first question is the
                      original question and the others are generated by a
                      Reformulator model.
      answer_lists: A list of lists of answers to the questions given by the
                    BiDAF model.
      score_lists: A list of lists of F1 scores for the answers given by the
                   BiDAF model.

    Returns:
      * A numpy array with dimensions [len(questions), max_sequence_length]
        containing the tokenized questions.
      * A numpy array with dimensions
        [len(original_questions), max_sequence_length] containing the tokenized
        original questions.
      * A numpy array with dimensions [len(answers), max_sequence_length]
        containing the tokenized answers.
      * A numpy array with dimensions [len(answers)] containing the differences
        of the F1 score from the average of all rewrites with the same source.
    """
    rewritten_questions = []
    original_questions = []
    ans = []
    labels = []

    for questions, answers, scores in zip(question_lists, answer_lists,
                                          score_lists):
      mean_score = np.mean(scores)
      original_question = questions[0]

      for question, answer, score in zip(questions, answers, scores):
        if score == mean_score:
          # Ignore all examples where the F1 score is equal to the mean. This
          # helps filter out examples that we cannot learn from; e.g. if all
          # rewrites in a set give the same F1 score, all of the set is ignored.
          continue
        rewritten_questions.append(question)
        original_questions.append(original_question)
        ans.append(answer)

        labels.append(score - mean_score)
    return self.encode_data(rewritten_questions, original_questions, ans,
                            labels)

  def train(self, questions, answers, scores):
    """Train the model with the given data.

    Args:
      questions: A list of lists of questions. The first question is the
                 original question and the others are generated by a
                 Reformulator model.
      answers: A list of lists of answers to the questions given by the BiDAF
               model.
      scores: A list of lists of F1 scores for the answers given by the BiDAF
              model.

    Returns:
      A tuple containing the training loss and accuracy of the batch.
    """
    (question_array, original_question_array, answer_array,
     train_labels) = self.encode_train(questions, answers, scores)
    train_labels_binary = (np.sign(train_labels) + 1) / 2
    train_labels_array_binary = np.array(train_labels_binary)
    return self.model.train_on_batch(
        x=[question_array, original_question_array, answer_array],
        y=train_labels_array_binary)

  def eval(self, question_lists, answer_lists, score_lists):
    """Run an eval with the given data.

    Args:
      question_lists: A list of lists of questions. The first question is the
                      original question and the others are generated by a
                      Reformulator model.
      answer_lists: A list of lists of answers to the questions given by the
                    BiDAF model.
      score_lists: A list of lists of F1 scores for the answers given by the
                   BiDAF model.

    Returns:
      Average F1 score achieved with the model.
    """
    f1s = []
    for questions, answers, scores in zip(question_lists, answer_lists,
                                          score_lists):
      original_questions = [questions[0]] * len(questions)
      xs1, xs2, xs3, ys = self.encode_data(questions, original_questions,
                                           answers, scores)
      prediction = np.argmax(self.model.predict([xs1, xs2, xs3]))
      f1s.append(ys[prediction])
    return np.mean(f1s)
