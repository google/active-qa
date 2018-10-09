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

"""Wraps the DocQA model for use as an environment.

The environment uses a DocQA model to produce an answer from a specified
document.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import pickle

import tensorflow as tf

from tensorflow import gfile

from docqa.data_processing.document_splitter import FirstN
from docqa.data_processing.document_splitter import MergeParagraphs
from docqa.data_processing.document_splitter import ShallowOpenWebRanker
from docqa.data_processing.document_splitter import TopTfIdf
from docqa.data_processing.preprocessed_corpus import preprocess_par
from docqa.data_processing.qa_training_data import ParagraphAndQuestionDataset
from docqa.data_processing.span_data import TokenSpans
from docqa.data_processing.text_utils import NltkAndPunctTokenizer
from docqa.data_processing.text_utils import NltkPlusStopWords

from docqa.dataset import FixedOrderBatcher
from docqa.eval.triviaqa_full_document_eval import RecordParagraphSpanPrediction
from docqa.evaluator import AysncEvaluatorRunner
from docqa.evaluator import EvaluatorRunner
from docqa.model_dir import ModelDir
from docqa.triviaqa.build_span_corpus import TriviaQaOpenDataset
from docqa.triviaqa.build_span_corpus import TriviaQaWebDataset
from docqa.triviaqa.build_span_corpus import TriviaQaWikiDataset
from docqa.triviaqa.training_data import DocumentParagraphQuestion
from docqa.triviaqa.training_data import ExtractMultiParagraphs
from docqa.triviaqa.training_data import ExtractMultiParagraphsPerQuestion
from docqa.utils import ResourceLoader


class DocqaEnvironment(object):
  """Environment containing the DocQA model.

  This environment loads a DocQA model and preprocessed data for the chosen
  datasets. The environment is queried with a pointer to an existing datapoint,
  which contains preprocessed documents, and a question. DocQA is run
  using the given question against the documents and the top answer with its
  score is returned.
  """

  def __init__(self,
               precomputed_data_path,
               corpus_dir,
               model_dir,
               nltk_dir,
               async=0,
               batch_size=32,
               corpus_name="wiki",
               debug_mode=False,
               filter_name=None,
               load_test=False,
               max_answer_len=8,
               max_tokens=400,
               n_paragraphs=5,
               n_processes=12,
               step="latest"):
    """Constructor loads the DocQA configuration, model and data.

    Args:
      precomputed_data_path: Path to the precomputed data stored in a pickle
          file.
      corpus_dir: Path to corpus directory.
      model_dir: Directory containing parameters of a pre-trained DocQA model.
      nltk_dir: Folder containing the nltk package.
      async: If greater than 0, run <async> evaluations in parallel.
      batch_size: Maximum batch size.
      corpus_name: Name of the corpus: "wiki" or "web".
      debug_mode: If true, logs additional debug information.
      filter_name: Type of the filter to select documents. Valid values are:
          "linear", "tfidf", or "truncate".
      load_test: If True, loads the test set as well.
      max_answer_len: Maximum number of tokens an answer will have. Truncate if
          it is longer.
      max_tokens: Maximum number of tokens per paragraph.
      n_paragraphs: Maximum number of paragraphs to be retrieved.
      n_processes: Number of parallel processes to use whe loading the data.
      step: Which step from the checkpoint the model will be loaded from.
          When step="latest", the lastest checkpoint in model_dir will be used.
    """

    self.async = async
    self.debug_mode = debug_mode
    self.max_tokens = max_tokens

    self.evaluators = [RecordParagraphSpanPrediction(max_answer_len, True)]
    self.tokenizer = NltkAndPunctTokenizer(nltk_dir=nltk_dir)

    datasets = ["train", "dev"]
    if load_test:
      datasets.append("test")

    print("Loading model...")
    model_dir = ModelDir(model_dir)
    self.model = model_dir.get_model()
    print("Loading data...")
    self.corpus_name = corpus_name
    self.load_data(
        precomputed_data_path=precomputed_data_path,
        corpus_name=corpus_name,
        corpus_dir=corpus_dir,
        nltk_dir=nltk_dir,
        datasets=datasets,
        filter_name=filter_name,
        n_paragraphs=n_paragraphs,
        n_processes=n_processes)

    print("Setting up model")
    # Tell the model the batch size (can be None) and vocab to expect. This will
    # load the needed word vectors and fix the batch size to use when building
    # the graph / encoding the input.
    # This step is here to compute the vocabulary.
    data_flattened = []
    for val in self.data.values():
      data_flattened.extend(val)
    temp_data = ParagraphAndQuestionDataset(data_flattened,
                                            FixedOrderBatcher(batch_size, True))

    self.model.set_inputs([temp_data], ResourceLoader())

    if self.async > 0:
      self.evaluator_runner = AysncEvaluatorRunner(self.evaluators, self.model,
                                                   self.async)
      inputs = self.evaluator_runner.dequeue_op
    else:
      self.evaluator_runner = EvaluatorRunner(self.evaluators, self.model)
      inputs = self.model.get_placeholders()

    input_dict = {p: x for p, x in zip(self.model.get_placeholders(), inputs)}

    self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    with self.sess.as_default():
      pred = self.model.get_predictions_for(input_dict)
    self.evaluator_runner.set_input(pred)

    if step is not None:
      if step == "latest":
        checkpoint = model_dir.get_latest_checkpoint()
      else:
        checkpoint = model_dir.get_checkpoint(int(step))
    else:
      checkpoint = model_dir.get_best_weights()
      if checkpoint is not None:
        print("Using best weights")
      else:
        print("Using latest checkpoint")
        checkpoint = model_dir.get_latest_checkpoint()

    saver = tf.train.Saver()
    saver.restore(self.sess, checkpoint)
    tf.get_default_graph().finalize()

  def GetAnswers(self, questions, document_ids):
    """Computes an answer for a given question and document id.

    Runs a DocQA model on a specified datapoint, but using the input
    question in place of the original.

    Args:
      questions: List of strings used to replace the original question.
      document_ids: A list of strings representing the identifiers for the
          context documents.

    Returns:
      A list

    Raises:
      ValueError: If the number of questions and document_ids differ.
    """
    if len(questions) != len(document_ids):
      raise ValueError("Number of questions and document_ids must be equal.")

    # Prepare questions:
    data_preprocessed = []
    question_ids = []
    for question, document_id in zip(questions, document_ids):
      question_tokenized = self.tokenizer.tokenize_paragraph_flat(question)
      original_paragraph_questions = self.data[document_id]
      for rank, original_paragraph_question in enumerate(
          original_paragraph_questions):
        if rank == 0:
          question_ids.append(original_paragraph_question.question_id)
        new_paragraph_question = DocumentParagraphQuestion(
            q_id=original_paragraph_question.question_id,
            doc_id=original_paragraph_question.doc_id,
            para_range=original_paragraph_question.para_range,
            question=question_tokenized,
            context=original_paragraph_question.context,
            answer=original_paragraph_question.answer,
            rank=rank)
        data_preprocessed.append(new_paragraph_question)

    data_preprocessed = ParagraphAndQuestionDataset(
        data_preprocessed,
        FixedOrderBatcher(batch_size=len(questions), truncate_batches=True))

    evaluation = self.evaluator_runner.run_evaluators(
        sess=self.sess,
        dataset=data_preprocessed,
        name=self.corpus_name,
        n_sample=None,
        feed_dict={})

    # create a pandas dataframe that will have the following columns:
    # question_id, doc_id, n_answers, para_end, para_start, predicted_start,
    # predicted_end, predicted_score, rank, text_answer, text_em, text_f1
    dataframe = pd.DataFrame(evaluation.per_sample)

    answers = self.best_answers(dataframe)
    # align questions and answers
    output = [answers[question_id] for question_id in question_ids]

    return output

  def best_answers(self, dataframe):
    """Return the best answer based on the predicted score.
    """

    answers = {}
    for question_id, text_answer, predicted_score, text_f1 in dataframe[[
        "question_id", "text_answer", "predicted_score", "text_f1"
    ]].itertuples(index=False):
      if question_id not in answers:
        answers[question_id] = (text_answer, predicted_score, text_f1)
      else:

        if predicted_score > answers[question_id][1]:
          answers[question_id] = (text_answer, predicted_score, text_f1)
    return answers

  def load_data(self, precomputed_data_path, corpus_name, corpus_dir, nltk_dir,
                datasets, filter_name, n_paragraphs, n_processes):
    """Load corpus and question-answer data onto memory.
    """

    if corpus_name.startswith("web"):
      self.dataset = TriviaQaWebDataset(corpus_dir)
    elif corpus_name.startswith("wiki"):
      self.dataset = TriviaQaWikiDataset(corpus_dir)
    else:
      self.dataset = TriviaQaOpenDataset(corpus_dir)

    questions = []
    if "train" in datasets:
      questions.extend(self.dataset.get_train())
    if "dev" in datasets:
      questions.extend(self.dataset.get_dev())
    if "test" in datasets:
      questions.extend(self.dataset.get_test())

    # wiki and web are both multi-document
    per_document = corpus_name.startswith("web")

    if per_document:
      self.group_by = ["question_id", "doc_id"]
    else:
      self.group_by = ["question_id"]

    if gfile.Exists(precomputed_data_path):
      print("Loading precomputed data from {}".format(precomputed_data_path))
      with gfile.GFile(precomputed_data_path, "rb") as f:
        self.data = pickle.load(f)
    else:
      print("Building question/paragraph pairs...")
      corpus = self.dataset.evidence
      splitter = MergeParagraphs(self.max_tokens)

      if filter_name is None:
        # Pick default depending on the kind of data we are using
        if per_document:
          filter_name = "tfidf"
        else:
          filter_name = "linear"

      if filter_name == "tfidf":
        para_filter = TopTfIdf(
            NltkPlusStopWords(punctuation=True, nltk_dir=nltk_dir),
            n_to_select=n_paragraphs)
      elif filter_name == "truncate":
        para_filter = FirstN(n_paragraphs)
      elif filter_name == "linear":
        para_filter = ShallowOpenWebRanker(
            n_to_select=n_paragraphs, nltk_dir=nltk_dir)
      else:
        raise ValueError()

      # Loads the relevant questions/documents, selects the right paragraphs,
      # and runs the model's preprocessor.
      if per_document:
        prep = ExtractMultiParagraphs(
            splitter,
            para_filter,
            self.model.preprocessor,
            require_an_answer=False)
      else:
        prep = ExtractMultiParagraphsPerQuestion(
            splitter,
            para_filter,
            self.model.preprocessor,
            require_an_answer=False)

      prepped_data = preprocess_par(questions, corpus, prep, n_processes, 1000)

      self.data = {}
      for q in prepped_data.data:
        self.data[q.question_id] = []
        for rank, p in enumerate(q.paragraphs):
          if q.answer_text is None:
            ans = None
          else:
            ans = TokenSpans(q.answer_text, p.answer_spans)
          self.data[q.question_id].append(
              DocumentParagraphQuestion(
                  q_id=q.question_id,
                  doc_id=p.doc_id,
                  para_range=(p.start, p.end),
                  question=q.question,
                  context=p.text,
                  answer=ans,
                  rank=rank))

      print("Saving precomputed data to {}".format(precomputed_data_path))
      with gfile.GFile(precomputed_data_path, "wb") as f:
        pickle.dump(self.data, f, -1)
      print("Done.")
