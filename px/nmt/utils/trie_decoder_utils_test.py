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

"""Tests for trie_decoder_utils.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import shutil
import tempfile


import tensorflow as tf

from px.nmt.utils import trie_decoder_utils
from px.proto import aqa_pb2


class TrieDecoderUtilsTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.tmpdir = tempfile.mkdtemp()

    cls.vocab_path = os.path.join(cls.tmpdir, 'test.vocab')
    cls.save_path = os.path.join(cls.tmpdir, 'trie.pkl')

    cls.test_vocab = """
    <unk>
    <s>
    </s>
    a
    b
    c
    """.strip().split()

    cls.test_trie_entries = [
        ('a b c b a', 'x'),
        ('a b c b a c', 'y'),
        ('a c c b', 'z'),
    ]

    with open(cls.vocab_path, 'w') as f:
      f.write('\n'.join(cls.test_vocab))


  @classmethod
  def tearDownClass(cls):
    shutil.rmtree(cls.tmpdir)


  def testSaveLoadTrie(self):
    trie = trie_decoder_utils.DecoderTrie(TrieDecoderUtilsTest.vocab_path)
    for question, answer in TrieDecoderUtilsTest.test_trie_entries:
      trie.insert_question(question, answer)
    NEW_KEY = 'newkey'
    self.assertNotIn(NEW_KEY, trie)
    self.assertFalse(os.path.exists(TrieDecoderUtilsTest.save_path))
    trie.save_to_file(TrieDecoderUtilsTest.save_path)
    self.assertTrue(os.path.exists(TrieDecoderUtilsTest.save_path))
    trie[NEW_KEY] = 'foo'
    self.assertIn(NEW_KEY, trie)
    loaded_trie = trie_decoder_utils.DecoderTrie.load_from_file(
        TrieDecoderUtilsTest.save_path)
    self.assertNotIn(NEW_KEY, loaded_trie)

  def testCreateEmptyTrie(self):
    eos_token = '</s>'
    trie = trie_decoder_utils.DecoderTrie(TrieDecoderUtilsTest.vocab_path,
                                          eos_token)
    self.assertLen(trie, 0)
    self.assertEqual(trie.eos_idx, '2')

  def testEmptyPathDoesNotExist(self):
    self.assertFalse(tf.gfile.Exists(''))


if __name__ == '__main__':
  tf.test.main()
