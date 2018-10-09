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

"""Utility functions for logging, with special regard to Unicode handling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf


def safe_string(s):
  """Safely converts unicode and plain strings to byte strings."""
  if isinstance(s, unicode):
    try:
      s = s.encode('utf-8')
    except UnicodeDecodeError:
      s = repr(s)
  return s


def safe_print(s):
  """Safely prints byte strings and unicode strings."""
  print(safe_string(s))


def add_histogram(summary_writer, step, tag, values, bins=10):
  """Logs the histogram of a list/vector of values.

  Args:
    summary_writer: a tf.summary.FileWriter instance.
    step: the step in which this value corresponds to.
    tag: string. The name of the summary value.
    values: a list or a numpy array.
    bins: If bins is an int, it defines the number of equal-width bins in the
        given range (10, by default). If bins is a sequence, it defines the bin
        edges, including the rightmost edge, allowing for non-uniform bin
        widths.
  """

  values = np.array(values)
  if isinstance(bins, int):
    histogram_min = 0
    histogram_max = bins - 1
  else:
    histogram_min = np.min(bins)
    histogram_max = np.max(bins)

  counts, bin_edges = np.histogram(
      values, bins=bins, range=(histogram_min, histogram_max))

  hist = tf.HistogramProto()

  hist.min = histogram_min
  hist.max = histogram_max
  hist.num = values.size
  hist.sum = float(np.sum(values))
  hist.sum_squares = float(np.sum(values**2))

  # Requires equal number as bins, where the first goes from -DBL_MAX to
  # bin_edges[1]. See:
  # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
  # Thus, we drop the start of the first bin
  bin_edges = bin_edges[1:]

  # Add bin edges and counts
  hist.bucket_limit.extend(bin_edges)
  hist.bucket.extend(counts)

  # Create and write Summary
  summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
  summary_writer.add_summary(summary, step)
  summary_writer.flush()
