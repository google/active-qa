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

import re

from tensorflow.python.eager import context
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging
from tensorflow.python.training.checkpoint_utils import load_checkpoint
from tensorflow.python.training.saver import BaseSaverBuilder
from tensorflow.python.training.saver import Saver

__all__ = ['OptimisticRestoreSaver']


class OptimisticRestoreSaver(Saver):
  """A Saver that restores variables in a best-effort manner.

  Only restores variables in `var_list` that are present in the checkpoint on
  restore.
  However, on save, all variables in `var_list` are written to the checkpoint.
  Pass `init_uninitialized_variables=True` to the constructor in order to
  initialize all uninitialized variables in `var_list` automatically after
  the restore operation.
  """

  def __init__(self,
               var_list=None,
               init_uninitialized_variables=False,
               **kwargs):
    kwargs['restore_sequentially'] = False
    kwargs['builder'] = BaseSaverBuilder()
    super(OptimisticRestoreSaver, self).__init__(var_list=var_list, **kwargs)
    self.init_uninitialized_variables = init_uninitialized_variables
    if self.init_uninitialized_variables:
      self.uninit_vars_op = variables.report_uninitialized_variables(
          var_list=self._var_list)
      self.init_ops = dict((v.name, variables.variables_initializer([v]))
                           for v in self._var_list)

  def restore(self, sess, save_path, var_filter=lambda v: True):
    """Restores variables in a best-effort manner.

    Restores only variables that are contained in `save_path` and match in shape
    and dtype and return `True` when passed to `var_filter`.

    Args:
      sess: Tensorflow session.
      save_path: Path to checkpoint.
      var_filter: Callable that receives a `tf.Variable` and returns `False`
        when the variable should not be restored, and `True` otherwise. By
        default, it returns `True` for all variables.

    Raises:
      ValueError: When `save_path` is `None`.
    """
    if self._is_empty:
      return
    if save_path is None:
      raise ValueError("Can't load save_path when it is None.")
    tf_logging.info('Restoring parameters from %s', save_path)

    reader = load_checkpoint(save_path)
    shape_map = reader.get_variable_to_shape_map()
    dtype_map = reader.get_variable_to_dtype_map()

    restore_op_name = self.saver_def.restore_op_name
    restore_op_grouped = sess.graph.get_operation_by_name(restore_op_name)

    def get_restore_ops(r_op):
      return sum((get_restore_ops(i) for i in r_op.control_inputs), [r_op]
                 if r_op.type == 'Assign' else [])

    all_restore_ops = get_restore_ops(restore_op_grouped)
    filtered_restore_ops = []

    for r_op in all_restore_ops:
      v = r_op.inputs[0]
      tensor_name = v.op.name
      part_match = re.search(r'/part_\d+$', tensor_name)
      if part_match:
        tf_logging.info('variable %s is sharded', tensor_name)
        tensor_name = tensor_name[:part_match.span()[0]]
      tensor_shape = v.get_shape().as_list()
      tensor_dtype = v.dtype.base_dtype
      if tensor_name not in shape_map or tensor_name not in dtype_map:
        tf_logging.warn('variable %s not in checkpoint', tensor_name)
      elif shape_map[tensor_name] != tensor_shape and not part_match:
        tf_logging.warn(
            'variable %s in checkpoint, but checkpoint shape %r does not match '
            'graph shape %r', tensor_name, shape_map[tensor_name], tensor_shape)
      elif dtype_map[tensor_name] != tensor_dtype:
        tf_logging.warn(
            'variable %s in checkpoint, but checkpoint dtype %r does not match '
            'graph dtype %r', tensor_name, dtype_map[tensor_name], tensor_dtype)
      elif not var_filter(v):
        tf_logging.info('variable %s (dtype %r) rejected by var_filter',
                        tensor_name, tensor_dtype)
      else:
        filtered_restore_ops.append(r_op)
        tf_logging.info('adding variable %s to be restored', tensor_name)

    if context.in_eager_mode():
      raise NotImplementedError('eager selective restoring not supported yet')

    sess.run(filtered_restore_ops,
             {self.saver_def.filename_tensor_name: save_path})

    if self.init_uninitialized_variables:
      tf_logging.info('Initializing uninitialized variables.')
      uninitialized_vars = sess.run(self.uninit_vars_op)
      init_ops = []
      for v in uninitialized_vars:
        tf_logging.info('Initializing %s', v)
        init_ops.append(self.init_ops[v + ':0'])
      sess.run(init_ops)
