import multiprocessing
import os

import numpy as np

from ..tfops import modify_variable_names, tf


class TfMixin(object):
    def __init__(self, data_info, tf_sess_config=None):
        self.data_info = data_info
        self.sess = self._sess_config(tf_sess_config)

    @staticmethod
    def _sess_config(tf_sess_config=None):
        if not tf_sess_config:
            # Session config based on:
            # https://software.intel.com/content/www/us/en/develop/articles/tips-to-improve-performance-for-popular-deep-learning-frameworks-on-multi-core-cpus.html
            tf_sess_config = {
                "intra_op_parallelism_threads": 0,
                "inter_op_parallelism_threads": 0,
                "allow_soft_placement": True,
                "device_count": {"CPU": multiprocessing.cpu_count()},
            }
            # os.environ["OMP_NUM_THREADS"] = f"{self.cpu_num}"

        config = tf.ConfigProto(**tf_sess_config)
        return tf.Session(config=config)

    # rebuild_tf_graph
    def rebuild_model(self, path, model_name, full_assign=False):
        variable_path = os.path.join(path, f"{model_name}_tf_variables.npz")
        variables = np.load(variable_path)
        variables = dict(variables.items())

        (
            user_variables,
            item_variables,
            sparse_variables,
            dense_variables,
            manual_variables,
        ) = modify_variable_names(self, trainable=True)

        update_ops = []
        for v in tf.trainable_variables():
            if user_variables is not None and v.name in user_variables:
                # remove oov values
                old_var = variables[v.name][: self.data_info.n_users]
                user_op = tf.IndexedSlices(old_var, tf.range(len(old_var)))
                update_ops.append(v.scatter_update(user_op))

            if item_variables is not None and v.name in item_variables:
                old_var = variables[v.name][: self.data_info.n_items]
                item_op = tf.IndexedSlices(old_var, tf.range(len(old_var)))
                update_ops.append(v.scatter_update(item_op))

            if sparse_variables is not None and v.name in sparse_variables:
                old_var = variables[v.name]
                # remove oov values
                old_var = np.delete(old_var, self.data_info.old_sparse_oov, axis=0)
                indices = []
                for offset, size in zip(
                    self.data_info.sparse_offset, self.data_info.old_sparse_len
                ):
                    if size != -1:
                        indices.extend(range(offset, offset + size))
                sparse_op = tf.IndexedSlices(old_var, indices)
                update_ops.append(v.scatter_update(sparse_op))

            if dense_variables is not None and v.name in dense_variables:
                # dense values are same, no need to scatter_update
                old_var = variables[v.name]
                update_ops.append(v.assign(old_var))

        if full_assign:
            (
                optimizer_user_variables,
                optimizer_item_variables,
                optimizer_sparse_variables,
                optimizer_dense_variables,
                _,
            ) = modify_variable_names(self, trainable=False)

            other_variables = [
                v for v in tf.global_variables() if v.name not in manual_variables
            ]
            for v in other_variables:
                if (
                    optimizer_user_variables is not None
                    and v.name in optimizer_user_variables
                ):
                    old_var = variables[v.name][: self.data_info.n_users]
                    user_op = tf.IndexedSlices(old_var, tf.range(len(old_var)))
                    update_ops.append(v.scatter_update(user_op))

                elif (
                    optimizer_item_variables is not None
                    and v.name in optimizer_item_variables
                ):
                    old_var = variables[v.name][: self.data_info.n_items]
                    item_op = tf.IndexedSlices(old_var, tf.range(len(old_var)))
                    update_ops.append(v.scatter_update(item_op))

                elif (
                    optimizer_sparse_variables is not None
                    and v.name in optimizer_sparse_variables
                ):
                    old_var = variables[v.name]
                    # remove oov values
                    old_var = np.delete(old_var, self.data_info.old_sparse_oov, axis=0)
                    indices = []
                    for offset, size in zip(
                        self.data_info.sparse_offset, self.data_info.old_sparse_len
                    ):
                        if size != -1:
                            indices.extend(range(offset, offset + size))
                    sparse_op = tf.IndexedSlices(old_var, indices)
                    update_ops.append(v.scatter_update(sparse_op))

                elif (
                    optimizer_dense_variables is not None
                    and v.name in optimizer_dense_variables
                ):
                    old_var = variables[v.name]
                    update_ops.append(v.assign(old_var))

                elif v.name in variables:
                    old_var = variables[v.name]
                    if list(old_var.shape) != v.get_shape().as_list():
                        print(
                            f'old and new shape of variable "{v.name}" '
                            f"doesn't match, will be skipped."
                        )
                        continue
                    update_ops.append(v.assign(old_var))

        self.sess.run(update_ops)
