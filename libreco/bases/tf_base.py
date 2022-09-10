import os

from .base import Base
from .tf_mixin import TfMixin
from ..prediction import predict_tf_feat
from ..recommendation import recommend_tf_feat
from ..tfops import modify_variable_names, tf
from ..utils.save_load import (
    load_tf_model,
    load_tf_variables,
    save_params,
    save_tf_model,
    save_tf_variables,
)


class TfBase(Base, TfMixin):
    def __init__(self, task, data_info, lower_upper_bound, tf_sess_config):
        Base.__init__(self, task, data_info, lower_upper_bound)
        TfMixin.__init__(self, data_info, tf_sess_config)
        self.trainer = None

    def fit(
        self,
        train_data,
        verbose=1,
        shuffle=True,
        eval_data=None,
        metrics=None,
        **kwargs,
    ):
        self.show_start_time()
        self.trainer.run(train_data, verbose, shuffle, eval_data, metrics)
        self.assign_tf_variables_oov()

    def predict(self, user, item, feats=None, cold_start="average", inner_id=False):
        return predict_tf_feat(self, user, item, feats, cold_start, inner_id)

    def recommend_user(
        self,
        user,
        n_rec,
        user_feats=None,
        item_data=None,
        cold_start="average",
        inner_id=False,
    ):
        return recommend_tf_feat(
            self, user, n_rec, user_feats, item_data, cold_start, inner_id
        )

    def assign_tf_variables_oov(self):
        (
            user_variables,
            item_variables,
            sparse_variables,
            dense_variables,
            _,
        ) = modify_variable_names(self, trainable=True)

        update_ops = []
        for v in tf.trainable_variables():
            if user_variables is not None and v.name in user_variables:
                # size = v.get_shape().as_list()[1]
                mean_op = tf.IndexedSlices(
                    tf.reduce_mean(
                        tf.gather(v, tf.range(self.n_users)), axis=0, keepdims=True
                    ),
                    [self.n_users],
                )
                update_ops.append(v.scatter_update(mean_op))

            if item_variables is not None and v.name in item_variables:
                mean_op = tf.IndexedSlices(
                    tf.reduce_mean(
                        tf.gather(v, tf.range(self.n_items)), axis=0, keepdims=True
                    ),
                    [self.n_items],
                )
                update_ops.append(v.scatter_update(mean_op))

            if sparse_variables is not None and v.name in sparse_variables:
                sparse_oovs = self.data_info.sparse_oov
                start = 0
                for oov in sparse_oovs:
                    # multi_sparse case
                    if start >= oov:
                        continue
                    mean_tensor = tf.reduce_mean(
                        tf.gather(v, tf.range(start, oov)), axis=0, keepdims=True
                    )
                    update_ops.append(v.scatter_nd_update([[oov]], mean_tensor))
                    start = oov + 1

        self.sess.run(update_ops)

    def save(self, path, model_name, manual=True, inference_only=False):
        if not os.path.isdir(path):
            print(f"file folder {path} doesn't exists, creating a new one...")
            os.makedirs(path)
        save_params(self, path, model_name)
        if manual:
            save_tf_variables(self.sess, path, model_name, inference_only)
        else:
            save_tf_model(self.sess, path, model_name)

    @classmethod
    def load(cls, path, model_name, data_info, manual=True):
        if manual:
            return load_tf_variables(cls, path, model_name, data_info)
        else:
            return load_tf_model(cls, path, model_name, data_info)
