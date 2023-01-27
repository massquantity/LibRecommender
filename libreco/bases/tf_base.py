import os

import numpy as np

from ..prediction import predict_tf_feat
from ..recommendation import cold_start_rec, construct_rec, recommend_tf_feat
from ..tfops import modify_variable_names, sess_config, tf
from ..utils.save_load import (
    load_tf_model,
    load_tf_variables,
    save_default_recs,
    save_params,
    save_tf_model,
    save_tf_variables,
)
from ..utils.validate import check_unknown_user
from .base import Base


class TfBase(Base):
    def __init__(self, task, data_info, lower_upper_bound, tf_sess_config):
        Base.__init__(self, task, data_info, lower_upper_bound)
        self.sess = sess_config(tf_sess_config)
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
        k = kwargs.get("k", 10)
        eval_batch_size = kwargs.get("eval_batch_size", 2**15)
        eval_user_num = kwargs.get("eval_user_num", None)
        assert k <= self.n_items, f"eval `k` {k} exceeds num of items {self.n_items}"
        self.show_start_time()
        self.trainer.run(
            train_data,
            verbose,
            shuffle,
            eval_data,
            metrics,
            k,
            eval_batch_size,
            eval_user_num,
        )
        self.assign_tf_variables_oov()
        self.default_recs = recommend_tf_feat(
            model=self,
            user_ids=[self.n_users],
            n_rec=min(2000, self.n_items),
            user_feats=None,
            item_data=None,
            filter_consumed=False,
            random_rec=False,
        ).flatten()

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
        filter_consumed=True,
        random_rec=False,
    ):
        if (
            (user_feats is not None or item_data is not None)
            and not np.isscalar(user)
            and len(user) > 1
        ):
            raise ValueError(
                f"Batch recommend doesn't support assigning arbitrary features: {user}"
            )

        result_recs = dict()
        user_ids, unknown_users = check_unknown_user(self.data_info, user, inner_id)
        if unknown_users:
            cold_recs = cold_start_rec(
                self.data_info,
                self.default_recs,
                cold_start,
                unknown_users,
                n_rec,
                inner_id,
            )
            result_recs.update(cold_recs)
        if user_ids:
            computed_recs = recommend_tf_feat(
                self,
                user_ids,
                n_rec,
                user_feats,
                item_data,
                filter_consumed,
                random_rec,
            )
            user_recs = construct_rec(self.data_info, user_ids, computed_recs, inner_id)
            result_recs.update(user_recs)
        return result_recs

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
        save_default_recs(self, path, model_name)
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
