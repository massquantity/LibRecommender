"""

Reference: Heng-Tze Cheng et al. "Wide & Deep Learning for Recommender Systems"  (https://arxiv.org/pdf/1606.07792.pdf)

author: massquantity

"""
from itertools import islice
import os
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from tensorflow.keras.initializers import (
    truncated_normal as tf_truncated_normal
)
from .base import Base, TfMixin
from ..evaluation.evaluate import EvalMixin
from ..utils.tf_ops import (
    reg_config,
    dropout_config,
    dense_nn,
    lr_decay_config,
    var_list_by_name
)
from ..data.data_generator import DataGenFeat
from ..utils.sampling import NegativeSampling
from ..utils.misc import count_params
from ..feature import (
    get_predict_indices_and_values,
    get_recommend_indices_and_values,
    features_from_dict,
    add_item_features
)
tf.disable_v2_behavior()


class WideDeep(Base, TfMixin, EvalMixin):
    """
    According to the original paper, the Wide part used
    FTRL with L1 regularization as the optimizer, so we'll also adopt it here.
    Note this may not be suitable for your specific task.
    """
    user_variables = ["wide_user_feat", "deep_user_feat"]
    item_variables = ["wide_item_feat", "deep_item_feat"]
    sparse_variables = ["wide_sparse_feat", "deep_sparse_feat"]
    dense_variables = ["wide_dense_feat", "deep_dense_feat"]

    def __init__(
            self,
            task,
            data_info=None,
            embed_size=16,
            n_epochs=20,
            lr=None,
            lr_decay=False,
            reg=None,
            batch_size=256,
            num_neg=1,
            use_bn=True,
            dropout_rate=None,
            hidden_units="128,64,32",
            batch_sampling=False,
            seed=42,
            lower_upper_bound=None,
            tf_sess_config=None
    ):
        Base.__init__(self, task, data_info, lower_upper_bound)
        TfMixin.__init__(self, tf_sess_config)
        EvalMixin.__init__(self, task, data_info)

        self.task = task
        self.data_info = data_info
        self.embed_size = embed_size
        self.n_epochs = n_epochs
        self.lr = lr if lr is not None else {"wide": 0.01, "deep": 1e-4}
        self.lr_decay = lr_decay
        self.reg = reg_config(reg)
        self.batch_size = batch_size
        self.num_neg = num_neg
        self.use_bn = use_bn
        self.dropout_rate = dropout_config(dropout_rate)
        self.batch_sampling = batch_sampling
        self.hidden_units = list(map(int, hidden_units.split(",")))
        self.n_users = data_info.n_users
        self.n_items = data_info.n_items
        self.seed = seed
        self.user_consumed = data_info.user_consumed
        self.sparse = self._decide_sparse_indices(data_info)
        self.dense = self._decide_dense_values(data_info)
        if self.sparse:
            self.sparse_feature_size = self._sparse_feat_size(data_info)
            self.sparse_field_size = self._sparse_field_size(data_info)
        if self.dense:
            self.dense_field_size = self._dense_field_size(data_info)
        self.all_args = locals()

    def _build_model(self):
        self.graph_built = True
        tf.set_random_seed(self.seed)
        self.labels = tf.placeholder(tf.float32, shape=[None])
        self.is_training = tf.placeholder_with_default(False, shape=[])
        self.wide_embed, self.deep_embed = [], []

        self._build_user_item()
        if self.sparse:
            self._build_sparse()
        if self.dense:
            self._build_dense()

        wide_embed = tf.concat(self.wide_embed, axis=1)
        wide_term = tf.layers.dense(wide_embed,
                                    units=1,
                                    activation=None,
                                    name="wide_term")

        deep_embed = tf.concat(self.deep_embed, axis=1)
        deep_layer = dense_nn(deep_embed,
                              self.hidden_units,
                              use_bn=self.use_bn,
                              dropout_rate=self.dropout_rate,
                              is_training=self.is_training,
                              name="deep")
        deep_term = tf.layers.dense(deep_layer,
                                    units=1,
                                    activation=None,
                                    name="deep_term")
        self.output = tf.squeeze(tf.add(wide_term, deep_term))

    def _build_user_item(self):
        self.user_indices = tf.placeholder(tf.int32, shape=[None])
        self.item_indices = tf.placeholder(tf.int32, shape=[None])

        wide_user_feat = tf.get_variable(
            name="wide_user_feat",
            shape=[self.n_users + 1, 1],
            initializer=tf_truncated_normal(0.0, 0.01),
            regularizer=self.reg)
        wide_item_feat = tf.get_variable(
            name="wide_item_feat",
            shape=[self.n_items + 1, 1],
            initializer=tf_truncated_normal(0.0, 0.01),
            regularizer=self.reg)
        deep_user_feat = tf.get_variable(
            name="deep_user_feat",
            shape=[self.n_users + 1, self.embed_size],
            initializer=tf_truncated_normal(0.0, 0.01),
            regularizer=self.reg)
        deep_item_feat = tf.get_variable(
            name="deep_item_feat",
            shape=[self.n_items + 1, self.embed_size],
            initializer=tf_truncated_normal(0.0, 0.01),
            regularizer=self.reg)

        wide_user_embed = tf.nn.embedding_lookup(wide_user_feat,
                                                 self.user_indices)
        wide_item_embed = tf.nn.embedding_lookup(wide_item_feat,
                                                 self.item_indices)
        self.wide_embed.extend([wide_user_embed, wide_item_embed])

        deep_user_embed = tf.nn.embedding_lookup(deep_user_feat,
                                                 self.user_indices)
        deep_item_embed = tf.nn.embedding_lookup(deep_item_feat,
                                                 self.item_indices)
        self.deep_embed.extend([deep_user_embed, deep_item_embed])

    def _build_sparse(self):
        self.sparse_indices = tf.placeholder(
            tf.int32, shape=[None, self.sparse_field_size])

        wide_sparse_feat = tf.get_variable(
            name="wide_sparse_feat",
            shape=[self.sparse_feature_size],
            initializer=tf_truncated_normal(0.0, 0.01),
            regularizer=self.reg)
        deep_sparse_feat = tf.get_variable(
            name="deep_sparse_feat",
            shape=[self.sparse_feature_size, self.embed_size],
            initializer=tf_truncated_normal(0.0, 0.01),
            regularizer=self.reg)

        wide_sparse_embed = tf.nn.embedding_lookup(
            wide_sparse_feat, self.sparse_indices)
        deep_sparse_embed = tf.nn.embedding_lookup(
            deep_sparse_feat, self.sparse_indices)
        deep_sparse_embed = tf.reshape(
            deep_sparse_embed, [-1, self.sparse_field_size * self.embed_size])
        self.wide_embed.append(wide_sparse_embed)
        self.deep_embed.append(deep_sparse_embed)

    def _build_dense(self):
        self.dense_values = tf.placeholder(
            tf.float32, shape=[None, self.dense_field_size])
        dense_values_reshape = tf.reshape(
            self.dense_values, [-1, self.dense_field_size, 1])
        batch_size = tf.shape(self.dense_values)[0]

        wide_dense_feat = tf.get_variable(
            name="wide_dense_feat",
            shape=[self.dense_field_size],
            initializer=tf_truncated_normal(0.0, 0.01),
            regularizer=self.reg)
        deep_dense_feat = tf.get_variable(
            name="deep_dense_feat",
            shape=[self.dense_field_size, self.embed_size],
            initializer=tf_truncated_normal(0.0, 0.01),
            regularizer=self.reg)

        wide_dense_embed = tf.tile(wide_dense_feat, [batch_size])
        wide_dense_embed = tf.reshape(
            wide_dense_embed, [-1, self.dense_field_size])
        wide_dense_embed = tf.multiply(
            wide_dense_embed, self.dense_values)

        deep_dense_embed = tf.expand_dims(deep_dense_feat, axis=0)
        deep_dense_embed = tf.tile(deep_dense_embed, [batch_size, 1, 1])
        deep_dense_embed = tf.multiply(deep_dense_embed, dense_values_reshape)
        deep_dense_embed = tf.reshape(
            deep_dense_embed, [-1, self.dense_field_size * self.embed_size])
        self.wide_embed.append(wide_dense_embed)
        self.deep_embed.append(deep_dense_embed)

    def _build_train_ops(self, **kwargs):
        if self.task == "rating":
            self.loss = tf.losses.mean_squared_error(labels=self.labels,
                                                     predictions=self.output)
        elif self.task == "ranking":
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels,
                                                        logits=self.output)
            )

        if self.reg is not None:
            reg_keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            total_loss = self.loss + tf.add_n(reg_keys)
        else:
            total_loss = self.loss

        if self.lr_decay:
            n_batches = int(self.data_info.data_size / self.batch_size)
            self.lr, global_steps = lr_decay_config(self.lr, n_batches,
                                                    **kwargs)
        else:
            global_steps = None

        var_dict = var_list_by_name(names=["wide", "deep"])
        # print(f"{colorize('Wide_variables', 'blue')}: {var_dict['wide']}\n"
        #       f"{colorize('Deep_variables', 'blue')}: {var_dict['deep']}")
        wide_optimizer = tf.train.FtrlOptimizer(
            self.lr["wide"], l1_regularization_strength=1e-3)
        wide_optimizer_op = wide_optimizer.minimize(total_loss,
                                                    global_step=global_steps,
                                                    var_list=var_dict["wide"])

        deep_optimizer = tf.train.AdamOptimizer(self.lr["deep"])
        deep_optimizer_op = deep_optimizer.minimize(total_loss,
                                                    global_step=global_steps,
                                                    var_list=var_dict["deep"])

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.training_op = tf.group([wide_optimizer_op,
                                     deep_optimizer_op,
                                     update_ops])
        self.sess.run(tf.global_variables_initializer())

    def fit(self, train_data, verbose=1, shuffle=True,
            eval_data=None, metrics=None, **kwargs):
        self.show_start_time()
        if not self.graph_built:
            self._build_model()
            self._build_train_ops(**kwargs)

        if self.task == "ranking" and self.batch_sampling:
            self._check_has_sampled(train_data, verbose)
            data_generator = NegativeSampling(train_data,
                                              self.data_info,
                                              self.num_neg,
                                              self.sparse,
                                              self.dense,
                                              batch_sampling=True)

        else:
            data_generator = DataGenFeat(train_data,
                                         self.sparse,
                                         self.dense)

        self.train_feat(data_generator, verbose, shuffle, eval_data, metrics,
                        **kwargs)
        self.assign_oov()

    def predict(self, user, item, feats=None, cold_start="average",
                inner_id=False):
        user, item = self.convert_id(user, item, inner_id)
        unknown_num, unknown_index, user, item = self._check_unknown(user, item)

        (
            user_indices,
            item_indices,
            sparse_indices,
            dense_values
        ) = get_predict_indices_and_values(
            self.data_info, user, item, self.n_items, self.sparse, self.dense)

        if feats is not None:
            assert isinstance(feats, (dict, pd.Series)), (
                "feats must be dict or pandas.Series.")
            assert len(user_indices) == 1, "only support single user for feats"
            sparse_indices, dense_values = features_from_dict(
                self.data_info, sparse_indices, dense_values, feats, "predict")

        feed_dict = self._get_feed_dict(user_indices, item_indices,
                                        sparse_indices, dense_values,
                                        None, False)

        preds = self.sess.run(self.output, feed_dict)
        if self.task == "rating":
            preds = np.clip(preds, self.lower_bound, self.upper_bound)
        elif self.task == "ranking":
            preds = 1 / (1 + np.exp(-preds))

        if unknown_num > 0 and cold_start == "popular":
            preds[unknown_index] = self.default_prediction
        return preds

    def recommend_user(self, user, n_rec, user_feats=None, item_data=None,
                       cold_start="average", inner_id=False):
        user_id = self._check_unknown_user(user, inner_id)
        if user_id is None:
            if cold_start == "average":
                user_id = self.n_users
            elif cold_start == "popular":
                return self.data_info.popular_items[:n_rec]
            else:
                raise ValueError(user)

        (
            user_indices,
            item_indices,
            sparse_indices,
            dense_values
        ) = get_recommend_indices_and_values(
            self.data_info, user_id, self.n_items, self.sparse, self.dense)

        if user_feats is not None:
            assert isinstance(user_feats, (dict, pd.Series)), (
                "feats must be dict or pandas.Series.")
            sparse_indices, dense_values = features_from_dict(
                self.data_info, sparse_indices, dense_values, user_feats,
                "recommend")
        if item_data is not None:
            assert isinstance(item_data, pd.DataFrame), (
                "item_data must be pandas DataFrame")
            assert "item" in item_data.columns, (
                "item_data must contain 'item' column")
            sparse_indices, dense_values = add_item_features(
                self.data_info, sparse_indices, dense_values, item_data)

        feed_dict = self._get_feed_dict(user_indices, item_indices,
                                        sparse_indices, dense_values,
                                        None, False)

        recos = self.sess.run(self.output, feed_dict)
        if self.task == "ranking":
            recos = 1 / (1 + np.exp(-recos))
        consumed = set(self.user_consumed[user_id])
        count = n_rec + len(consumed)
        ids = np.argpartition(recos, -count)[-count:]
        rank = sorted(zip(ids, recos[ids]), key=lambda x: -x[1])
        recs_and_scores = islice(
            (rec if inner_id else (self.data_info.id2item[rec[0]], rec[1])
             for rec in rank if rec[0] not in consumed),
            n_rec
        )
        return list(recs_and_scores)

    def save(self, path, model_name, manual=True, inference_only=False):
        if not os.path.isdir(path):
            print(f"file folder {path} doesn't exists, creating a new one...")
            os.makedirs(path)
        self.save_params(path)
        if manual:
            self.save_variables(path, model_name, inference_only)
        else:
            self.save_tf_model(path, model_name)

    @classmethod
    def load(cls, path, model_name, data_info, manual=True):
        if manual:
            return cls.load_variables(path, model_name, data_info)
        else:
            return cls.load_tf_model(path, model_name, data_info)
