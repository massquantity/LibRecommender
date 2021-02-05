"""

Reference: Paul Covington et al.  "Deep Neural Networks for YouTube Recommendations"
           (https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf)

author: massquantity

"""
import os
from itertools import islice
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
    lr_decay_config
)
from ..data.data_generator import DataGenSequence
from ..data.sequence import user_last_interacted
from ..utils.misc import time_block, colorize
from ..utils.misc import count_params
from ..feature import (
    get_predict_indices_and_values,
    get_recommend_indices_and_values,
    features_from_dict,
    add_item_features
)
tf.disable_v2_behavior()


class YouTubeRanking(Base, TfMixin, EvalMixin):
    """
    The model implemented mainly corresponds to the ranking phase
    based on the original paper.
    """
    user_variables = ["user_features"]
    item_variables = ["item_features"]
    sparse_variables = ["sparse_features"]
    dense_variables = ["dense_features"]

    def __init__(
            self,
            task="ranking",
            data_info=None,
            embed_size=16,
            n_epochs=20,
            lr=0.01,
            lr_decay=False,
            reg=None,
            batch_size=256,
            num_neg=1,
            use_bn=True,
            dropout_rate=None,
            hidden_units="128,64,32",
            recent_num=10,
            random_num=None,
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
        self.lr = lr
        self.lr_decay = lr_decay
        self.reg = reg_config(reg)
        self.batch_size = batch_size
        self.num_neg = num_neg
        self.use_bn = use_bn
        self.dropout_rate = dropout_config(dropout_rate)
        self.hidden_units = list(map(int, hidden_units.split(",")))
        self.n_users = data_info.n_users
        self.n_items = data_info.n_items
        (
            self.interaction_mode,
            self.interaction_num
        ) = self._check_interaction_mode(recent_num, random_num)
        self.seed = seed
        self.user_consumed = data_info.user_consumed
        self.sparse = self._decide_sparse_indices(data_info)
        self.dense = self._decide_dense_values(data_info)
        if self.sparse:
            self.sparse_feature_size = self._sparse_feat_size(data_info)
            self.sparse_field_size = self._sparse_field_size(data_info)
        if self.dense:
            self.dense_field_size = self._dense_field_size(data_info)
        self.user_last_interacted = None
        self.last_interacted_len = None
        self.all_args = locals()

    def _build_model(self):
        self.graph_built = True
        tf.set_random_seed(self.seed)
        self.user_indices = tf.placeholder(tf.int32, shape=[None])
        self.item_indices = tf.placeholder(tf.int32, shape=[None])
        self.user_interacted_seq = tf.placeholder(
            tf.int32, shape=[None, self.interaction_num])
        self.user_interacted_len = tf.placeholder(tf.float32, shape=[None])
        self.labels = tf.placeholder(tf.float32, shape=[None])
        self.is_training = tf.placeholder_with_default(False, shape=[])
        self.concat_embed = []

        user_features = tf.get_variable(
            name="user_features",
            shape=[self.n_users + 1, self.embed_size],
            initializer=tf_truncated_normal(0.0, 0.01),
            regularizer=self.reg)
        item_features = tf.get_variable(
            name="item_features",
            shape=[self.n_items + 1, self.embed_size],
            initializer=tf_truncated_normal(0.0, 0.01),
            regularizer=self.reg)
        user_embed = tf.nn.embedding_lookup(user_features, self.user_indices)
        item_embed = tf.nn.embedding_lookup(item_features, self.item_indices)

        # unknown items are padded to 0-vector
        zero_padding_op = tf.scatter_update(
            item_features, self.n_items,
            tf.zeros([self.embed_size], dtype=tf.float32)
        )
        with tf.control_dependencies([zero_padding_op]):
            multi_item_embed = tf.nn.embedding_lookup(
                item_features, self.user_interacted_seq)  # B * seq * K
        pooled_embed = tf.div_no_nan(
            tf.reduce_sum(multi_item_embed, axis=1),
            tf.expand_dims(tf.sqrt(self.user_interacted_len), axis=1))
        self.concat_embed.extend([user_embed, item_embed, pooled_embed])

        if self.sparse:
            self._build_sparse()
        if self.dense:
            self._build_dense()

        concat_embed = tf.concat(self.concat_embed, axis=1)
        mlp_layer = dense_nn(concat_embed,
                             self.hidden_units,
                             use_bn=self.use_bn,
                             dropout_rate=self.dropout_rate,
                             is_training=self.is_training)
        self.output = tf.reshape(
            tf.layers.dense(inputs=mlp_layer, units=1), [-1])

    def _build_sparse(self):
        self.sparse_indices = tf.placeholder(
            tf.int32, shape=[None, self.sparse_field_size])
        sparse_features = tf.get_variable(
            name="sparse_features",
            shape=[self.sparse_feature_size, self.embed_size],
            initializer=tf_truncated_normal(0.0, 0.01),
            regularizer=self.reg)

        sparse_embed = tf.nn.embedding_lookup(
            sparse_features, self.sparse_indices)
        sparse_embed = tf.reshape(
            sparse_embed, [-1, self.sparse_field_size * self.embed_size])
        self.concat_embed.append(sparse_embed)

    def _build_dense(self):
        self.dense_values = tf.placeholder(
            tf.float32, shape=[None, self.dense_field_size])
        dense_values_reshape = tf.reshape(
            self.dense_values, [-1, self.dense_field_size, 1])
        batch_size = tf.shape(self.dense_values)[0]

        dense_features = tf.get_variable(
            name="dense_features",
            shape=[self.dense_field_size, self.embed_size],
            initializer=tf_truncated_normal(0.0, 0.01),
            regularizer=self.reg)

        dense_embed = tf.tile(dense_features, [batch_size, 1])
        dense_embed = tf.reshape(
            dense_embed, [-1, self.dense_field_size, self.embed_size])
        dense_embed = tf.multiply(dense_embed, dense_values_reshape)
        dense_embed = tf.reshape(
            dense_embed, [-1, self.dense_field_size * self.embed_size])
        self.concat_embed.append(dense_embed)

    def _build_train_ops(self, **kwargs):
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

        optimizer = tf.train.AdamOptimizer(self.lr)
        optimizer_op = optimizer.minimize(total_loss, global_step=global_steps)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.training_op = tf.group([optimizer_op, update_ops])
        self.sess.run(tf.global_variables_initializer())

    def fit(self, train_data, verbose=1, shuffle=True,
            eval_data=None, metrics=None, **kwargs):
        assert self.task == "ranking", (
            "YouTube models is only suitable for ranking")
        self.show_start_time()
        if not self.graph_built:
            self._build_model()
            self._build_train_ops(**kwargs)

        data_generator = DataGenSequence(train_data, self.data_info,
                                         self.sparse, self.dense,
                                         mode=self.interaction_mode,
                                         num=self.interaction_num,
                                         padding_idx=self.n_items)
        for epoch in range(1, self.n_epochs + 1):
            if self.lr_decay:
                print(f"With lr_decay, epoch {epoch} learning rate: "
                      f"{self.sess.run(self.lr)}")
            with time_block(f"Epoch {epoch}", verbose):
                train_total_loss = []
                for (u_seq, u_len, user, item, label, sparse_idx, dense_val
                     ) in data_generator(shuffle, self.batch_size):
                    feed_dict = self._get_seq_feed_dict(
                        u_seq, u_len, user, item, label,
                        sparse_idx, dense_val, True)
                    train_loss, _ = self.sess.run(
                        [self.loss, self.training_op], feed_dict)
                    train_total_loss.append(train_loss)

            if verbose > 1:
                train_loss_str = "train_loss: " + str(
                    round(float(np.mean(train_total_loss)), 4)
                )
                print(f"\t {colorize(train_loss_str, 'green')}")
                # for evaluation
                self._set_last_interacted()
                self.print_metrics(eval_data=eval_data, metrics=metrics,
                                   **kwargs)
                print("=" * 30)

        # for prediction and recommendation
        self._set_last_interacted()
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

        feed_dict = self._get_seq_feed_dict(self.user_last_interacted[user],
                                            self.last_interacted_len[user],
                                            user_indices, item_indices,
                                            None, sparse_indices,
                                            dense_values, False)

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

        u_last_interacted = np.tile(self.user_last_interacted[user],
                                    (self.n_items, 1))
        u_interacted_len = np.repeat(self.last_interacted_len[user],
                                     self.n_items)
        feed_dict = self._get_seq_feed_dict(u_last_interacted, u_interacted_len,
                                            user_indices, item_indices, None,
                                            sparse_indices, dense_values, False)

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

    def _set_last_interacted(self):
        if (self.user_last_interacted is None
                and self.last_interacted_len is None):
            user_indices = np.arange(self.n_users)
            (
                self.user_last_interacted,
                self.last_interacted_len
            ) = user_last_interacted(user_indices, self.user_consumed,
                                     self.n_items, self.interaction_num)

            oov = np.full(self.interaction_num, self.n_items, dtype=np.int32)
            self.user_last_interacted = np.vstack(
                [self.user_last_interacted, oov]
            )
            self.last_interacted_len = np.append(self.last_interacted_len, [1])

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
