"""

Reference: Guorui Zhou et al.  "Deep Interest Network for Click-Through Rate Prediction"
           (https://arxiv.org/pdf/1706.06978.pdf)

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


class DIN(Base, TfMixin, EvalMixin):
    user_variables = ["user_feat"]
    item_variables = ["item_feat"]
    sparse_variables = ["sparse_feat"]
    dense_variables = ["dense_feat"]

    def __init__(
            self,
            task,
            data_info=None,
            embed_size=16,
            n_epochs=20,
            lr=0.001,
            lr_decay=False,
            reg=None,
            batch_size=256,
            num_neg=1,
            use_bn=True,
            dropout_rate=None,
            hidden_units="128,64,32",
            recent_num=10,
            random_num=None,
            use_tf_attention=False,
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
        self.use_tf_attention = use_tf_attention
        (
            self.interaction_mode,
            self.max_seq_len
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
        self.item_sparse = (
            True
            if data_info.item_sparse_unique is not None
            else False
        )
        self.item_dense = (
            True
            if data_info.item_dense_unique is not None
            else False
        )
        if self.item_sparse:
            # item sparse col indices in all sparse cols
            self.item_sparse_col_indices = data_info.item_sparse_col.index
        if self.item_dense:
            # item dense col indices in all dense cols
            self.item_dense_col_indices = data_info.item_dense_col.index
        self.user_last_interacted = None
        self.last_interacted_len = None
        self.all_args = locals()

    def _build_model(self):
        self.graph_built = True
        tf.set_random_seed(self.seed)
        self.concat_embed, self.item_embed, self.seq_embed = [], [], []
        self._build_placeholders()
        self._build_variables()
        self._build_user_item()
        if self.sparse:
            self._build_sparse()
        if self.dense:
            self._build_dense()
        self._build_attention()

        concat_embed = tf.concat(self.concat_embed, axis=1)
        mlp_layer = dense_nn(concat_embed,
                             self.hidden_units,
                             use_bn=self.use_bn,
                             dropout_rate=self.dropout_rate,
                             is_training=self.is_training,
                             name="mlp")
        self.output = tf.reshape(
            tf.layers.dense(inputs=mlp_layer, units=1), [-1])
        count_params()

    def _build_placeholders(self):
        self.user_indices = tf.placeholder(tf.int32, shape=[None])
        self.item_indices = tf.placeholder(tf.int32, shape=[None])
        self.user_interacted_seq = tf.placeholder(
            tf.int32, shape=[None, self.max_seq_len])   # B * seq
        self.user_interacted_len = tf.placeholder(tf.float32, shape=[None])
        self.labels = tf.placeholder(tf.float32, shape=[None])
        self.is_training = tf.placeholder_with_default(False, shape=[])

        if self.sparse:
            self.sparse_indices = tf.placeholder(
                tf.int32, shape=[None, self.sparse_field_size])
        if self.dense:
            self.dense_values = tf.placeholder(
                tf.float32, shape=[None, self.dense_field_size])

    def _build_variables(self):
        self.user_feat = tf.get_variable(
            name="user_feat",
            shape=[self.n_users + 1, self.embed_size],
            initializer=tf_truncated_normal(0.0, 0.01),
            regularizer=self.reg)
        self.item_feat = tf.get_variable(
            name="item_feat",
            shape=[self.n_items + 1, self.embed_size],
            initializer=tf_truncated_normal(0.0, 0.01),
            regularizer=self.reg)
        if self.sparse:
            self.sparse_feat = tf.get_variable(
                name="sparse_feat",
                shape=[self.sparse_feature_size, self.embed_size],
                initializer=tf_truncated_normal(0.0, 0.01),
                regularizer=self.reg)
        if self.dense:
            self.dense_feat = tf.get_variable(
                name="dense_feat",
                shape=[self.dense_field_size, self.embed_size],
                initializer=tf_truncated_normal(0.0, 0.01),
                regularizer=self.reg)

    def _build_user_item(self):
        user_embed = tf.nn.embedding_lookup(self.user_feat, self.user_indices)
        item_embed = tf.nn.embedding_lookup(self.item_feat, self.item_indices)
        self.concat_embed.extend([user_embed, item_embed])
        self.item_embed.append(item_embed)

    def _build_sparse(self):
        sparse_embed = tf.nn.embedding_lookup(
            self.sparse_feat, self.sparse_indices)
        self.concat_embed.append(tf.reshape(
            sparse_embed, [-1, self.sparse_field_size * self.embed_size])
        )

        if self.item_sparse:
            item_sparse_embed = tf.layers.flatten(
                tf.gather(sparse_embed, self.item_sparse_col_indices, axis=1)
            )
            self.item_embed.append(item_sparse_embed)

    def _build_dense(self):
        batch_size = tf.shape(self.dense_values)[0]
        # 1 * F_dense * K
        dense_embed = tf.expand_dims(self.dense_feat, axis=0)
        # B * F_dense * K
        dense_embed = tf.tile(dense_embed, [batch_size, 1, 1])
        dense_values_reshape = tf.reshape(
            self.dense_values, [-1, self.dense_field_size, 1])
        dense_embed = tf.multiply(dense_embed, dense_values_reshape)
        self.concat_embed.append(tf.reshape(
            dense_embed, [-1, self.dense_field_size * self.embed_size])
        )

        if self.item_dense:
            item_dense_embed = tf.layers.flatten(
                tf.gather(dense_embed, self.item_dense_col_indices, axis=1)
            )
            self.item_embed.append(item_dense_embed)

    def _build_attention(self):
        # B * seq * K
        seq_item_embed = tf.nn.embedding_lookup(
            self.item_feat, self.user_interacted_seq)
        self.seq_embed.append(seq_item_embed)

        if self.item_sparse:
            # contains unique field indices for each item
            item_sparse_fields = tf.convert_to_tensor(
                self.data_info.item_sparse_unique, dtype=tf.int64)
            item_sparse_fields_num = tf.shape(item_sparse_fields)[1]

            # B * seq * F_sparse
            seq_sparse_fields = tf.gather(
                item_sparse_fields, self.user_interacted_seq)
            # B * seq * F_sparse * K
            seq_sparse_embed = tf.nn.embedding_lookup(
                self.sparse_feat, seq_sparse_fields)
            # B * seq * FK
            seq_sparse_embed = tf.reshape(
                seq_sparse_embed,
                [-1, self.max_seq_len, item_sparse_fields_num * self.embed_size]
            )
            self.seq_embed.append(seq_sparse_embed)

        if self.item_dense:
            # contains unique dense values for each item
            item_dense_values = tf.convert_to_tensor(
                self.data_info.item_dense_unique, dtype=tf.float32)
            item_dense_fields_num = tf.shape(item_dense_values)[1]
            # B * seq * F_dense
            seq_dense_values = tf.gather(
                item_dense_values, self.user_interacted_seq)
            # B * seq * F_dense * 1
            seq_dense_values = tf.expand_dims(seq_dense_values, axis=-1)

            batch_size = tf.shape(seq_dense_values)[0]
            dense_embed = tf.reshape(
                self.dense_feat, [1, 1, self.dense_field_size, self.embed_size])
            # B * seq * F_dense * K
            # Since dense_embeddings are same for all items,
            # we can simply repeat it (batch * seq) times
            seq_dense_embed = tf.tile(
                dense_embed, [batch_size, self.max_seq_len, 1, 1])
            seq_dense_embed = tf.multiply(
                seq_dense_embed, seq_dense_values)
            # B * seq * FK
            seq_dense_embed = tf.reshape(
                seq_dense_embed,
                [-1, self.max_seq_len, item_dense_fields_num * self.embed_size]
            )
            self.seq_embed.append(seq_dense_embed)

        # B * K
        item_total_embed = tf.concat(self.item_embed, axis=1)
        # B * seq * K
        seq_total_embed = tf.concat(self.seq_embed, axis=2)

        attention_layer = self._attention_unit(
            item_total_embed, seq_total_embed, self.user_interacted_len)
        self.concat_embed.append(tf.layers.flatten(attention_layer))

    def _attention_unit(self, queries, keys, keys_len):
        if self.use_tf_attention:
            query_masks = tf.cast(
                tf.ones_like(tf.reshape(self.user_interacted_len, [-1, 1])),
                dtype=tf.bool
            )
            key_masks = tf.sequence_mask(
                self.user_interacted_len, self.max_seq_len
            )
            queries = tf.expand_dims(queries, axis=1)
            attention = tf.keras.layers.Attention(use_scale=False)
            pooled_outputs = attention(inputs=[queries, keys],
                                       mask=[query_masks, key_masks])
            return pooled_outputs
        else:
            # queries: B * K, keys: B * seq * K
            queries = tf.expand_dims(queries, axis=1)
            # B * seq * K
            queries = tf.tile(queries, [1, self.max_seq_len, 1])
            queries_keys_cross = tf.concat(
                [queries, keys, queries - keys, queries * keys], axis=2)
            mlp_layer = dense_nn(queries_keys_cross, (16,), use_bn=False,
                                 activation=tf.nn.sigmoid, name="attention")
            # B * seq * 1
            mlp_layer = tf.layers.dense(mlp_layer, units=1, activation=None)
            # attention_weights = tf.transpose(mlp_layer, [0, 2, 1])
            attention_weights = tf.layers.flatten(mlp_layer)

            key_masks = tf.sequence_mask(keys_len, self.max_seq_len)
            paddings = tf.ones_like(attention_weights) * (-2**32 + 1)
            attention_scores = tf.where(key_masks, attention_weights, paddings)
            attention_scores = tf.div_no_nan(
                attention_scores,
                tf.sqrt(
                    tf.cast(keys.get_shape().as_list()[-1], tf.float32)
                )
            )
            # B * 1 * seq
            attention_scores = tf.expand_dims(
                tf.nn.softmax(attention_scores), 1)
            # B * 1 * K
            pooled_outputs = attention_scores @ keys
            return pooled_outputs

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

        optimizer = tf.train.AdamOptimizer(self.lr)
        optimizer_op = optimizer.minimize(total_loss, global_step=global_steps)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.training_op = tf.group([optimizer_op, update_ops])
        self.sess.run(tf.global_variables_initializer())

    def fit(self, train_data, verbose=1, shuffle=True,
            eval_data=None, metrics=None, **kwargs):
        self.show_start_time()
        if not self.graph_built:
            self._build_model()
            self._build_train_ops(**kwargs)

        data_generator = DataGenSequence(train_data, self.data_info,
                                         self.sparse, self.dense,
                                         mode=self.interaction_mode,
                                         num=self.max_seq_len,
                                         padding_idx=0)
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
                        sparse_idx, dense_val, True
                    )
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

        u_last_interacted = np.tile(self.user_last_interacted[user_id],
                                    (self.n_items, 1))
        u_interacted_len = np.repeat(self.last_interacted_len[user_id],
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
                                     self.n_items, self.max_seq_len)

            oov = np.full(self.max_seq_len, self.n_items, dtype=np.int32)
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
