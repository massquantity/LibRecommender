"""

Reference: Paul Covington et al.  "Deep Neural Networks for YouTube Recommendations"
           (https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf)

author: massquantity

"""
import numpy as np
from tensorflow.keras.initializers import (
    zeros as tf_zeros,
    truncated_normal as tf_truncated_normal,
)

from ..bases import EmbedBase, TfMixin
from ..data.sequence import sparse_user_last_interacted
from ..tfops import (
    dense_nn,
    dropout_config,
    multi_sparse_combine_embedding,
    reg_config,
    tf,
)
from ..training import YoutubeRetrievalTrainer
from ..utils.misc import count_params
from ..utils.validate import (
    check_dense_values,
    check_interaction_mode,
    check_multi_sparse,
    check_sparse_indices,
    dense_field_size,
    sparse_feat_size,
    sparse_field_size,
    true_sparse_field_size,
)


class YouTubeRetrieval(EmbedBase, TfMixin):
    """
    The model implemented mainly corresponds to the candidate generation
    phase based on the original paper.
    """

    item_variables = ["item_interaction_features", "nce_weights", "nce_biases"]
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
        num_sampled_per_batch=None,
        use_bn=True,
        dropout_rate=None,
        hidden_units="128,64",
        loss_type="nce",
        recent_num=10,
        random_num=None,
        multi_sparse_combiner="sqrtn",
        sampler="uniform",
        seed=42,
        k=10,
        eval_batch_size=8192,
        eval_user_num=None,
        lower_upper_bound=None,
        tf_sess_config=None,
        with_training=True,
    ):
        EmbedBase.__init__(self, task, data_info, embed_size, lower_upper_bound)
        TfMixin.__init__(self, data_info, tf_sess_config)

        assert task == "ranking", "YouTube models is only suitable for ranking"
        if len(data_info.item_col) > 0:
            raise ValueError("The YouTuBeRetrieval model assumes no item features.")
        self.all_args = locals()
        self.reg = reg_config(reg)
        self.use_bn = use_bn
        self.dropout_rate = dropout_config(dropout_rate)
        self.hidden_units = list(map(int, hidden_units.split(","))) + [self.embed_size]
        self.seed = seed
        self.interaction_mode, self.max_seq_len = check_interaction_mode(
            recent_num, random_num
        )
        self.sparse = check_sparse_indices(data_info)
        self.dense = check_dense_values(data_info)
        if self.sparse:
            self.sparse_feature_size = sparse_feat_size(data_info)
            self.sparse_field_size = sparse_field_size(data_info)
            self.multi_sparse_combiner = check_multi_sparse(
                data_info, multi_sparse_combiner
            )
            self.true_sparse_field_size = true_sparse_field_size(
                data_info, self.sparse_field_size, self.multi_sparse_combiner
            )
        if self.dense:
            self.dense_field_size = dense_field_size(data_info)
        (
            self.last_interacted_indices,
            self.last_interacted_values,
        ) = sparse_user_last_interacted(
            self.n_users, self.user_consumed, self.max_seq_len
        )
        if with_training:
            self._build_model()
            self.trainer = YoutubeRetrievalTrainer(
                self,
                task,
                loss_type,
                n_epochs,
                lr,
                lr_decay,
                batch_size,
                num_sampled_per_batch,
                k,
                eval_batch_size,
                eval_user_num,
                sampler,
            )

    def _build_model(self):
        tf.set_random_seed(self.seed)
        # item_indices actually serve as labels in YouTuBeRetrieval model
        self.item_indices = tf.placeholder(tf.int64, shape=[None])
        self.is_training = tf.placeholder_with_default(False, shape=[])
        self.concat_embed = []

        self._build_item_interaction()
        self._build_variables()
        if self.sparse:
            self._build_sparse()
        if self.dense:
            self._build_dense()

        concat_features = tf.concat(self.concat_embed, axis=1)
        self.user_vector_repr = dense_nn(
            concat_features,
            self.hidden_units,
            use_bn=self.use_bn,
            dropout_rate=self.dropout_rate,
            is_training=self.is_training,
        )
        count_params()

    def _build_item_interaction(self):
        self.item_interaction_indices = tf.placeholder(tf.int64, shape=[None, 2])
        self.item_interaction_values = tf.placeholder(tf.int32, shape=[None])
        self.modified_batch_size = tf.placeholder(tf.int32, shape=[])

        item_interaction_features = tf.get_variable(
            name="item_interaction_features",
            shape=[self.n_items, self.embed_size],
            initializer=tf_truncated_normal(0.0, 0.01),
            regularizer=self.reg,
        )
        sparse_item_interaction = tf.SparseTensor(
            self.item_interaction_indices,
            self.item_interaction_values,
            [self.modified_batch_size, self.n_items],
        )
        pooled_embed = tf.nn.safe_embedding_lookup_sparse(
            item_interaction_features,
            sparse_item_interaction,
            sparse_weights=None,
            combiner="sqrtn",
            default_id=None,
        )  # unknown user will return 0-vector
        self.concat_embed.append(pooled_embed)

    def _build_sparse(self):
        self.sparse_indices = tf.placeholder(
            tf.int32, shape=[None, self.sparse_field_size]
        )
        sparse_features = tf.get_variable(
            name="sparse_features",
            shape=[self.sparse_feature_size, self.embed_size],
            initializer=tf_truncated_normal(0.0, 0.01),
            regularizer=self.reg,
        )

        if self.data_info.multi_sparse_combine_info and self.multi_sparse_combiner in (
            "sum",
            "mean",
            "sqrtn",
        ):
            sparse_embed = multi_sparse_combine_embedding(
                self.data_info,
                sparse_features,
                self.sparse_indices,
                self.multi_sparse_combiner,
                self.embed_size,
            )
        else:
            sparse_embed = tf.nn.embedding_lookup(sparse_features, self.sparse_indices)

        sparse_embed = tf.reshape(
            sparse_embed, [-1, self.true_sparse_field_size * self.embed_size]
        )
        self.concat_embed.append(sparse_embed)

    def _build_dense(self):
        self.dense_values = tf.placeholder(
            tf.float32, shape=[None, self.dense_field_size]
        )
        dense_values_reshape = tf.reshape(
            self.dense_values, [-1, self.dense_field_size, 1]
        )
        batch_size = tf.shape(self.dense_values)[0]

        dense_features = tf.get_variable(
            name="dense_features",
            shape=[self.dense_field_size, self.embed_size],
            initializer=tf_truncated_normal(0.0, 0.01),
            regularizer=self.reg,
        )

        dense_embed = tf.expand_dims(dense_features, axis=0)
        # B * F2 * K
        dense_embed = tf.tile(dense_embed, [batch_size, 1, 1])
        dense_embed = tf.multiply(dense_embed, dense_values_reshape)
        dense_embed = tf.reshape(
            dense_embed, [-1, self.dense_field_size * self.embed_size]
        )
        self.concat_embed.append(dense_embed)

    def _build_variables(self):
        self.nce_weights = tf.get_variable(
            name="nce_weights",
            # n_classes, embed_size
            shape=[self.n_items, self.embed_size],
            initializer=tf_truncated_normal(0.0, 0.01),
            regularizer=self.reg,
        )
        self.nce_biases = tf.get_variable(
            name="nce_biases",
            shape=[self.n_items],
            initializer=tf_zeros,
            regularizer=self.reg,
            trainable=True,
        )

    def set_embeddings(self):
        feed_dict = {
            self.item_interaction_indices: self.last_interacted_indices,
            self.item_interaction_values: self.last_interacted_values,
            self.modified_batch_size: self.n_users,
            self.is_training: False,
        }
        if self.sparse:
            # remove oov
            user_sparse_indices = self.data_info.user_sparse_unique[:-1]
            feed_dict.update({self.sparse_indices: user_sparse_indices})
        if self.dense:
            user_dense_values = self.data_info.user_dense_unique[:-1]
            feed_dict.update({self.dense_values: user_dense_values})

        user_vector = self.sess.run(self.user_vector_repr, feed_dict)
        item_weights = self.sess.run(self.nce_weights)
        item_biases = self.sess.run(self.nce_biases)

        user_bias = np.ones([len(user_vector), 1], dtype=user_vector.dtype)
        item_bias = item_biases[:, None]
        self.user_embed = np.hstack([user_vector, user_bias])
        self.item_embed = np.hstack([item_weights, item_bias])
