"""

Reference: Huifeng Guo et al. "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction"
           (https://arxiv.org/pdf/1703.04247.pdf)

author: massquantity

"""
from tensorflow.keras.initializers import truncated_normal as tf_truncated_normal

from ..bases import TfBase
from ..tfops import (
    dense_nn,
    dropout_config,
    multi_sparse_combine_embedding,
    reg_config,
    tf,
    tf_dense,
)
from ..training import TensorFlowTrainer
from ..utils.misc import count_params
from ..utils.validate import (
    check_dense_values,
    check_multi_sparse,
    check_sparse_indices,
    dense_field_size,
    sparse_feat_size,
    sparse_field_size,
    true_sparse_field_size,
)


class DeepFM(TfBase):
    user_variables = ["linear_user_feat", "embed_user_feat"]
    item_variables = ["linear_item_feat", "embed_item_feat"]
    sparse_variables = ["linear_sparse_feat", "embed_sparse_feat"]
    dense_variables = ["linear_dense_feat", "embed_dense_feat"]

    def __init__(
        self,
        task,
        data_info=None,
        loss_type="cross_entropy",
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
        multi_sparse_combiner="sqrtn",
        seed=42,
        k=10,
        eval_batch_size=8192,
        eval_user_num=None,
        lower_upper_bound=None,
        tf_sess_config=None,
        with_training=True,
    ):
        super().__init__(task, data_info, lower_upper_bound, tf_sess_config)

        self.all_args = locals()
        self.embed_size = embed_size
        self.reg = reg_config(reg)
        self.use_bn = use_bn
        self.dropout_rate = dropout_config(dropout_rate)
        self.hidden_units = list(map(int, hidden_units.split(",")))
        self.seed = seed
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
        self._build_model()
        if with_training:
            self.trainer = TensorFlowTrainer(
                self,
                task,
                loss_type,
                n_epochs,
                lr,
                lr_decay,
                batch_size,
                num_neg,
                k,
                eval_batch_size,
                eval_user_num,
            )

    def _build_model(self):
        tf.set_random_seed(self.seed)
        self.labels = tf.placeholder(tf.float32, shape=[None])
        self.is_training = tf.placeholder_with_default(False, shape=[])
        self.linear_embed, self.pairwise_embed, self.deep_embed = [], [], []

        self._build_user_item()
        if self.sparse:
            self._build_sparse()
        if self.dense:
            self._build_dense()

        linear_embed = tf.concat(self.linear_embed, axis=1)
        pairwise_embed = tf.concat(self.pairwise_embed, axis=1)
        deep_embed = tf.concat(self.deep_embed, axis=1)

        linear_term = tf_dense(units=1, activation=None)(linear_embed)
        pairwise_term = 0.5 * tf.subtract(
            tf.square(tf.reduce_sum(pairwise_embed, axis=1)),
            tf.reduce_sum(tf.square(pairwise_embed), axis=1),
        )
        deep_term = dense_nn(
            deep_embed,
            self.hidden_units,
            use_bn=self.use_bn,
            dropout_rate=self.dropout_rate,
            is_training=self.is_training,
        )

        concat_layer = tf.concat([linear_term, pairwise_term, deep_term], axis=1)
        self.output = tf.squeeze(tf_dense(units=1, activation=None)(concat_layer))
        count_params()

    def _build_user_item(self):
        self.user_indices = tf.placeholder(tf.int32, shape=[None])
        self.item_indices = tf.placeholder(tf.int32, shape=[None])

        linear_user_feat = tf.get_variable(
            name="linear_user_feat",
            shape=[self.n_users + 1, 1],
            initializer=tf_truncated_normal(0.0, 0.01),
            regularizer=self.reg,
        )
        linear_item_feat = tf.get_variable(
            name="linear_item_feat",
            shape=[self.n_items + 1, 1],
            initializer=tf_truncated_normal(0.0, 0.01),
            regularizer=self.reg,
        )
        embed_user_feat = tf.get_variable(
            name="embed_user_feat",
            shape=[self.n_users + 1, self.embed_size],
            initializer=tf_truncated_normal(0.0, 0.01),
            regularizer=self.reg,
        )
        embed_item_feat = tf.get_variable(
            name="embed_item_feat",
            shape=[self.n_items + 1, self.embed_size],
            initializer=tf_truncated_normal(0.0, 0.01),
            regularizer=self.reg,
        )

        linear_user_embed = tf.nn.embedding_lookup(linear_user_feat, self.user_indices)
        linear_item_embed = tf.nn.embedding_lookup(linear_item_feat, self.item_indices)
        self.linear_embed.extend([linear_user_embed, linear_item_embed])

        pairwise_user_embed = tf.expand_dims(
            tf.nn.embedding_lookup(embed_user_feat, self.user_indices), axis=1
        )
        pairwise_item_embed = tf.expand_dims(
            tf.nn.embedding_lookup(embed_item_feat, self.item_indices), axis=1
        )
        self.pairwise_embed.extend([pairwise_user_embed, pairwise_item_embed])

        deep_user_embed = tf.nn.embedding_lookup(embed_user_feat, self.user_indices)
        deep_item_embed = tf.nn.embedding_lookup(embed_item_feat, self.item_indices)
        self.deep_embed.extend([deep_user_embed, deep_item_embed])

    def _build_sparse(self):
        self.sparse_indices = tf.placeholder(
            tf.int32, shape=[None, self.sparse_field_size]
        )
        linear_sparse_feat = tf.get_variable(
            name="linear_sparse_feat",
            shape=[self.sparse_feature_size],
            initializer=tf_truncated_normal(0.0, 0.01),
            regularizer=self.reg,
        )
        embed_sparse_feat = tf.get_variable(
            name="embed_sparse_feat",
            shape=[self.sparse_feature_size, self.embed_size],
            initializer=tf_truncated_normal(0.0, 0.01),
            regularizer=self.reg,
        )

        if self.data_info.multi_sparse_combine_info and self.multi_sparse_combiner in (
            "sum",
            "mean",
            "sqrtn",
        ):
            linear_sparse_embed = multi_sparse_combine_embedding(
                self.data_info,
                linear_sparse_feat,
                self.sparse_indices,
                self.multi_sparse_combiner,
                embed_size=1,
            )
            pairwise_sparse_embed = multi_sparse_combine_embedding(
                self.data_info,
                embed_sparse_feat,
                self.sparse_indices,
                self.multi_sparse_combiner,
                self.embed_size,
            )
        else:
            linear_sparse_embed = tf.nn.embedding_lookup(  # B * F1
                linear_sparse_feat, self.sparse_indices
            )
            pairwise_sparse_embed = tf.nn.embedding_lookup(  # B * F1 * K
                embed_sparse_feat, self.sparse_indices
            )

        deep_sparse_embed = tf.reshape(
            pairwise_sparse_embed, [-1, self.true_sparse_field_size * self.embed_size]
        )
        self.linear_embed.append(linear_sparse_embed)
        self.pairwise_embed.append(pairwise_sparse_embed)
        self.deep_embed.append(deep_sparse_embed)

    def _build_dense(self):
        self.dense_values = tf.placeholder(
            tf.float32, shape=[None, self.dense_field_size]
        )
        dense_values_reshape = tf.reshape(
            self.dense_values, [-1, self.dense_field_size, 1]
        )
        batch_size = tf.shape(self.dense_values)[0]

        linear_dense_feat = tf.get_variable(
            name="linear_dense_feat",
            shape=[self.dense_field_size],
            initializer=tf_truncated_normal(0.0, 0.01),
            regularizer=self.reg,
        )
        embed_dense_feat = tf.get_variable(
            name="embed_dense_feat",
            shape=[self.dense_field_size, self.embed_size],
            initializer=tf_truncated_normal(0.0, 0.01),
            regularizer=self.reg,
        )

        # B * F2
        linear_dense_embed = tf.tile(linear_dense_feat, [batch_size])
        linear_dense_embed = tf.reshape(linear_dense_embed, [-1, self.dense_field_size])
        linear_dense_embed = tf.multiply(linear_dense_embed, self.dense_values)

        pairwise_dense_embed = tf.expand_dims(embed_dense_feat, axis=0)
        # B * F2 * K
        pairwise_dense_embed = tf.tile(pairwise_dense_embed, [batch_size, 1, 1])
        pairwise_dense_embed = tf.multiply(pairwise_dense_embed, dense_values_reshape)

        deep_dense_embed = tf.reshape(
            pairwise_dense_embed, [-1, self.dense_field_size * self.embed_size]
        )
        self.linear_embed.append(linear_dense_embed)
        self.pairwise_embed.append(pairwise_dense_embed)
        self.deep_embed.append(deep_dense_embed)
