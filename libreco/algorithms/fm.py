"""

References:
    [1] Steffen Rendle "Factorization Machines"
        (https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    [2] Xiangnan He et al. "Neural Factorization Machines for Sparse Predictive Analytics"
        (https://arxiv.org/pdf/1708.05027.pdf)

author: massquantity

"""
from tensorflow.keras.initializers import truncated_normal as tf_truncated_normal

from ..bases import TfBase
from ..tfops import (
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


class FM(TfBase):
    """
    Note this implementation is actually a mixture of FM and NFM,
    since it uses one dense layer in the final output
    """

    user_variables = ["linear_user_feat", "pairwise_user_feat"]
    item_variables = ["linear_item_feat", "pairwise_item_feat"]
    sparse_variables = ["linear_sparse_feat", "pairwise_sparse_feat"]
    dense_variables = ["linear_dense_feat", "pairwise_dense_feat"]

    def __init__(
        self,
        task,
        data_info=None,
        loss_type="cross_entropy",
        embed_size=16,
        n_epochs=20,
        lr=0.01,
        lr_decay=False,
        reg=None,
        batch_size=256,
        num_neg=1,
        use_bn=True,
        dropout_rate=None,
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
        self.batch_size = batch_size
        self.use_bn = use_bn
        self.dropout_rate = dropout_config(dropout_rate)
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
        self.linear_embed, self.pairwise_embed = [], []

        self._build_user_item()
        if self.sparse:
            self._build_sparse()
        if self.dense:
            self._build_dense()

        linear_embed = tf.concat(self.linear_embed, axis=1)
        pairwise_embed = tf.concat(self.pairwise_embed, axis=1)

        # B * 1
        linear_term = tf_dense(units=1, activation=None)(linear_embed)
        # B * K
        pairwise_term = 0.5 * tf.subtract(
            tf.square(tf.reduce_sum(pairwise_embed, axis=1)),
            tf.reduce_sum(tf.square(pairwise_embed), axis=1),
        )

        # For original FM, just add K dim together:
        # pairwise_term = 0.5 * tf.reduce_sum(pairwise_term, axis=1)
        if self.use_bn:
            pairwise_term = tf.layers.batch_normalization(
                pairwise_term, training=self.is_training
            )
        pairwise_term = tf_dense(units=1, activation=tf.nn.elu)(pairwise_term)
        self.output = tf.squeeze(tf.add(linear_term, pairwise_term))
        count_params()

    def _build_user_item(self):
        self.user_indices = tf.placeholder(tf.int32, shape=[None])
        self.item_indices = tf.placeholder(tf.int32, shape=[None])

        linear_user_feat = tf.get_variable(
            name="linear_user_feat",
            shape=[self.n_users + 1, 1],
            initializer=tf_truncated_normal(0.0, 0.03),
            regularizer=self.reg,
        )
        linear_item_feat = tf.get_variable(
            name="linear_item_feat",
            shape=[self.n_items + 1, 1],
            initializer=tf_truncated_normal(0.0, 0.03),
            regularizer=self.reg,
        )
        pairwise_user_feat = tf.get_variable(
            name="pairwise_user_feat",
            shape=[self.n_users + 1, self.embed_size],
            initializer=tf_truncated_normal(0.0, 0.03),
            regularizer=self.reg,
        )
        pairwise_item_feat = tf.get_variable(
            name="pairwise_item_feat",
            shape=[self.n_items + 1, self.embed_size],
            initializer=tf_truncated_normal(0.0, 0.03),
            regularizer=self.reg,
        )

        # print(linear_embed.get_shape().as_list())
        linear_user_embed = tf.nn.embedding_lookup(linear_user_feat, self.user_indices)
        linear_item_embed = tf.nn.embedding_lookup(linear_item_feat, self.item_indices)
        self.linear_embed.extend([linear_user_embed, linear_item_embed])

        pairwise_user_embed = tf.expand_dims(
            tf.nn.embedding_lookup(pairwise_user_feat, self.user_indices), axis=1
        )
        pairwise_item_embed = tf.expand_dims(
            tf.nn.embedding_lookup(pairwise_item_feat, self.item_indices), axis=1
        )
        self.pairwise_embed.extend([pairwise_user_embed, pairwise_item_embed])

    def _build_sparse(self):
        self.sparse_indices = tf.placeholder(
            tf.int32, shape=[None, self.sparse_field_size]
        )

        linear_sparse_feat = tf.get_variable(
            name="linear_sparse_feat",
            shape=[self.sparse_feature_size],
            initializer=tf_truncated_normal(0.0, 0.03),
            regularizer=self.reg,
        )
        pairwise_sparse_feat = tf.get_variable(
            name="pairwise_sparse_feat",
            shape=[self.sparse_feature_size, self.embed_size],
            initializer=tf_truncated_normal(0.0, 0.03),
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
                pairwise_sparse_feat,
                self.sparse_indices,
                self.multi_sparse_combiner,
                self.embed_size,
            )
        else:
            linear_sparse_embed = tf.nn.embedding_lookup(  # B * F1
                linear_sparse_feat, self.sparse_indices
            )
            pairwise_sparse_embed = tf.nn.embedding_lookup(  # B * F1 * K
                pairwise_sparse_feat, self.sparse_indices
            )

        self.linear_embed.append(linear_sparse_embed)
        self.pairwise_embed.append(pairwise_sparse_embed)

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
            initializer=tf_truncated_normal(0.0, 0.03),
            regularizer=self.reg,
        )
        pairwise_dense_feat = tf.get_variable(
            name="pairwise_dense_feat",
            shape=[self.dense_field_size, self.embed_size],
            initializer=tf_truncated_normal(0.0, 0.03),
            regularizer=self.reg,
        )

        # B * F2
        linear_dense_embed = tf.tile(linear_dense_feat, [batch_size])
        linear_dense_embed = tf.reshape(linear_dense_embed, [-1, self.dense_field_size])
        linear_dense_embed = tf.multiply(linear_dense_embed, self.dense_values)

        pairwise_dense_embed = tf.expand_dims(pairwise_dense_feat, axis=0)
        # B * F2 * K
        pairwise_dense_embed = tf.tile(pairwise_dense_embed, [batch_size, 1, 1])
        pairwise_dense_embed = tf.multiply(pairwise_dense_embed, dense_values_reshape)
        self.linear_embed.append(linear_dense_embed)
        self.pairwise_embed.append(pairwise_dense_embed)
