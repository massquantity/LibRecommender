"""

References: Yehuda Koren "Matrix Factorization Techniques for Recommender Systems"
            (https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)

author: massquantity

"""
import numpy as np
from tensorflow.keras.initializers import (
    zeros as tf_zeros,
    truncated_normal as tf_truncated_normal,
)

from ..bases import EmbedBase, TfMixin
from ..tfops import reg_config, tf
from ..training import TensorFlowTrainer


class SVD(EmbedBase, TfMixin):
    user_variables = ["bu_var", "pu_var"]
    item_variables = ["bi_var", "qi_var"]

    def __init__(
        self,
        task,
        data_info,
        loss_type="cross_entropy",
        embed_size=16,
        n_epochs=20,
        lr=0.01,
        lr_decay=False,
        reg=None,
        batch_size=256,
        num_neg=1,
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

        self.all_args = locals()
        self.reg = reg_config(reg)
        self.seed = seed
        if with_training:
            self._build_model()
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
        self.user_indices = tf.placeholder(tf.int32, shape=[None])
        self.item_indices = tf.placeholder(tf.int32, shape=[None])
        self.labels = tf.placeholder(tf.float32, shape=[None])

        self.bu_var = tf.get_variable(
            name="bu_var",
            shape=[self.n_users],
            initializer=tf_zeros,
            regularizer=self.reg,
        )
        self.bi_var = tf.get_variable(
            name="bi_var",
            shape=[self.n_items],
            initializer=tf_zeros,
            regularizer=self.reg,
        )
        self.pu_var = tf.get_variable(
            name="pu_var",
            shape=[self.n_users, self.embed_size],
            initializer=tf_truncated_normal(0.0, 0.05),
            regularizer=self.reg,
        )
        self.qi_var = tf.get_variable(
            name="qi_var",
            shape=[self.n_items, self.embed_size],
            initializer=tf_truncated_normal(0.0, 0.05),
            regularizer=self.reg,
        )

        bias_user = tf.nn.embedding_lookup(self.bu_var, self.user_indices)
        bias_item = tf.nn.embedding_lookup(self.bi_var, self.item_indices)
        embed_user = tf.nn.embedding_lookup(self.pu_var, self.user_indices)
        embed_item = tf.nn.embedding_lookup(self.qi_var, self.item_indices)
        self.output = (
            bias_user
            + bias_item
            + tf.reduce_sum(tf.multiply(embed_user, embed_item), axis=1)
        )

    def set_embeddings(self):
        bu, bi, pu, qi = self.sess.run(
            [self.bu_var, self.bi_var, self.pu_var, self.qi_var]
        )
        user_bias = np.ones([len(pu), 2], dtype=pu.dtype)
        user_bias[:, 0] = bu
        item_bias = np.ones([len(qi), 2], dtype=qi.dtype)
        item_bias[:, 1] = bi
        self.user_embed = np.hstack([pu, user_bias])
        self.item_embed = np.hstack([qi, item_bias])
