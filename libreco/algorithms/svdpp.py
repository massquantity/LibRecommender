"""

References: Yehuda Koren "Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model"
            (https://dl.acm.org/citation.cfm?id=1401944)

author: massquantity

"""
import numpy as np

from ..bases import EmbedBase
from ..tfops import rebuild_tf_model, reg_config, sess_config, tf


class SVDpp(EmbedBase):
    user_variables = ["bu_var", "pu_var", "yj_var"]
    item_variables = ["bi_var", "qi_var"]

    def __init__(
        self,
        task,
        data_info,
        loss_type="cross_entropy",
        embed_size=16,
        n_epochs=20,
        lr=0.001,
        lr_decay=False,
        epsilon=1e-5,
        reg=None,
        batch_size=256,
        num_neg=1,
        seed=42,
        recent_num=30,
        lower_upper_bound=None,
        tf_sess_config=None,
    ):
        super().__init__(task, data_info, embed_size, lower_upper_bound)

        self.all_args = locals()
        self.sess = sess_config(tf_sess_config)
        self.loss_type = loss_type
        self.n_epochs = n_epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.epsilon = epsilon
        self.reg = reg_config(reg)
        self.batch_size = batch_size
        self.num_neg = num_neg
        self.recent_num = recent_num
        self.seed = seed
        self.sparse_interaction = None

    def build_model(self):
        tf.set_random_seed(self.seed)
        self.user_indices = tf.placeholder(tf.int32, shape=[None])
        self.item_indices = tf.placeholder(tf.int32, shape=[None])
        self.labels = tf.placeholder(tf.float32, shape=[None])

        self.bu_var = tf.get_variable(
            name="bu_var",
            shape=[self.n_users],
            initializer=tf.zeros_initializer(),
            regularizer=self.reg,
        )
        self.bi_var = tf.get_variable(
            name="bi_var",
            shape=[self.n_items],
            initializer=tf.zeros_initializer(),
            regularizer=self.reg,
        )
        self.pu_var = tf.get_variable(
            name="pu_var",
            shape=[self.n_users, self.embed_size],
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )
        self.qi_var = tf.get_variable(
            name="qi_var",
            shape=[self.n_items, self.embed_size],
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )

        yj_var = tf.get_variable(
            name="yj_var",
            shape=[self.n_items, self.embed_size],
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )
        uj = tf.nn.safe_embedding_lookup_sparse(
            yj_var,
            self.sparse_interaction,
            sparse_weights=None,
            combiner="sqrtn",
            default_id=None,
        )  # unknown user will return 0-vector
        self.puj_var = self.pu_var + uj

        bias_user = tf.nn.embedding_lookup(self.bu_var, self.user_indices)
        bias_item = tf.nn.embedding_lookup(self.bi_var, self.item_indices)
        embed_user = tf.nn.embedding_lookup(self.puj_var, self.user_indices)
        embed_item = tf.nn.embedding_lookup(self.qi_var, self.item_indices)
        self.output = (
            bias_user
            + bias_item
            + tf.reduce_sum(tf.multiply(embed_user, embed_item), axis=1)
        )

    def fit(
        self,
        train_data,
        verbose=1,
        shuffle=True,
        eval_data=None,
        metrics=None,
        **kwargs,
    ):
        if self.sparse_interaction is None:
            self.sparse_interaction = self._set_sparse_interaction()
        super().fit(train_data, verbose, shuffle, eval_data, metrics, **kwargs)

    def set_embeddings(self):
        bu, bi, puj, qi = self.sess.run(
            [self.bu_var, self.bi_var, self.puj_var, self.qi_var]
        )
        user_bias = np.ones([len(puj), 2], dtype=puj.dtype)
        user_bias[:, 0] = bu
        item_bias = np.ones([len(qi), 2], dtype=qi.dtype)
        item_bias[:, 1] = bi
        self.user_embed = np.hstack([puj, user_bias])
        self.item_embed = np.hstack([qi, item_bias])

    def _set_sparse_interaction(self):
        assert self.recent_num is None or (
            isinstance(self.recent_num, int) and self.recent_num > 0
        ), "`recent_num` must be None or positive int"
        indices = []
        values = []
        for u in range(self.n_users):
            items = self.user_consumed[u]
            u_data = items if self.recent_num is None else items[-self.recent_num:]
            indices.extend([u] * len(u_data))
            values.extend(u_data)
        indices = np.array(indices)[:, None]
        indices = np.concatenate([indices, np.zeros_like(indices)], axis=1)
        sparse_interaction = tf.SparseTensor(
            indices=indices, values=values, dense_shape=(self.n_users, self.n_items)
        )
        return sparse_interaction

    def rebuild_model(self, path, model_name, full_assign=False):
        self.sparse_interaction = self._set_sparse_interaction()
        rebuild_tf_model(self, path, model_name, full_assign)
