"""

References: Yehuda Koren "Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model"
            (https://dl.acm.org/citation.cfm?id=1401944)

author: massquantity

"""
import os

import numpy as np
from tensorflow.keras.initializers import (
    zeros as tf_zeros,
    truncated_normal as tf_truncated_normal,
)

from ..bases import EmbedBase, TfMixin
from ..data.sequence import sparse_tensor_interaction
from ..tfops import modify_variable_names, reg_config, tf
from ..training import TensorFlowTrainer
from ..utils.save_load import load_params


class SVDpp(EmbedBase, TfMixin):
    user_variables = ["bu_var", "pu_var", "yj_var"]
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
        recent_num=None,
        random_sample_rate=None,
        lower_upper_bound=None,
        tf_sess_config=None,
        with_training=True,
    ):
        EmbedBase.__init__(self, task, data_info, embed_size, lower_upper_bound)
        TfMixin.__init__(self, data_info, tf_sess_config)

        self.all_args = locals()
        self.loss_type = loss_type
        self.n_epochs = n_epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.reg = reg_config(reg)
        self.batch_size = batch_size
        self.num_neg = num_neg
        self.seed = seed
        self.k = k
        self.eval_batch_size = eval_batch_size
        self.eval_user_num = eval_user_num
        self.recent_num = recent_num
        self.random_sample_rate = random_sample_rate
        self.trainer = None
        self.with_training = with_training

    def _build_model(self, sparse_implicit_interaction):
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
            initializer=tf_truncated_normal(0.0, 0.03),
            regularizer=self.reg,
        )
        self.qi_var = tf.get_variable(
            name="qi_var",
            shape=[self.n_items, self.embed_size],
            initializer=tf_truncated_normal(0.0, 0.03),
            regularizer=self.reg,
        )

        yj_var = tf.get_variable(
            name="yj_var",
            shape=[self.n_items, self.embed_size],
            initializer=tf_truncated_normal(0.0, 0.03),
            regularizer=self.reg,
        )
        uj = tf.nn.safe_embedding_lookup_sparse(
            yj_var,
            sparse_implicit_interaction,
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
        self.show_start_time()
        # check_has_sampled(train_data, verbose)
        if self.with_training:
            sparse_implicit_interaction = sparse_tensor_interaction(
                data=train_data,
                random_sample_rate=self.random_sample_rate,
                recent_num=self.recent_num,
            )
            self._build_model(sparse_implicit_interaction)
            self.trainer = TensorFlowTrainer(
                self,
                self.task,
                self.loss_type,
                self.n_epochs,
                self.lr,
                self.lr_decay,
                self.batch_size,
                self.num_neg,
                self.k,
                self.eval_batch_size,
                self.eval_user_num,
            )
            self.with_training = False
        self.trainer.run(train_data, verbose, shuffle, eval_data, metrics)
        self.set_embeddings()
        self.assign_embedding_oov()

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

    def rebuild_model(self, path, model_name, full_assign=False, train_data=None):
        if train_data is None:
            raise ValueError(
                "SVDpp model must provide train_data when rebuilding graph"
            )

        sparse_implicit_interaction = sparse_tensor_interaction(
            data=train_data,
            recent_num=self.recent_num,
            random_sample_rate=self.random_sample_rate,
        )
        self._build_model(sparse_implicit_interaction)

        hparams = load_params(SVDpp, path, self.data_info, model_name)
        self.trainer = TensorFlowTrainer(
            self,
            hparams["task"],
            hparams["loss_type"],
            hparams["n_epochs"],
            hparams["lr"],
            hparams["lr_decay"],
            hparams["batch_size"],
            hparams["num_neg"],
            hparams["k"],
            hparams["eval_batch_size"],
            hparams["eval_user_num"],
        )
        # self.trainer._build_train_ops()
        self.with_training = False

        variable_path = os.path.join(path, f"{model_name}_tf_variables.npz")
        variables = np.load(variable_path)
        variables = dict(variables.items())
        (
            user_variables,
            item_variables,
            _,
            _,
            manual_variables,
        ) = modify_variable_names(self, trainable=True)

        update_ops = []
        for v in tf.trainable_variables():
            if user_variables is not None and v.name in user_variables:
                # no need to remove oov values
                old_var = variables[v.name]
                user_op = tf.IndexedSlices(old_var, tf.range(len(old_var)))
                update_ops.append(v.scatter_update(user_op))

            if item_variables is not None and v.name in item_variables:
                old_var = variables[v.name]
                item_op = tf.IndexedSlices(old_var, tf.range(len(old_var)))
                update_ops.append(v.scatter_update(item_op))

        if full_assign:
            (
                optimizer_user_variables,
                optimizer_item_variables,
                _,
                _,
                _,
            ) = modify_variable_names(self, trainable=False)

            other_variables = [
                v for v in tf.global_variables() if v.name not in manual_variables
            ]
            for v in other_variables:
                if (
                    optimizer_user_variables is not None
                    and v.name in optimizer_user_variables
                ):
                    old_var = variables[v.name]
                    user_op = tf.IndexedSlices(old_var, tf.range(len(old_var)))
                    update_ops.append(v.scatter_update(user_op))
                elif (
                    optimizer_item_variables is not None
                    and v.name in optimizer_item_variables
                ):
                    old_var = variables[v.name]
                    item_op = tf.IndexedSlices(old_var, tf.range(len(old_var)))
                    update_ops.append(v.scatter_update(item_op))
                else:
                    old_var = variables[v.name]
                    update_ops.append(v.assign(old_var))

        self.sess.run(update_ops)
