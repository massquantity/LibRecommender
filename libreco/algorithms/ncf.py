"""

Reference: Xiangnan He et al. "Neural Collaborative Filtering" (https://arxiv.org/pdf/1708.05031.pdf)

author: massquantity

"""
import time
from itertools import islice
import numpy as np
import tensorflow as tf2
from tensorflow.keras.initializers import (
    zeros as tf_zeros,
    truncated_normal as tf_truncated_normal
)
from .base import Base, TfMixin
from ..evaluate.evaluate import EvalMixin
from ..utils.tf_ops import (
    reg_config,
    dropout_config,
    dense_nn,
    lr_decay_config
)
from ..utils.sampling import NegativeSampling
from ..data.data_generator import DataGenPure
tf = tf2.compat.v1
tf.disable_v2_behavior()


class NCF(Base, TfMixin, EvalMixin):
    def __init__(
            self,
            task,
            data_info,
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
            seed=42,
            batch_sampling=False,
            lower_upper_bound=None,
            tf_sess_config=None
    ):
        Base.__init__(self, task, data_info, lower_upper_bound)
        TfMixin.__init__(self, tf_sess_config)
        EvalMixin.__init__(self, task)

        self.task = task
        self.data_info = data_info
        self.embed_size = embed_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.reg = reg_config(reg)
        self.batch_size = batch_size
        self.batch_sampling = batch_sampling
        self.num_neg = num_neg
        self.use_bn = use_bn
        self.dropout_rate = dropout_config(dropout_rate)
        self.hidden_units = list(map(int, hidden_units.split(",")))
        self.n_users = data_info.n_users
        self.n_items = data_info.n_items
        self.seed = seed
        self.user_consumed = data_info.user_consumed

    def _build_model(self):
        self.user_indices = tf.placeholder(tf.int32, shape=[None])
        self.item_indices = tf.placeholder(tf.int32, shape=[None])
        self.labels = tf.placeholder(tf.float32, shape=[None])
        self.is_training = tf.placeholder_with_default(False, shape=[])

        user_gmf = tf.get_variable(name="user_gmf",
                                   shape=[self.n_users, self.embed_size],
                                   initializer=tf_truncated_normal(0.0, 0.01),
                                   regularizer=self.reg)
        item_gmf = tf.get_variable(name="item_gmf",
                                   shape=[self.n_items, self.embed_size],
                                   initializer=tf_truncated_normal(0.0, 0.01),
                                   regularizer=self.reg)
        user_mlp = tf.get_variable(name="user_mlp",
                                   shape=[self.n_users, self.embed_size],
                                   initializer=tf_truncated_normal(0.0, 0.01),
                                   regularizer=self.reg)
        item_mlp = tf.get_variable(name="item_mlp",
                                   shape=[self.n_items, self.embed_size],
                                   initializer=tf_truncated_normal(0.0, 0.01),
                                   regularizer=self.reg)

        user_gmf_embed = tf.nn.embedding_lookup(user_gmf, self.user_indices)
        item_gmf_embed = tf.nn.embedding_lookup(item_gmf, self.item_indices)
        user_mlp_embed = tf.nn.embedding_lookup(user_mlp, self.user_indices)
        item_mlp_embed = tf.nn.embedding_lookup(item_mlp, self.item_indices)

        gmf_layer = tf.multiply(user_gmf_embed, item_gmf_embed)
        mlp_input = tf.concat([user_mlp_embed, item_mlp_embed], axis=1)
        mlp_layer = dense_nn(mlp_input,
                             self.hidden_units,
                             use_bn=self.use_bn,
                             dropout_rate=self.dropout_rate,
                             is_training=self.is_training)

        concat_layer = tf.concat([gmf_layer, mlp_layer], axis=1)
        self.output = tf.reshape(
            tf.layers.dense(inputs=concat_layer, units=1), [-1])

    def _build_train_ops(self, global_steps=None):
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

        optimizer = tf.train.AdamOptimizer(self.lr)
        optimizer_op = optimizer.minimize(total_loss, global_step=global_steps)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.training_op = tf.group([optimizer_op, update_ops])
        self.sess.run(tf.global_variables_initializer())

    def fit(self, train_data, verbose=1, shuffle=True,
            eval_data=None, metrics=None, **kwargs):
        self.show_start_time()
        if self.lr_decay:
            n_batches = int(len(train_data) / self.batch_size)
            self.lr, global_steps = lr_decay_config(self.lr, n_batches,
                                                    **kwargs)
        else:
            global_steps = None

        self._build_model()
        self._build_train_ops(global_steps)

        if self.task == "ranking" and self.batch_sampling:
            self._check_has_sampled(train_data, verbose)
            data_generator = NegativeSampling(train_data,
                                              self.data_info,
                                              self.num_neg,
                                              self.batch_size,
                                              batch_sampling=True)

        else:
            data_generator = DataGenPure(train_data)

        self.train_pure(data_generator, verbose, shuffle, eval_data, metrics)

    def predict(self, user, item):
        user = np.asarray(
            [user]) if isinstance(user, int) else np.asarray(user)
        item = np.asarray(
            [item]) if isinstance(item, int) else np.asarray(item)

        unknown_num, unknown_index, user, item = self._check_unknown(
            user, item)

        preds = self.sess.run(self.output, feed_dict={
            self.user_indices: user,
            self.item_indices: item,
            self.is_training: False
        })

        if self.task == "rating":
            preds = np.clip(preds, self.lower_bound, self.upper_bound)
        elif self.task == "ranking":
            preds = 1 / (1 + np.exp(-preds))

        if unknown_num > 0:
            preds[unknown_index] = self.default_prediction

        return preds[0] if len(user) == 1 else preds

    def recommend_user(self, user, n_rec, **kwargs):
        user = self._check_unknown_user(user)
        if not user:
            return   # popular ?

        user_indices = np.full(self.n_items, user)
        item_indices = np.arange(self.n_items)
        recos = self.sess.run(self.output, feed_dict={
            self.user_indices: user_indices,
            self.item_indices: item_indices,
            self.is_training: False
        })
        if self.task == "ranking":
            recos = 1 / (1 + np.exp(-recos))

        consumed = self.user_consumed[user]
        count = n_rec + len(consumed)
        ids = np.argpartition(recos, -count)[-count:]
        rank = sorted(zip(ids, recos[ids]), key=lambda x: -x[1])
        return list(
            islice(
                (rec for rec in rank if rec[0] not in consumed), n_rec
            )
        )




