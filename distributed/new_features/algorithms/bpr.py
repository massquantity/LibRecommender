"""

References: Steffen Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback"
            (https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf)

author: massquantity

"""
import time
import logging
from itertools import islice
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.initializers import (
    zeros as tf_zeros,
    truncated_normal as tf_truncated_normal
)
from .base import Base, TfMixin
from ..evaluate.evaluate import EvalMixin
from ..utils.tf_ops import reg_config
from ..utils.samplingNEW import PairwiseSampling
from ..utils.colorize import colorize
from ..utils.timing import time_block
from ..utils.initializers import truncated_normal
from ..utils.misc import shuffle_data
try:
    from ._bpr import bpr_update
except ImportError:
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=LOG_FORMAT)
    logging.warn("Cython version is not available")
    pass


class BPR(Base, TfMixin, EvalMixin):
    """
    BPR is only suitable for ranking task
    """
    def __init__(self, task="ranking", data_info=None, embed_size=16,
                 n_epochs=20, lr=0.01, reg=None, batch_size=256,
                 num_neg=1, use_tf=True, seed=42):

        Base.__init__(self, task, data_info)
        TfMixin.__init__(self)
        EvalMixin.__init__(self, task)

        self.task = task
        self.data_info = data_info
        self.embed_size = embed_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg_config(reg)
        self.batch_size = batch_size
        self.num_neg = num_neg
        self.n_users = data_info.n_users
        self.n_items = data_info.n_items
        self.default_prediction = 0.0
        self.seed = seed

        self.user_consumed = None
        self.item_bias = None
        self.user_embed = None
        self.item_embed = None

        if use_tf:
            self.sess = tf.Session()
            self._build_model_tf()
            self._build_train_ops()
        else:
            self._build_model()

    def _build_model(self):
        np.random.seed(self.seed)
        # last dimension is item bias, so for user all set to 0
        self.user_embed = truncated_normal(
            shape=(self.n_users, self.embed_size + 1), mean=0.0, scale=0.03)
        self.user_embed[:, self.embed_size] = 1.0
        self.item_embed = truncated_normal(
            shape=(self.n_items, self.embed_size + 1), mean=0.0, scale=0.03)

    def fit_cython(self, train_data, verbose=1, shuffle=True, num_threads=1,
                   eval_data=None, metrics=None):

        start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print(f"training start time: {colorize(start_time, 'magenta')}")
        self.user_consumed = train_data.user_consumed
        if not self.reg:
            self.reg = 0.0

        for epoch in range(1, self.n_epochs + 1):
            with time_block(f"Epoch {epoch}", verbose):
                bpr_update(train_data,
                           self.user_embed,
                           self.item_embed,
                           self.lr,
                           self.reg,
                           self.n_users,
                           self.n_items,
                           shuffle,
                           num_threads,
                           self.seed)

            if verbose > 1:
                self.print_metrics(eval_data=eval_data, metrics=metrics)
                print("="*30)

    def fit_normal(self, train_data, verbose=1, shuffle=True):
        start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print(f"training start time: {colorize(start_time, 'magenta')}")
        self.user_consumed = train_data.user_consumed
        if not self.reg:
            self.reg = 0.0

        self._check_has_sampled(train_data, verbose)
        data_generator = PairwiseSampling(train_data,
                                          self.data_info,
                                          self.num_neg,
                                          self.batch_size)

        for epoch in range(1, self.n_epochs + 1):
            with time_block(f"Epoch {epoch}", verbose):
                for user, item_pos, item_neg in data_generator(shuffle=shuffle):
                    bias_item_pos = self.item_bias[item_pos]
                    bias_item_neg = self.item_bias[item_neg]
                    embed_user = self.user_embed[user]
                    embed_item_pos = self.item_embed[item_pos]
                    embed_item_neg = self.item_embed[item_neg]

                    item_diff = (bias_item_pos - bias_item_neg) + np.sum(
                        np.multiply(
                            embed_user, (embed_item_pos - embed_item_neg)
                        ), axis=1, keepdims=True
                    )
                    log_sigmoid_grad = 1.0 / (1.0 + np.exp(item_diff))
            #        log_sigmoid_grad = log_sigmoid_grad[None, :]

                    self.item_bias[item_pos] += self.lr * (
                            log_sigmoid_grad -
                            self.reg * self.item_bias[item_pos]
                    )
                    self.item_bias[item_neg] += self.lr * (
                            -log_sigmoid_grad -
                            self.reg * self.item_bias[item_neg]
                    )
                    self.user_embed[user] += self.lr * (
                        log_sigmoid_grad * (embed_item_pos - embed_item_neg)
                        - self.reg * self.user_embed[user]
                    )
                    self.item_embed[item_pos] += self.lr * (
                        log_sigmoid_grad * embed_user
                        - self.reg * self.item_embed[item_pos]
                    )
                    self.item_embed[item_neg] += self.lr * (
                        log_sigmoid_grad * (-embed_user)
                        - self.reg * self.item_embed[item_neg]
                    )

    def _build_model_tf(self):
        self.user_indices = tf.placeholder(tf.int32, shape=[None])
        self.item_indices_pos = tf.placeholder(tf.int32, shape=[None])
        self.item_indices_neg = tf.placeholder(tf.int32, shape=[None])

        self.item_bias_var = tf.get_variable(name="item_bias_var",
                                             shape=[self.n_items],
                                             initializer=tf_zeros,
                                             regularizer=self.reg)
        self.user_embed_var = tf.get_variable(name="user_embed_var",
                                              shape=[self.n_users,
                                                     self.embed_size],
                                              initializer=tf_truncated_normal(
                                                  0.0, 0.03),
                                              regularizer=self.reg)
        self.item_embed_var = tf.get_variable(name="item_embed_var",
                                              shape=[self.n_items,
                                                     self.embed_size],
                                              initializer=tf_truncated_normal(
                                                  0.0, 0.03),
                                              regularizer=self.reg)

        bias_item_pos = tf.nn.embedding_lookup(
            self.item_bias_var, self.item_indices_pos)
        bias_item_neg = tf.nn.embedding_lookup(
            self.item_bias_var, self.item_indices_neg)
        embed_user = tf.nn.embedding_lookup(
            self.user_embed_var, self.user_indices)
        embed_item_pos = tf.nn.embedding_lookup(
            self.item_embed_var, self.item_indices_pos)
        embed_item_neg = tf.nn.embedding_lookup(
            self.item_embed_var, self.item_indices_neg)

        item_diff = tf.subtract(bias_item_pos, bias_item_neg) + tf.reduce_sum(
            tf.multiply(
                embed_user,
                tf.subtract(embed_item_pos, embed_item_neg)
            ), axis=1
        )
        self.log_sigmoid = tf.log_sigmoid(item_diff)

    def _build_train_ops(self):
        self.loss = -self.log_sigmoid
        if self.reg is not None:
            reg_keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            total_loss = self.loss + tf.add_n(reg_keys)
        else:
            total_loss = self.loss

#        optimizer = tf.train.FtrlOptimizer(learning_rate=self.lr, l1_regularization_strength=1e-3)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.training_op = optimizer.minimize(total_loss)
        self.sess.run(tf.global_variables_initializer())

    def fit(self, train_data, verbose=1, shuffle=True,
            eval_data=None, metrics=None):

        start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print(f"training start time: {colorize(start_time, 'magenta')}")
        self.user_consumed = train_data.user_consumed

        self._check_has_sampled(train_data, verbose)
        data_generator = PairwiseSampling(train_data,
                                          self.data_info,
                                          self.num_neg,
                                          self.batch_size)

        for epoch in range(1, self.n_epochs + 1):
            with time_block(f"Epoch {epoch}", verbose):
                for user, item_pos, item_neg in data_generator(shuffle=shuffle):
                    self.sess.run(self.training_op,
                                  feed_dict={self.user_indices: user,
                                             self.item_indices_pos: item_pos,
                                             self.item_indices_neg: item_neg})

            if verbose > 1:
                # set up parameters for evaluate
                self._set_latent_factors()
                self.print_metrics(eval_data=eval_data, metrics=metrics)
                print("="*30)

        self._set_latent_factors()  # for prediction and recommend

    def predict(self, user, item):
        user = np.asarray(
            [user]) if isinstance(user, int) else np.asarray(user)
        item = np.asarray(
            [item]) if isinstance(item, int) else np.asarray(item)

        unknown_num, unknown_index, user, item = self._check_unknown(
            user, item)
        preds = np.sum(
            np.multiply(self.user_embed[user],
                        self.item_embed[item]),
            axis=1)
        preds = 1 / (1 + np.exp(-preds))

        if unknown_num > 0:
            preds[unknown_index] = self.default_prediction

        return preds[0] if len(user) == 1 else preds

    def recommend_user(self, user, n_rec, **kwargs):
        user = self._check_unknown_user(user)
        if not user:
            return   # popular ?

        consumed = self.user_consumed[user]
        count = n_rec + len(consumed)
        recos = self.user_embed[user] @ self.item_embed.T
        recos = 1 / (1 + np.exp(-recos))

        ids = np.argpartition(recos, -count)[-count:]
        rank = sorted(zip(ids, recos[ids]), key=lambda x: -x[1])
        return list(
            islice(
                (rec for rec in rank if rec[0] not in consumed), n_rec
            )
        )
























