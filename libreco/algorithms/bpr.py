"""

References: Steffen Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback"
            (https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf)

author: massquantity

"""
import logging
from functools import partial

import numpy as np
from tensorflow.keras.initializers import (
    zeros as tf_zeros,
    truncated_normal as tf_truncated_normal,
)

from ..bases import EmbedBase, TfMixin
from ..evaluation import print_metrics
from ..tfops import reg_config, tf
from ..training import BPRTrainer
from ..utils.initializers import truncated_normal
from ..utils.misc import time_block
from ..utils.validate import check_has_sampled

try:
    from ._bpr import bpr_update
except (ImportError, ModuleNotFoundError):
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=LOG_FORMAT)
    logging.warning("BPR cython version is not available")


class BPR(EmbedBase, TfMixin):
    """
    BPR is only suitable for ranking task
    """

    user_variables = ["user_embed_var"]
    item_variables = ["item_embed_var", "item_bias_var"]

    def __init__(
        self,
        task="ranking",
        data_info=None,
        loss_type="bpr",
        embed_size=16,
        n_epochs=20,
        lr=0.01,
        lr_decay=False,
        reg=None,
        batch_size=256,
        num_neg=1,
        use_tf=True,
        seed=42,
        k=10,
        eval_batch_size=8192,
        eval_user_num=None,
        lower_upper_bound=None,
        tf_sess_config=None,
        optimizer="adam",
        num_threads=1,
        with_training=True,
    ):
        EmbedBase.__init__(self, task, data_info, embed_size)
        if use_tf:
            TfMixin.__init__(self, data_info, tf_sess_config)

        assert task == "ranking", "BPR is only suitable for ranking"
        assert loss_type == "bpr", "BPR should use bpr loss"
        self.all_args = locals()
        self.reg = reg_config(reg) if use_tf else reg
        self.n_epochs = n_epochs
        self.lr = lr
        self.use_tf = use_tf
        self.seed = seed
        self.k = k
        self.eval_batch_size = eval_batch_size
        self.eval_user_num = eval_user_num
        self.optimizer = optimizer
        self.num_threads = num_threads
        if with_training:
            self._build_model()
            if use_tf:
                self.tf_trainer = BPRTrainer(
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
        if self.use_tf:
            self._build_model_tf()
        else:
            self._build_model_cython()

    def _build_model_cython(self):
        np.random.seed(self.seed)
        # last dimension is item bias, so for user all set to 1.0
        self.user_embed = truncated_normal(
            shape=(self.n_users, self.embed_size + 1), mean=0.0, scale=0.03
        )
        self.user_embed[:, self.embed_size] = 1.0
        self.item_embed = truncated_normal(
            shape=(self.n_items, self.embed_size + 1), mean=0.0, scale=0.03
        )
        self.item_embed[:, self.embed_size] = 0.0

    def _build_model_tf(self):
        self.user_indices = tf.placeholder(tf.int32, shape=[None])
        self.item_indices_pos = tf.placeholder(tf.int32, shape=[None])
        self.item_indices_neg = tf.placeholder(tf.int32, shape=[None])

        self.user_embed_var = tf.get_variable(
            name="user_embed_var",
            shape=[self.n_users, self.embed_size],
            initializer=tf_truncated_normal(0.0, 0.03),
            regularizer=self.reg,
        )
        self.item_embed_var = tf.get_variable(
            name="item_embed_var",
            shape=[self.n_items, self.embed_size],
            initializer=tf_truncated_normal(0.0, 0.03),
            regularizer=self.reg,
        )
        self.item_bias_var = tf.get_variable(
            name="item_bias_var",
            shape=[self.n_items],
            initializer=tf_zeros,
            regularizer=self.reg,
        )

        embed_user = tf.nn.embedding_lookup(self.user_embed_var, self.user_indices)
        embed_item_pos = tf.nn.embedding_lookup(
            self.item_embed_var, self.item_indices_pos
        )
        embed_item_neg = tf.nn.embedding_lookup(
            self.item_embed_var, self.item_indices_neg
        )
        bias_item_pos = tf.nn.embedding_lookup(
            self.item_bias_var, self.item_indices_pos
        )
        bias_item_neg = tf.nn.embedding_lookup(
            self.item_bias_var, self.item_indices_neg
        )

        item_diff = tf.subtract(bias_item_pos, bias_item_neg) + tf.reduce_sum(
            tf.multiply(embed_user, tf.subtract(embed_item_pos, embed_item_neg)), axis=1
        )
        self.bpr_loss = tf.log_sigmoid(item_diff)

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
        check_has_sampled(train_data, verbose)
        if self.use_tf:
            self.tf_trainer.run(train_data, verbose, shuffle, eval_data, metrics)
            self.set_embeddings()
        else:
            self._fit_cython(
                train_data=train_data,
                verbose=verbose,
                shuffle=shuffle,
                eval_data=eval_data,
                metrics=metrics,
            )
        self.assign_embedding_oov()

    def _fit_cython(
        self,
        train_data,
        verbose=1,
        shuffle=True,
        eval_data=None,
        metrics=None,
    ):
        if self.optimizer == "sgd":
            trainer = partial(bpr_update)

        elif self.optimizer == "momentum":
            user_velocity = np.zeros_like(self.user_embed, dtype=np.float32)
            item_velocity = np.zeros_like(self.item_embed, dtype=np.float32)
            momentum = 0.9
            trainer = partial(
                bpr_update,
                u_velocity=user_velocity,
                i_velocity=item_velocity,
                momentum=momentum,
            )

        elif self.optimizer == "adam":
            # refer to the `Deep Learning` book,
            # which is called first and second moment
            user_1st_moment = np.zeros_like(self.user_embed, dtype=np.float32)
            item_1st_moment = np.zeros_like(self.item_embed, dtype=np.float32)
            user_2nd_moment = np.zeros_like(self.user_embed, dtype=np.float32)
            item_2nd_moment = np.zeros_like(self.item_embed, dtype=np.float32)
            rho1, rho2 = 0.9, 0.999
            trainer = partial(
                bpr_update,
                u_1st_mom=user_1st_moment,
                i_1st_mom=item_1st_moment,
                u_2nd_mom=user_2nd_moment,
                i_2nd_mom=item_2nd_moment,
                rho1=rho1,
                rho2=rho2,
            )

        else:
            raise ValueError(
                "optimizer must be one of these: ('sgd', 'momentum', 'adam')"
            )

        for epoch in range(1, self.n_epochs + 1):
            with time_block(f"Epoch {epoch}", verbose):
                trainer(
                    optimizer=self.optimizer,
                    train_data=train_data,
                    user_embed=self.user_embed,
                    item_embed=self.item_embed,
                    lr=self.lr,
                    reg=self.reg,
                    n_users=self.n_users,
                    n_items=self.n_items,
                    shuffle=shuffle,
                    num_threads=self.num_threads,
                    seed=self.seed,
                    epoch=epoch,
                )

            if verbose > 1:
                print_metrics(
                    model=self,
                    eval_data=eval_data,
                    metrics=metrics,
                    eval_batch_size=self.eval_batch_size,
                    k=self.k,
                    sample_user_num=self.eval_user_num,
                    seed=self.seed,
                )
                print("=" * 30)

    def set_embeddings(self):
        item_bias, user_embed, item_embed = self.sess.run(
            [self.item_bias_var, self.user_embed_var, self.item_embed_var]
        )
        # to be compatible with cython version, bias is concatenated with embedding
        user_bias = np.ones([len(user_embed), 1], dtype=user_embed.dtype)
        item_bias = item_bias[:, None]
        self.user_embed = np.hstack([user_embed, user_bias])
        self.item_embed = np.hstack([item_embed, item_bias])
