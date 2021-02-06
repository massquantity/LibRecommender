"""

References: Steffen Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback"
            (https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf)

author: massquantity

"""
import os
import logging
from itertools import islice
from functools import partial
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras.initializers import (
    zeros as tf_zeros,
    truncated_normal as tf_truncated_normal
)
from .base import Base, TfMixin
from ..evaluation.evaluate import EvalMixin
from ..utils.sampling import PairwiseSampling
from ..utils.misc import time_block, assign_oov_vector
from ..utils.initializers import truncated_normal
try:
    from ._bpr import bpr_update
except (ImportError, ModuleNotFoundError):
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=LOG_FORMAT)
    logging.warning("BPR cython version is not available")
    pass  # may use tf version, then raise error will fail
tf.disable_v2_behavior()


class BPR(Base, TfMixin, EvalMixin):
    """
    BPR is only suitable for ranking task
    """
    user_variables = ["user_embed_var"]
    item_variables = ["item_embed_var", "item_bias_var"]
    user_variables_np = ["user_embed"]
    item_variables_np = ["item_embed"]

    def __init__(
            self,
            task="ranking",
            data_info=None,
            embed_size=16,
            n_epochs=20,
            lr=0.01,
            reg=None,
            batch_size=256,
            num_neg=1,
            use_tf=True,
            seed=42
    ):

        Base.__init__(self, task, data_info)
        EvalMixin.__init__(self, task, data_info)
        if use_tf:
            TfMixin.__init__(self)

        self.task = task
        self.data_info = data_info
        self.embed_size = embed_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.num_neg = num_neg
        self.n_users = data_info.n_users
        self.n_items = data_info.n_items
        self.use_tf = use_tf
        self.seed = seed
        self.user_consumed = data_info.user_consumed
        self.user_embed = None
        self.item_embed = None
        self.all_args = locals()

    def _build_model(self):
        if not self.use_tf:
            self.model_built = True
            np.random.seed(self.seed)
            # last dimension is item bias, so for user all set to 1.0
            self.user_embed = truncated_normal(
                shape=(self.n_users, self.embed_size + 1), mean=0.0, scale=0.03)
            self.user_embed[:, self.embed_size] = 1.0
            self.item_embed = truncated_normal(
                shape=(self.n_items, self.embed_size + 1), mean=0.0, scale=0.03)
            self.item_embed[:, self.embed_size] = 0.0
        else:
            self._build_model_tf()

    def _build_model_tf(self):
        self.graph_built = True
        if isinstance(self.reg, float) and self.reg > 0.0:
            tf_reg = tf.keras.regularizers.l2(self.reg)
        else:
            tf_reg = None

        self.user_indices = tf.placeholder(tf.int32, shape=[None])
        self.item_indices_pos = tf.placeholder(tf.int32, shape=[None])
        self.item_indices_neg = tf.placeholder(tf.int32, shape=[None])

        self.item_bias_var = tf.get_variable(
            name="item_bias_var",
            shape=[self.n_items + 1],
            initializer=tf_zeros,
            regularizer=tf_reg
        )
        self.user_embed_var = tf.get_variable(
            name="user_embed_var",
            shape=[self.n_users + 1, self.embed_size],
            initializer=tf_truncated_normal(0.0, 0.03),
            regularizer=tf_reg
        )
        self.item_embed_var = tf.get_variable(
            name="item_embed_var",
            shape=[self.n_items + 1, self.embed_size],
            initializer=tf_truncated_normal(0.0, 0.03),
            regularizer=tf_reg
        )

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
        self.loss = -tf.reduce_mean(self.log_sigmoid)
        if self.reg is not None:
            reg_keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            total_loss = self.loss + tf.add_n(reg_keys)
        else:
            total_loss = self.loss

        optimizer = tf.train.AdamOptimizer(self.lr)
        self.training_op = optimizer.minimize(total_loss)
        self.sess.run(tf.global_variables_initializer())

    def fit(self, train_data, verbose=1, shuffle=True, num_threads=1,
            eval_data=None, metrics=None, optimizer="sgd", **kwargs):
        self.show_start_time()
        self._check_has_sampled(train_data, verbose)

        if not self.graph_built and not self.model_built:
            if self.use_tf:
                self._build_model_tf()
                self._build_train_ops()
            else:
                self._build_model()

        if self.use_tf:
            self._fit_tf(train_data, verbose=verbose, shuffle=shuffle,
                         eval_data=eval_data, metrics=metrics, **kwargs)
        else:
            self._fit_cython(train_data, verbose=verbose, shuffle=shuffle,
                             num_threads=num_threads, eval_data=eval_data,
                             metrics=metrics, optimizer=optimizer, **kwargs)

    def _fit_cython(self, train_data, verbose=1, shuffle=True, num_threads=1,
                    eval_data=None, metrics=None, optimizer="sgd", **kwargs):
        if optimizer == "sgd":
            trainer = partial(bpr_update)

        elif optimizer == "momentum":
            user_velocity = np.zeros_like(self.user_embed, dtype=np.float32)
            item_velocity = np.zeros_like(self.item_embed, dtype=np.float32)
            momentum = 0.9
            trainer = partial(bpr_update,
                              u_velocity=user_velocity,
                              i_velocity=item_velocity,
                              momentum=momentum)

        elif optimizer == "adam":
            # refer to the "Deep Learning" book,
            # which is called first and second moment
            user_1st_moment = np.zeros_like(self.user_embed, dtype=np.float32)
            item_1st_moment = np.zeros_like(self.item_embed, dtype=np.float32)
            user_2nd_moment = np.zeros_like(self.user_embed, dtype=np.float32)
            item_2nd_moment = np.zeros_like(self.item_embed, dtype=np.float32)
            rho1, rho2 = 0.9, 0.999
            trainer = partial(bpr_update,
                              u_1st_mom=user_1st_moment,
                              i_1st_mom=item_1st_moment,
                              u_2nd_mom=user_2nd_moment,
                              i_2nd_mom=item_2nd_moment,
                              rho1=rho1,
                              rho2=rho2)

        else:
            raise ValueError("optimizer must be one of these: "
                             "('sgd', 'momentum', 'adam')")

        for epoch in range(1, self.n_epochs + 1):
            with time_block(f"Epoch {epoch}", verbose):
                trainer(optimizer=optimizer,
                        train_data=train_data,
                        user_embed=self.user_embed,
                        item_embed=self.item_embed,
                        lr=self.lr,
                        reg=self.reg,
                        n_users=self.n_users,
                        n_items=self.n_items,
                        shuffle=shuffle,
                        num_threads=num_threads,
                        seed=self.seed,
                        epoch=epoch)

            if verbose > 1:
                self.print_metrics(eval_data=eval_data, metrics=metrics,
                                   **kwargs)
                print("="*30)
        assign_oov_vector(self)

    def _fit_tf(self, train_data, verbose=1, shuffle=True,
                eval_data=None, metrics=None, **kwargs):
        data_generator = PairwiseSampling(train_data,
                                          self.data_info,
                                          self.num_neg)

        for epoch in range(1, self.n_epochs + 1):
            with time_block(f"Epoch {epoch}", verbose):
                for user, item_pos, item_neg in data_generator(
                        shuffle=shuffle, batch_size=self.batch_size
                ):
                    self.sess.run(self.training_op,
                                  feed_dict={self.user_indices: user,
                                             self.item_indices_pos: item_pos,
                                             self.item_indices_neg: item_neg})

            if verbose > 1:
                # set up parameters for evaluation
                self._set_latent_factors()
                self.print_metrics(eval_data=eval_data, metrics=metrics,
                                   **kwargs)
                print("="*30)

        # for prediction and recommendation
        self.assign_oov()
        self._set_latent_factors()

    def predict(self, user, item, cold_start="average", inner_id=False):
        user, item = self.convert_id(user, item, inner_id)
        unknown_num, unknown_index, user, item = self._check_unknown(user, item)

        preds = np.sum(
            np.multiply(self.user_embed[user], self.item_embed[item]),
            axis=1
        )
        preds = 1 / (1 + np.exp(-preds))

        if unknown_num > 0 and cold_start == "popular":
            preds[unknown_index] = self.default_prediction
        return preds

    def recommend_user(self, user, n_rec, cold_start="average", inner_id=False):
        user_id = self._check_unknown_user(user, inner_id)
        if user_id is None:
            if cold_start == "average":
                user_id = self.n_users
            elif cold_start == "popular":
                return self.data_info.popular_items[:n_rec]
            else:
                raise ValueError(user)

        consumed = set(self.user_consumed[user_id])
        count = n_rec + len(consumed)
        recos = self.user_embed[user_id] @ self.item_embed.T
        recos = 1 / (1 + np.exp(-recos))

        ids = np.argpartition(recos, -count)[-count:]
        rank = sorted(zip(ids, recos[ids]), key=lambda x: -x[1])
        recs_and_scores = islice(
            (rec if inner_id else (self.data_info.id2item[rec[0]], rec[1])
             for rec in rank if rec[0] not in consumed),
            n_rec
        )
        return list(recs_and_scores)

    def _set_latent_factors(self):
        item_bias, user_embed, item_embed = self.sess.run(
            [self.item_bias_var, self.user_embed_var, self.item_embed_var]
        )

        # to be compatible with cython version,
        # bias is concatenated with embedding
        user_bias = np.ones([len(user_embed), 1], dtype=user_embed.dtype)
        item_bias = item_bias[:, None]
        self.user_embed = np.hstack([user_embed, user_bias])
        self.item_embed = np.hstack([item_embed, item_bias])

    def save(self, path, model_name, manual=True, inference_only=False):
        if not os.path.isdir(path):
            print(f"file folder {path} doesn't exists, creating a new one...")
            os.makedirs(path)
        self.save_params(path)
        if inference_only:
            variable_path = os.path.join(path, model_name)
            np.savez_compressed(variable_path,
                                user_embed=self.user_embed,
                                item_embed=self.item_embed)
        else:
            self.save_variables(path, model_name, inference_only=False)

    @classmethod
    def load(cls, path, model_name, data_info, manual=True):
        variable_path = os.path.join(path, f"{model_name}.npz")
        variables = np.load(variable_path)
        hparams = cls.load_params(path, data_info)
        model = cls(**hparams)
        model.user_embed = variables["user_embed"]
        model.item_embed = variables["item_embed"]
        return model
