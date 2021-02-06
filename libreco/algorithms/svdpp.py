"""

References: Yehuda Koren "Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model"
            (https://dl.acm.org/citation.cfm?id=1401944)

author: massquantity

"""
import os
from itertools import islice
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras.initializers import (
    zeros as tf_zeros,
    truncated_normal as tf_truncated_normal
)
from .base import Base, TfMixin
from ..evaluation.evaluate import EvalMixin
from ..utils.tf_ops import reg_config
from ..utils.sampling import NegativeSampling
from ..data.data_generator import DataGenPure
from ..utils.tf_ops import sparse_tensor_interaction, modify_variable_names
from ..utils.misc import assign_oov_vector
tf.disable_v2_behavior()


class SVDpp(Base, TfMixin, EvalMixin):
    user_variables = ["bu_var", "pu_var", "yj_var"]
    item_variables = ["bi_var", "qi_var"]
    user_variables_np = ["bu", "pu", "puj"]
    item_variables_np = ["bi", "qi"]

    def __init__(
            self,
            task,
            data_info,
            embed_size=16,
            n_epochs=20,
            lr=0.01,
            reg=None,
            batch_size=256,
            batch_sampling=False,
            num_neg=1,
            seed=42,
            lower_upper_bound=None,
            tf_sess_config=None
    ):

        Base.__init__(self, task, data_info, lower_upper_bound)
        TfMixin.__init__(self, tf_sess_config)
        EvalMixin.__init__(self, task, data_info)

        self.task = task
        self.data_info = data_info
        self.embed_size = embed_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg_config(reg)
        self.batch_size = batch_size
        self.batch_sampling = batch_sampling
        self.num_neg = num_neg
        self.n_users = data_info.n_users
        self.n_items = data_info.n_items
        self.seed = seed
        self.user_consumed = data_info.user_consumed
        self.bu = None
        self.bi = None
        self.pu = None
        self.qi = None
        self.yj = None
        self.all_args = locals()

    def _build_model(self, sparse_implicit_interaction):
        self.graph_built = True
        self.user_indices = tf.placeholder(tf.int32, shape=[None])
        self.item_indices = tf.placeholder(tf.int32, shape=[None])
        self.labels = tf.placeholder(tf.float32, shape=[None])

        self.bu_var = tf.get_variable(name="bu_var", shape=[self.n_users],
                                      initializer=tf_zeros,
                                      regularizer=self.reg)
        self.bi_var = tf.get_variable(name="bi_var", shape=[self.n_items],
                                      initializer=tf_zeros,
                                      regularizer=self.reg)
        self.pu_var = tf.get_variable(name="pu_var",
                                      shape=[self.n_users, self.embed_size],
                                      initializer=tf_truncated_normal(
                                          0.0, 0.03),
                                      regularizer=self.reg)
        self.qi_var = tf.get_variable(name="qi_var",
                                      shape=[self.n_items, self.embed_size],
                                      initializer=tf_truncated_normal(
                                          0.0, 0.03),
                                      regularizer=self.reg)

        yj_var = tf.get_variable(name="yj_var",
                                 shape=[self.n_items, self.embed_size],
                                 initializer=tf_truncated_normal(0.0, 0.03),
                                 regularizer=self.reg)

        uj = tf.nn.safe_embedding_lookup_sparse(
            yj_var, sparse_implicit_interaction, sparse_weights=None,
            combiner="sqrtn", default_id=None
        )   # unknown user will return 0-vector
        self.puj_var = self.pu_var + uj

        bias_user = tf.nn.embedding_lookup(self.bu_var, self.user_indices)
        bias_item = tf.nn.embedding_lookup(self.bi_var, self.item_indices)
        embed_user = tf.nn.embedding_lookup(self.puj_var, self.user_indices)
        embed_item = tf.nn.embedding_lookup(self.qi_var, self.item_indices)

        self.output = bias_user + bias_item + tf.reduce_sum(
            tf.multiply(embed_user, embed_item), axis=1)

    def _build_train_ops(self):
        if self.task == "rating":
            pred = self.output + self.global_mean
            self.loss = tf.losses.mean_squared_error(labels=self.labels,
                                                     predictions=pred)
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
        self.training_op = optimizer.minimize(total_loss)
        self.sess.run(tf.global_variables_initializer())

    def fit(self, train_data, verbose=1, shuffle=True, sample_rate=None,
            recent_num=None, eval_data=None, metrics=None, **kwargs):
        self.show_start_time()
        if not self.graph_built:
            sparse_implicit_interaction = sparse_tensor_interaction(
                train_data, random_sample_rate=sample_rate,
                recent_num=recent_num)

            self._build_model(sparse_implicit_interaction)
            self._build_train_ops()

        if self.task == "ranking" and self.batch_sampling:
            self._check_has_sampled(train_data, verbose)
            data_generator = NegativeSampling(train_data,
                                              self.data_info,
                                              self.num_neg,
                                              batch_sampling=True)
        else:
            data_generator = DataGenPure(train_data)

        self.train_pure(data_generator, verbose, shuffle, eval_data, metrics,
                        **kwargs)
        self._set_latent_factors()
        assign_oov_vector(self)

    def predict(self, user, item, cold_start="average", inner_id=False):
        user, item = self.convert_id(user, item, inner_id)
        unknown_num, unknown_index, user, item = self._check_unknown(user, item)

        preds = self.bu[user] + self.bi[item] + np.sum(
            np.multiply(self.puj[user], self.qi[item]), axis=1)

        if self.task == "rating":
            preds = np.clip(
                preds + self.global_mean, self.lower_bound, self.upper_bound)
        elif self.task == "ranking":
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
        recos = self.bu[user_id] + self.bi + self.puj[user_id] @ self.qi.T

        if self.task == "rating":
            recos += self.global_mean
        elif self.task == "ranking":
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
        self.bu, self.bi, self.pu, self.qi, self.puj = self.sess.run(
            [self.bu_var, self.bi_var, self.pu_var, self.qi_var, self.puj_var]
        )

    def save(self, path, model_name, manual=True, inference_only=False):
        if not os.path.isdir(path):
            print(f"file folder {path} doesn't exists, creating a new one...")
            os.makedirs(path)
        self.save_params(path)
        if inference_only:
            variable_path = os.path.join(path, model_name)
            np.savez_compressed(variable_path,
                                bu=self.bu,
                                bi=self.bi,
                                pu=self.pu,
                                qi=self.qi,
                                puj=self.puj)
        else:
            self.save_variables(path, model_name, inference_only=False)

    @classmethod
    def load(cls, path, model_name, data_info, manual=True):
        variable_path = os.path.join(path, f"{model_name}.npz")
        variables = np.load(variable_path)
        hparams = cls.load_params(path, data_info)
        model = cls(**hparams)
        model.bu = variables["bu"]
        model.bi = variables["bi"]
        model.pu = variables["pu"]
        model.qi = variables["qi"]
        model.puj = variables["puj"]
        return model

    def rebuild_graph(self, path, model_name, full_assign=False,
                      train_data=None):
        if train_data is None:
            raise ValueError("SVDpp model must provide train_data "
                             "when rebuilding graph")
        sparse_implicit_interaction = sparse_tensor_interaction(
            train_data, recent_num=10)
        self._build_model(sparse_implicit_interaction)
        self._build_train_ops()

        variable_path = os.path.join(path, f"{model_name}_variables.npz")
        variables = np.load(variable_path)
        variables = dict(variables.items())

        (
            user_variables,
            item_variables,
            sparse_variables,
            dense_variables,
            manual_variables
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
                optimizer_sparse_variables,
                optimizer_dense_variables,
                _
            ) = modify_variable_names(self, trainable=False)

            other_variables = [v for v in tf.global_variables()
                               if v.name not in manual_variables]
            for v in other_variables:
                if (optimizer_user_variables is not None
                        and v.name in optimizer_user_variables):
                    old_var = variables[v.name]
                    user_op = tf.IndexedSlices(old_var, tf.range(len(old_var)))
                    update_ops.append(v.scatter_update(user_op))

                elif (optimizer_item_variables is not None
                          and v.name in optimizer_item_variables):
                    old_var = variables[v.name]
                    item_op = tf.IndexedSlices(old_var, tf.range(len(old_var)))
                    update_ops.append(v.scatter_update(item_op))

                else:
                    old_var = variables[v.name]
                    update_ops.append(v.assign(old_var))

        self.sess.run(update_ops)
