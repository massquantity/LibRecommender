#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""

References:
    [1] Haoming Li et al. Matrix Completion via Alternating Least Square(ALS)
        (https://stanford.edu/~rezab/classes/cme323/S15/notes/lec14.pdf)
    [2] Yifan Hu et al. Collaborative Filtering for Implicit Feedback Datasets
        (http://yifanhu.net/PUB/cf.pdf)
    [3] Gábor Takács et al. Applications of the Conjugate Gradient Method for Implicit Feedback Collaborative Filtering
        (http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.379.6473&rep=rep1&type=pdf)

author: massquantity

"""
import os
import logging
from itertools import islice
from functools import partial
import numpy as np
from .base import Base
from ..evaluation.evaluate import EvalMixin
from ..utils.misc import time_block, assign_oov_vector
from ..utils.initializers import truncated_normal
try:
    from ._als import als_update
except ImportError:
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=LOG_FORMAT)
    logging.warn("Als cython version is not available")
    pass


class ALS(Base, EvalMixin):
    user_variables_np = ["user_embed"]
    item_variables_np = ["item_embed"]

    def __init__(
            self,
            task,
            data_info=None,
            embed_size=16,
            n_epochs=20,
            reg=None,
            alpha=10,
            seed=42,
            lower_upper_bound=None
    ):
        Base.__init__(self, task, data_info, lower_upper_bound)
        EvalMixin.__init__(self, task, data_info)

        self.task = task
        self.data_info = data_info
        self.embed_size = embed_size
        self.n_epochs = n_epochs
        self.reg = reg
        self.alpha = alpha
        self.seed = seed
        self.n_users = data_info.n_users
        self.n_items = data_info.n_items
        self.user_consumed = data_info.user_consumed
        self.user_embed = None
        self.item_embed = None
        self.all_args = locals()

    def _build_model(self):
        np.random.seed(self.seed)
        self.user_embed = truncated_normal(
            shape=[self.n_users, self.embed_size], mean=0.0, scale=0.03)
        self.item_embed = truncated_normal(
            shape=[self.n_items, self.embed_size], mean=0.0, scale=0.03)

    def fit(self, train_data, verbose=1, shuffle=True, use_cg=True,
            n_threads=1, eval_data=None, metrics=None, **kwargs):
        self.show_start_time()
        if not self.model_built:
            self._build_model()

        user_interaction = train_data.sparse_interaction  # sparse.csr_matrix
        item_interaction = user_interaction.T.tocsr()
        if self.task == "ranking":
            self._check_has_sampled(train_data, verbose)
            user_interaction.data = user_interaction.data * self.alpha + 1
            item_interaction.data = item_interaction.data * self.alpha + 1
        trainer = self._choose_algo(use_cg)

        for epoch in range(1, self.n_epochs + 1):
            with time_block(f"Epoch {epoch}", verbose):
                trainer(interaction=user_interaction,
                        X=self.user_embed,
                        Y=self.item_embed,
                        reg=self.reg,
                        num_threads=n_threads)
                trainer(interaction=item_interaction,
                        X=self.item_embed,
                        Y=self.user_embed,
                        reg=self.reg,
                        num_threads=n_threads)

            if verbose > 1:
                self.print_metrics(eval_data=eval_data, metrics=metrics,
                                   **kwargs)
                print("="*30)
        assign_oov_vector(self)

    def _choose_algo(self, use_cg):
        trainer = partial(als_update, task=self.task, use_cg=use_cg)
        return trainer

    def predict(self, user, item, cold_start="average", inner_id=False):
        user, item = self.convert_id(user, item, inner_id)
        unknown_num, unknown_index, user, item = self._check_unknown(user, item)

        preds = np.sum(
            np.multiply(self.user_embed[user], self.item_embed[item]),
            axis=1
        )

        if self.task == "rating":
            preds = np.clip(preds, self.lower_bound, self.upper_bound)
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
        recos = self.user_embed[user_id] @ self.item_embed.T
        if self.task == "ranking":
            recos = 1 / (1 + np.exp(-recos))
        ids = np.argpartition(recos, -count)[-count:]
        rank = sorted(zip(ids, recos[ids]), key=lambda x: -x[1])
        recs_and_scores = islice(
            (rec if inner_id else (self.data_info.id2item[rec[0]], rec[1])
             for rec in rank if rec[0] not in consumed),
            n_rec
        )
        return list(recs_and_scores)

    def save(self, path, model_name, manual=True, inference_only=False):
        if not os.path.isdir(path):
            print(f"file folder {path} doesn't exists, creating a new one...")
            os.makedirs(path)
        self.save_params(path)
        variable_path = os.path.join(path, model_name)
        np.savez_compressed(variable_path,
                            user_embed=self.user_embed,
                            item_embed=self.item_embed)

    @classmethod
    def load(cls, path, model_name, data_info, manual=True):
        variable_path = os.path.join(path, f"{model_name}.npz")
        variables = np.load(variable_path)
        hparams = cls.load_params(path, data_info)
        model = cls(**hparams)
        model.user_embed = variables["user_embed"]
        model.item_embed = variables["item_embed"]
        return model

    def rebuild_graph(self, path, model_name, full_assign=False):
        self._build_model()
        variable_path = os.path.join(path, f"{model_name}.npz")
        variables = np.load(variable_path)
        # remove oov values
        old_var = variables["user_embed"][:-1]
        self.user_embed[:len(old_var)] = old_var
        old_var = variables["item_embed"][:-1]
        self.item_embed[:len(old_var)] = old_var


def _least_squares(sparse_interaction, X, Y, reg, embed_size, num, mode):
    indices = sparse_interaction.indices
    indptr = sparse_interaction.indptr
    data = sparse_interaction.data
    if mode == "explicit":
        for m in range(num):
            m_slice = slice(indptr[m], indptr[m + 1])
            interacted = Y[indices[m_slice]]
            labels = data[m_slice]
            A = interacted.T @ interacted + reg * np.eye(embed_size)
            b = interacted.T @ labels
            X[m] = np.linalg.solve(A, b)
    elif mode == "implicit":
        init_A = Y.T @ Y + reg * np.eye(embed_size, dtype=np.float32)
        for m in range(num):
            A = init_A.copy()
            b = np.zeros(embed_size, dtype=np.float32)
            for i in range(indptr[m], indptr[m+1]):
                factor = Y[indices[i]]
                confidence = data[i]
                # If confidence = 1, r_ui = 0 means no interaction.
                A += (confidence - 1) * np.outer(factor, factor)
                b += confidence * factor
            X[m] = np.linalg.solve(A, b)
    else:
        raise ValueError("mode must either be 'explicit' or 'implicit'")


# O(f^3) * m
def _least_squares_cg(sparse_interaction, X, Y, reg, embed_size, num,
                      mode, cg_steps=3):
    indices = sparse_interaction.indices
    indptr = sparse_interaction.indptr
    data = sparse_interaction.data
    if mode == "explicit":
        for m in range(num):
            m_slice = slice(indptr[m], indptr[m + 1])
            interacted = Y[indices[m_slice]]
            labels = data[m_slice]
            A = interacted.T @ interacted + reg * np.eye(embed_size)
            b = interacted.T @ labels
            X[m] = np.linalg.solve(A, b)
    elif mode == "implicit":
        init_A = Y.T @ Y + reg * np.eye(embed_size, dtype=np.float32)
        for m in range(num):
            x = X[m]
            r = -init_A @ x
            # compute r = b - Ax
            for i in range(indptr[m], indptr[m + 1]):
                y = Y[indices[i]]
                confidence = data[i]
                r += (confidence - (confidence - 1) * (y @ x)) * y

            p = r.copy()
            rs_old = r @ r
            if rs_old < 1e-10:
                continue

            for _ in range(cg_steps):
                Ap = init_A @ p
                for i in range(indptr[m], indptr[m+1]):
                    y = Y[indices[i]]
                    confidence = data[i]
                    Ap += (confidence - 1) * (y @ p) * y

                # standard CG update
                ak = rs_old / (p @ Ap)
                x += ak * p
                r -= ak * Ap
                rs_new = r @ r
                if rs_new < 1e-10:
                    break
                p = r + (rs_new / rs_old) * p
                rs_old = rs_new

            X[m] = x

    else:
        raise ValueError("mode must either be 'explicit' or 'implicit'")
