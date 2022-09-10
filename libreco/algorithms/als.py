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
import numbers
import os
import logging
from functools import partial

import numpy as np

from ..bases import EmbedBase
from ..evaluation import print_metrics
from ..utils.initializers import truncated_normal
from ..utils.misc import time_block
from ..utils.save_load import save_params
from ..utils.validate import check_has_sampled

try:
    from ._als import als_update
except ImportError:
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=LOG_FORMAT)
    logging.warn("Als cython version is not available")
    raise


class ALS(EmbedBase):
    def __init__(
        self,
        task,
        data_info=None,
        embed_size=16,
        n_epochs=20,
        reg=None,
        alpha=10,
        use_cg=True,
        n_threads=1,
        seed=42,
        k=10,
        eval_batch_size=8192,
        eval_user_num=None,
        lower_upper_bound=None,
        with_training=True,
    ):
        super().__init__(task, data_info, embed_size, lower_upper_bound)

        self.all_args = locals()
        self.n_epochs = n_epochs
        self.reg = reg
        self.alpha = alpha
        self.use_cg = use_cg
        self.n_threads = n_threads
        self.seed = seed
        self.k = k
        self.eval_batch_size = eval_batch_size
        self.eval_user_num = eval_user_num
        if with_training:
            self._build_model()

    def _build_model(self):
        np.random.seed(self.seed)
        self.user_embed = truncated_normal(
            shape=[self.n_users, self.embed_size], mean=0.0, scale=0.03
        )
        self.item_embed = truncated_normal(
            shape=[self.n_items, self.embed_size], mean=0.0, scale=0.03
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
        assert isinstance(self.reg, numbers.Real), "`reg` must be float"
        user_interaction = train_data.sparse_interaction  # sparse.csr_matrix
        item_interaction = user_interaction.T.tocsr()
        if self.task == "ranking":
            check_has_sampled(train_data, verbose)
            user_interaction.data = user_interaction.data * self.alpha + 1
            item_interaction.data = item_interaction.data * self.alpha + 1

        trainer = partial(als_update, task=self.task, use_cg=self.use_cg)
        for epoch in range(1, self.n_epochs + 1):
            with time_block(f"Epoch {epoch}", verbose):
                trainer(
                    interaction=user_interaction,
                    X=self.user_embed,
                    Y=self.item_embed,
                    reg=self.reg,
                    num_threads=self.n_threads,
                )
                trainer(
                    interaction=item_interaction,
                    X=self.item_embed,
                    Y=self.user_embed,
                    reg=self.reg,
                    num_threads=self.n_threads,
                )

            if verbose > 1:
                print_metrics(
                    model=self,
                    train_data=train_data,
                    eval_data=eval_data,
                    metrics=metrics,
                    eval_batch_size=self.eval_batch_size,
                    k=self.k,
                    sample_user_num=self.eval_user_num,
                    seed=self.seed,
                )
                print("=" * 30)
        self.set_embeddings()
        self.assign_embedding_oov()

    def save(self, path, model_name, **kwargs):
        if not os.path.isdir(path):
            print(f"file folder {path} doesn't exists, creating a new one...")
            os.makedirs(path)
        save_params(self, path, model_name)
        variable_path = os.path.join(path, model_name)
        np.savez_compressed(
            variable_path, user_embed=self.user_embed, item_embed=self.item_embed
        )

    def set_embeddings(self):
        pass

    def rebuild_model(self, path, model_name):
        variable_path = os.path.join(path, f"{model_name}.npz")
        variables = np.load(variable_path)
        # remove oov values
        old_var = variables["user_embed"][:-1]
        self.user_embed[: len(old_var)] = old_var
        old_var = variables["item_embed"][:-1]
        self.item_embed[: len(old_var)] = old_var


def least_squares(sparse_interaction, X, Y, reg, embed_size, num, mode):
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
            for i in range(indptr[m], indptr[m + 1]):
                factor = Y[indices[i]]
                confidence = data[i]
                # If confidence = 1, r_ui = 0 means no interaction.
                A += (confidence - 1) * np.outer(factor, factor)
                b += confidence * factor
            X[m] = np.linalg.solve(A, b)
    else:
        raise ValueError("mode must either be 'explicit' or 'implicit'")


# O(f^3) * m
def least_squares_cg(sparse_interaction, X, Y, reg, embed_size, num, mode, cg_steps=3):
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
                for i in range(indptr[m], indptr[m + 1]):
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
