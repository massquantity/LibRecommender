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
import time
from operator import itemgetter
import functools
import itertools
import logging
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from scipy import sparse
from ..evaluate import rmse, MAP_at_k, MAR_at_k, NDCG_at_k, accuracy
from ..utils.initializers import truncated_normal
from .Base import BasePure
try:
    from . import Als_cy
except ImportError:
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=LOG_FORMAT)
    logging.warn("Cython version is not available")
    pass


class Als(BasePure):
    def __init__(self, n_factors=100, n_epochs=20, reg=5.0, task="rating", seed=42,
                 alpha=10, neg_sampling=False):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.reg = reg
        self.task = task
        self.seed = seed
        self.alpha = alpha
        self.neg_sampling = neg_sampling
        super(Als, self).__init__()

    def fit(self, dataset, verbose=1, use_cg=False, cg_steps=3, use_cython=True, **kwargs):
        np.random.seed(self.seed)
        self.dataset = dataset
        self.global_mean = dataset.global_mean
        if dataset.lower_upper_bound is not None:
            self.lower_bound = dataset.lower_upper_bound[0]
            self.upper_bound = dataset.lower_upper_bound[1]
        else:
            self.lower_bound = None
            self.upper_bound = None

        self.X = truncated_normal(shape=(dataset.n_users, self.n_factors),
                                   mean=0.0, scale=0.05).astype(np.float32)
        self.Y = truncated_normal(shape=(dataset.n_items, self.n_factors),
                                   mean=0.0, scale=0.05).astype(np.float32)

        if self.task == "ranking":
            t0 = time.time()
            confidence_data = dok_matrix((dataset.n_users, dataset.n_items), dtype=np.float32)
            for u, i, l in zip(dataset.train_user_indices, dataset.train_item_indices, dataset.train_labels):
                confidence_data[u, i] = self.alpha * l + 1
            confidence_data = confidence_data.tocsr()
            print("sparse matrix constrtct time: ", time.time() - t0)

        if self.task == "rating":
            method = Als.least_squares
        elif self.task == "ranking" and use_cg:
            if use_cython:
                method = functools.partial(
                    Als_cy.least_squares_weighted_cg, cg_steps=cg_steps, data=confidence_data, num_threads=0)
            else:
                method = functools.partial(
                    Als.least_squares_weighted_cg, alpha=self.alpha, dataset=self.dataset, cg_steps=cg_steps)

        else:
            if use_cython:
                method = functools.partial(Als_cy.least_squares_weighted, data=confidence_data, num_threads=0)
            else:
                method = functools.partial(Als.least_squares_weighted, dataset=self.dataset, alpha=self.alpha)


        for epoch in range(1, self.n_epochs + 1):
            t0 = time.time()

            if self.task == "rating":
                method(self.dataset, self.X, self.Y, reg=self.reg, n_factors=self.n_factors, user=True)
                method(self.dataset, self.Y, self.X, reg=self.reg, n_factors=self.n_factors, user=False)

            elif self.task == "ranking":
                method(X=self.X, Y=self.Y, reg=self.reg, n_factors=self.n_factors, user=True)
                method(X=self.Y, Y=self.X, reg=self.reg, n_factors=self.n_factors, user=False)

            if verbose > 0:
                print("Epoch {}: training time: {:.4f}".format(epoch, time.time() - t0))
                metrics = kwargs.get("metrics", self.metrics)
                if hasattr(self, "sess"):
                    self.print_metrics_tf(dataset, epoch, **metrics)
                else:
                    self.print_metrics(dataset, epoch, **metrics)
                print()

        return self

    def predict(self, u, i):
        try:
            pred = np.dot(self.X[u], self.Y[i])
        except IndexError:
            return self.global_mean if self.task == "rating" else 0.0

        if self.task == "rating" and self.lower_bound is not None and self.upper_bound is not None:
            pred = np.clip(pred, self.lower_bound, self.upper_bound)

        return pred

    def recommend_user(self, u, n_rec):
        consumed = self.dataset.train_user[u]
        count = n_rec + len(consumed)
        preds = np.dot(self.X[u], self.Y.T)
        ids = np.argpartition(preds, -count)[-count:]
        rank = sorted(zip(ids, preds[ids]), key=lambda x: -x[1])
        return list(itertools.islice((rec for rec in rank if rec[0] not in consumed), n_rec))

    @staticmethod
    def least_squares(dataset, X, Y, reg, n_factors, user=True):
        if user:
            data = dataset.train_user
        else:
            data = dataset.train_item

        for s in data:
            consumed = np.array(list(data[s].keys()))
            labels = np.array(list(data[s].values()))
            A = Y[consumed].T.dot(Y[consumed]) + \
                     reg * np.eye(n_factors)
            b = Y[consumed].T.dot(labels)
            X[s] = np.linalg.solve(A, b)

    @staticmethod
    def least_squares_weighted(dataset, X, Y, reg, n_factors, alpha=10, user=True):
        if user:
            data = dataset.train_user
        else:
            data = dataset.train_item

        YtY = Y.T.dot(Y)
        for s in data:
            A = YtY + reg * np.eye(n_factors)
            b = np.zeros(n_factors)
            for i in data[s]:
                factor = Y[i]
                confidence = 1 + alpha * data[s][i]
                A += (confidence - 1) * np.outer(factor, factor)
                b += confidence * factor

            X[s] = np.linalg.solve(A, b)
        #    from scipy.sparse.linalg import cg, cgs, bicg
        #    X[s] = bicg(A, b)[0]

    @staticmethod
    def least_squares_weighted_cg(dataset, X, Y, reg, n_factors, alpha=10, cg_steps=3, user=True):  # O(f^3) * m
        if user:
            data = dataset.train_user
        else:
            data = dataset.train_item

        YtY = Y.T.dot(Y) + reg * np.eye(n_factors)
        for s in data:
            x = X[s]
            r = -YtY.dot(x)
            for item, label in data[s].items():
                confidence = 1 + alpha * label
                r += (confidence - (confidence - 1) * Y[item].dot(x)) * Y[item]  # b - Ax

            p = r.copy()
            rs_old = r.dot(r)
            if rs_old < 1e-10:
                continue

            for it in range(cg_steps):
                Ap = YtY.dot(p)
                for item, label in data[s].items():
                    confidence = 1 + alpha * label
                    Ap += (confidence - 1) * Y[item].dot(p) * Y[item]

                # standard CG update
                alpha = rs_old / p.dot(Ap)
                x += alpha * p
                r -= alpha * Ap
                rs_new = r.dot(r)
                if rs_new < 1e-10:
                    break
                p = r + (rs_new / rs_old) * p
                rs_old = rs_new
            X[s] = x



