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
    from . import ALS_cy, ALS_rating_cy
except ImportError:
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=LOG_FORMAT)
    logging.warn("Cython version is not available")
    pass


class Als(BasePure):
    def __init__(self, n_factors=100, n_epochs=20, reg=5.0, task="rating", seed=42, alpha=10, cg_steps=3):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.reg = reg
        self.task = task
        self.seed = seed
        self.alpha = alpha
        self.cg_steps = cg_steps
        super(Als, self).__init__()



class ALS_rating:
    def __init__(self, n_factors=100, n_epochs=20, reg=5.0, task="rating", seed=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.reg = reg
        self.task = task
        self.seed = seed

    def fit(self, dataset, verbose=1):
        np.random.seed(self.seed)
        self.dataset = dataset
        self.default_prediction = dataset.global_mean
        self.pu = truncated_normal(shape=(dataset.n_users, self.n_factors),
                                   mean=0.0, scale=0.05)
        self.qi = truncated_normal(shape=(dataset.n_items, self.n_factors),
                                   mean=0.0, scale=0.05)

        for epoch in range(1, self.n_epochs + 1):
            t0 = time.time()
            for u in dataset.train_user:
                u_items = np.array(list(dataset.train_user[u].keys()))
                u_labels = np.array(list(dataset.train_user[u].values()))
                yy_reg = self.qi[u_items].T.dot(self.qi[u_items]) + \
                         self.reg * np.eye(self.n_factors)
                r_y = self.qi[u_items].T.dot(u_labels)
                self.pu[u] = np.linalg.solve(yy_reg, r_y)

            for i in dataset.train_item:
                i_users = np.array(list(dataset.train_item[i].keys()))
                i_labels = np.array(list(dataset.train_item[i].values()))
                xx_reg = self.pu[i_users].T.dot(self.pu[i_users]) + \
                         self.reg * np.eye(self.n_factors)
                r_x = self.pu[i_users].T.dot(i_labels)
                self.qi[i] = np.linalg.solve(xx_reg, r_x)

            if verbose > 0 and epoch % 1 == 0 and self.task == "rating":
                print("Epoch {} time: {:.4f}".format(epoch, time.time() - t0))
                print("training rmse: ", rmse(self, dataset, "train"))
                print("test rmse: ", rmse(self, dataset, "test"))
            elif verbose > 0 and epoch % 1 == 0 and self.task == "ranking":
                print("Epoch {} time: {:.4f}".format(epoch, time.time() - t0))
                print("MAP@{}: {:.4f}".format(5, MAP_at_k(self, dataset, 5)))

        return self

    def predict(self, u, i):
        try:
            pred = np.dot(self.pu[u], self.qi[i])
            pred = np.clip(pred, 1, 5)
        except IndexError:
            pred = self.default_prediction
        return pred

    def recommend_user(self, u, n_rec, random_rec=False):
        unlabled_items = list(set(range(self.dataset.n_items)) - set(self.dataset.train_user[u]))
        if np.any(np.array(unlabled_items) > self.dataset.n_items):
            rank = [(j, self.predict(u, j)) for j in range(len(self.qi))
                    if j not in self.dataset.train_user[u]]
        else:
            pred = np.dot(self.pu[u], self.qi[unlabled_items].T)
            pred = np.clip(pred, 1, 5)
            rank = list(zip(unlabled_items, pred))

        if random_rec:
            item_pred_dict = {j: r for j, r in rank if r >= 4}
            item_list = list(item_pred_dict.keys())
            pred_list = list(item_pred_dict.values())
            p = [p / np.sum(pred_list) for p in pred_list]
            item_candidates = np.random.choice(item_list, n_rec, replace=False, p=p)
            reco = [(item, item_pred_dict[item]) for item in item_candidates]
            return reco
        else:
            rank.sort(key=itemgetter(1), reverse=True)
            return rank[:n_rec]


class ALS_ranking:
    def __init__(self, n_factors=100, n_epochs=20, reg=5.0, seed=42, alpha=10, cg_steps=3):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.reg = reg
        self.seed = seed
        self.alpha = alpha
        self.cg_steps = cg_steps
    '''
    @staticmethod
    def least_squares(dataset, X, Y, reg, n_factors, alpha=10, user=True):
        if user:
            data = dataset.train_user
            m_shape = dataset.n_items
        else:
            data = dataset.train_item
            m_shape = dataset.n_users

        YtY = Y.T.dot(Y) + reg * np.eye(n_factors)
        for s in data:
            t0 = time.time()
            Cui_indices = list(data[s].keys())
            labels = list(data[s].values())
            Cui_values = np.array(labels) * alpha
            Cui = csr_matrix((Cui_values, (Cui_indices, Cui_indices)), shape=[m_shape, m_shape])
            pui_indices = list(data[s].keys())
            pui = np.zeros(m_shape)
            pui[pui_indices] = 1.0
            print("1: ", time.time() - t0)
            t1 = time.time()
            A = YtY + np.dot(Y.T, Cui.dot(Y))
            print("2: ", time.time() - t1)

            t2 = time.time()
            C = Cui + sparse.eye(m_shape, format="csr")
            cp = C.dot(pui)
            b = np.dot(Y.T, cp)
            print("3: ", time.time() - t2)
            t3 = time.time()
            X[s] = np.linalg.solve(A, b)
            print("4: ", time.time() - t3)
        #    from scipy.sparse.linalg import cg, cgs, bicg
        #    X[s] = bicg(A, b)[0]
        '''

    @staticmethod
    def least_squares(dataset, X, Y, reg, n_factors, alpha=10, user=True):
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
    def least_squares_cg(dataset, X, Y, reg, n_factors, alpha=10, cg_steps=3, user=True):  # O(f^3) * m
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
                r += (confidence - (confidence - 1) * Y[item].dot(x)) * Y[item]

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
                #    print("sample {} converged in step {}!!!".format(s, it + 1))
                    break
                p = r + (rs_new / rs_old) * p
                rs_old = rs_new
            X[s] = x

    def fit(self, dataset, verbose=1, use_cg=True):
        np.random.seed(self.seed)
        self.dataset = dataset
        self.default_prediction = dataset.global_mean
        self.X = truncated_normal(shape=(dataset.n_users, self.n_factors),
                                   mean=0.0, scale=0.05).astype(np.float32)
        self.Y = truncated_normal(shape=(dataset.n_items, self.n_factors),
                                   mean=0.0, scale=0.05).astype(np.float32)

    #    t0 = time.time()
    #    confidence_data = dok_matrix((dataset.n_users, dataset.n_items), dtype=np.float32)
    #    for u, i, l in zip(dataset.train_user_indices, dataset.train_item_indices, dataset.train_labels):
    #        confidence_data[u, i] = self.alpha * l + 1
    #    confidence_data = confidence_data.tocsr()
    #    print("constrtct time: ", time.time() - t0)

        if use_cg:
            method = functools.partial(ALS_ranking.least_squares_cg, cg_steps=self.cg_steps)
        #    method = functools.partial(ALS_cy.least_squares_cg, cg_steps=self.cg_steps)
        else:

            method = ALS_ranking.least_squares
        #    method = ALS_cy.least_squares

        for epoch in range(1, self.n_epochs + 1):
            t0 = time.time()
            method(self.dataset, self.X, self.Y, reg=self.reg,
                    n_factors=self.n_factors, alpha=self.alpha, user=True)

            method(self.dataset, self.Y, self.X, reg=self.reg,
                    n_factors=self.n_factors, alpha=self.alpha, user=False)

        #    method(confidence_data, self.X, self.Y, reg=self.reg, user=True, num_threads=0)
        #    method(confidence_data, self.Y, self.X, reg=self.reg, user=False, num_threads=0)

            if verbose > 0:
                print("Epoch {} time: {:.4f}".format(epoch, time.time() - t0))
                t1 = time.time()
                print("MAP@{}: {:.4f}".format(10, MAP_at_k(self, dataset, 10, sample_user=None)))
                print("MAP time: {:.4f}".format(time.time() - t1))
                t2 = time.time()
                print("MAR@{}: {:.4f}".format(50, MAR_at_k(self, dataset, 50, sample_user=None)))
                print("MAR time: {:.4f}".format(time.time() - t2))
                t3 = time.time()
                print("NDCG@{}: {:.4f}".format(10, NDCG_at_k(self, dataset, 10, sample_user=None)))
                print("NDCG time: {:.4f}".format(time.time() - t3))
                print()
            #    print("training accuracy: ", accuracy(self, dataset, "train"))
            #    print("test accuracy: {:.4f}".format(accuracy(self, dataset, "test")))

        return self

    def predict(self, u, i):
        try:
            prob = 1 / (1 + np.exp(-np.dot(self.X[u], self.Y[i])))
            pred = 1.0 if prob >= 0.5 else 0.0
        except IndexError:
            pred = self.default_prediction
        return pred

    def recommend_user(self, u, n_rec):
        consumed = self.dataset.train_user[u]
        count = n_rec + len(consumed)
        preds = np.dot(self.X[u], self.Y.T)
        ids = np.argpartition(preds, -count)[-count:]
        rank = sorted(zip(ids, preds[ids]), key=lambda x: -x[1])
        return list(itertools.islice((rec for rec in rank if rec[0] not in consumed), n_rec))



