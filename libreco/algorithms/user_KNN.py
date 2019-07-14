import time, os
import pickle
from operator import itemgetter
import numpy as np
# import faiss
from ..utils.similarities import *
from ..utils.similarities_cy import cosine_cy, cosine_cym
from ..utils.baseline_estimates import baseline_als, baseline_sgd


class userKNN:
    def __init__(self, sim_option="pearson", k=50, min_support=1, baseline=True):
        self.k = k
        self.min_support = min_support
        self.baseline = baseline
        if sim_option == "cosine":
            self.sim_option = cosine_sim
        elif sim_option == "msd":
            self.sim_option = msd_sim
        elif sim_option == "pearson":
            self.sim_option = pearson_sim
        else:
            raise ValueError("sim_option %s not allowed" % sim_option)

    def fit(self, dataset):
        self.global_mean = dataset.global_mean
        self.default_prediction = dataset.global_mean
        self.train_user = dataset.train_user
        self.train_item = dataset.train_item
        n = len(self.train_user)
        ids = list(self.train_user.keys())
        item_user_list = {k: list(v.items()) for k, v in dataset.train_item.items()}
        t0 = time.time()
    #    self.sim = cosine_cy(dataset.n_users, item_user_list, min_support=self.min_support)
        self.sim = cosine_cym(dataset.n_users, item_user_list, min_support=self.min_support)


    #    self.sim = get_sim(self.train_user, self.sim_option, n, ids, min_support=self.min_support)
    #    with open(os.path.join(os.path.abspath(os.path.pardir), "sim_matrix.pkl")) as f:
    #        pickle.dump(self.sim, f)
    #    self.sim = invert_sim(self.train_item, n, min_support=self.min_support)

    #    self.sim = sk_sim_cy(self.train_item, dataset.n_users, dataset.n_items,
    #                      min_support=self.min_support, sparse=True)
    #    self.sim = sk_sim(self.train_item, dataset.n_users, dataset.n_items,
    #                      min_support=self.min_support, sparse=True)
        print("sim time: ", time.time() - t0)
        if self.baseline:
            self.bu, self.bi = baseline_als(dataset)
        return self

    def predict(self, u, i):
        if self.baseline:
            try:
                neighbors = [(v, self.sim[u, v], r) for (v, r) in self.train_item[i].items()]
                k_neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)[:self.k]
                bui = self.global_mean + self.bu[u] + self.bi[i]
            except IndexError:
                return self.default_prediction
            sim_ratings = 0
            sim_sums = 0
            for v, sim, r in k_neighbors:
                if sim > 0:
                    bvi = self.global_mean + self.bu[v] + self.bi[i]
                    sim_ratings += sim * (r - bvi)
                    sim_sums += sim
            try:
                pred = bui + sim_ratings / sim_sums
                pred = np.clip(pred, 1.0, 5.0)
            #    pred = min(5, pred)
            #    pred = max(1, pred)
                return pred
            except ZeroDivisionError:
                print("zero")
                return self.default_prediction

        else:
            try:
                neighbors = [(self.sim[u, v], r) for (v, r) in self.train_item[i].items()]
                k_neighbors = sorted(neighbors, key=lambda x: x[0], reverse=True)[:self.k]
            except IndexError:
                return self.default_prediction

            sim_ratings = 0
            sim_sums = 0
            for (sim, r) in k_neighbors:
                if sim > 0:
                    sim_ratings += sim * r
                    sim_sums += sim
            try:
                pred = sim_ratings / sim_sums
                pred = np.clip(pred, 1.0, 5.0)
                return pred
            except ZeroDivisionError:
                print("zero")
                return self.default_prediction

    def recommend_user(self, u, k, n_rec, random_rec=False):
        rank = set()
    #    neighbors = [(self.sim[u, v], v) for v in range(len(self.sim))]
    #    k_neighbors = sorted(neighbors, key=itemgetter(0), reverse=True)[1:k+1]  # exclude u
    #    for _, n in k_neighbors:

        k_neighbors = np.argsort(self.sim[u])[::-1][1:k+1]
        for n in k_neighbors:
            n_items = np.array(list(self.train_user[n].items()))
            n_items = [(j, r) for j, r in n_items if r > 3]
            for j, _ in n_items:
                if j in self.train_user[u]:
                    continue
                pred = self.predict(u, j)
                rank.add((j, pred))
        rank = list(rank)

        if random_rec:
            item_pred_dict = {j: pred for j, pred in rank if pred >= 4}
            if len(item_pred_dict) == 0:
                return "not enough candidates"
            item_list = list(item_pred_dict.keys())
            pred_list = list(item_pred_dict.values())
            p = [p / np.sum(pred_list) for p in pred_list]
            if len(item_list) < n_rec:
                item_candidates = item_list
            else:
                item_candidates = np.random.choice(item_list, n_rec, replace=False, p=p)
            reco = [(item, item_pred_dict[item]) for item in item_candidates]
            return reco
        else:
            rank.sort(key=itemgetter(1), reverse=True)
            return rank[:n_rec]


