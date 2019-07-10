from operator import itemgetter
import random
import numpy as np
from ..utils.similarities import *
from ..utils.baseline_estimates import baseline_als, baseline_sgd


class itemKNN:
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
        n = len(self.train_item)
        ids = list(self.train_item.keys())
        self.sim = get_sim(self.train_item, self.sim_option, n, ids, min_support=self.min_support)
        if self.baseline:
            self.bu, self.bi = baseline_als(dataset)
        return self

    def predict(self, u, i):
        if self.baseline:
            try:
                neighbors = [(j, self.sim[i, j], r) for (j, r) in self.train_user[u].items()]
                k_neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)[:self.k]
                bui = self.global_mean + self.bu[u] + self.bi[i]
            except IndexError:
                return self.default_prediction
            sim_ratings = 0
            sim_sums = 0
            for j, sim, r in k_neighbors:
                if sim > 0:
                    buj = self.global_mean + self.bu[u] + self.bi[j]
                    sim_ratings += sim * (r - buj)
                    sim_sums += sim

            try:
                pred = bui + sim_ratings / sim_sums
                pred = np.clip(pred, 1, 5)
                return pred
            except ZeroDivisionError:
                return self.default_prediction

        else:
            try:
                neighbors = [(self.sim[i, j], r) for (j, r) in self.train_user[u].items()]
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
                pred = np.clip(pred, 1, 5)
                return pred
            except ZeroDivisionError:
                return self.default_prediction

    def recommend_user(self, u, k, n_rec, random_rec=False):
        rank = []
        u_items = np.array(list(self.train_user[u].items()))
        u_items = [(i, r) for i, r in u_items if r > 3]
        for i, _ in u_items:
            neighbors = [(self.sim[i, j], j) for j in range(len(self.train_item))]
            k_neighbors = sorted(neighbors, key=itemgetter(0), reverse=True)[:k]
            for _, j in k_neighbors:
                if j in self.train_user[u]:
                    continue
                pred = self.predict(u, j)
                rank.append((j, pred))
        if random_rec:
            item_pred_dict = {j: pred for j, pred in rank if pred >= 4}
            item_list = list(item_pred_dict.keys())
            pred_list = list(item_pred_dict.values())
            p = [p / np.sum(pred_list) for p in pred_list]

            item_candidates = np.random.choice(item_list, n_rec, replace=False, p=p)
            reco = [(item, item_pred_dict[item]) for item in item_candidates]
            return reco
        else:
            rank.sort(key=itemgetter(1), reverse=True)
            return rank[:n_rec]



















