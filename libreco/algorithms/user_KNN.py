import time, os, sys
import pickle
from operator import itemgetter
import numpy as np
# import faiss
from ..utils.similarities import *
from ..evaluate import rmse, accuracy, MAR_at_k, MAP_at_k, NDCG_at_k
try:
    from ..utils.similarities_cy import cosine_cy, cosine_cym
    use_cython = True
except ImportError:
    use_cython = False
    pass
from ..utils.baseline_estimates import baseline_als, baseline_sgd


class userKNN:
    def __init__(self, sim_option="pearson", k=50, min_support=1, baseline=True, task="rating"):
        self.k = k
        self.min_support = min_support
        self.baseline = baseline
        self.task = task
        if sim_option == "cosine":
            self.sim_option = cosine_sim
        elif sim_option == "msd":
            self.sim_option = msd_sim
        elif sim_option == "pearson":
            self.sim_option = pearson_sim
        else:
            raise ValueError("sim_option %s not allowed" % sim_option)

    def fit(self, dataset, verbose=1):
        self.global_mean = dataset.global_mean
        self.default_prediction = dataset.global_mean
        self.train_user = dataset.train_user
        self.train_item = dataset.train_item
        if dataset.lower_upper_bound is not None:
            self.lower_bound = dataset.lower_upper_bound[0]
            self.upper_bound = dataset.lower_upper_bound[1]
        else:
            self.lower_bound = None
            self.upper_bound = None

        n = len(self.train_user)
        ids = list(self.train_user.keys())
        item_user_list = {k: list(v.items()) for k, v in dataset.train_item.items()}
        t0 = time.time()
        if use_cython:
        #    self.sim = cosine_cy(dataset.n_users, item_user_list, min_support=self.min_support)
            self.sim = cosine_cym(dataset.n_users, item_user_list, min_support=self.min_support)

        else:
            self.sim = get_sim(self.train_user, self.sim_option, n, ids, min_support=self.min_support)
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

        if verbose > 0 and self.task == "rating":
            print("training_time: {:.2f}".format(time.time() - t0))
            t1 = time.time()
            print("test rmse: {:.4f}".format(rmse(self, dataset, "test")))
            print("rmse time: {:.4f}".format(time.time() - t1))

        elif verbose > 0 and self.task == "ranking":
            print("training_time: {:.2f}".format(time.time() - t0))
            t1 = time.time()
            print("test accuracy: {:.4f}".format(accuracy(self, dataset, "test")))
            print("accuracy time: {:.4f}".format(time.time() - t1))

            t2 = time.time()
            mean_average_precision_10 = MAP_at_k(self, dataset, 10, sample_user=1000)
            print("\t MAP@{}: {:.4f}".format(10, mean_average_precision_10))
            print("\t MAP@10 time: {:.4f}".format(time.time() - t2))

            t3 = time.time()
            mean_average_recall_50 = MAR_at_k(self, dataset, 50, sample_user=1000)
            print("\t MAR@{}: {:.4f}".format(50, mean_average_recall_50))
            print("\t MAR@50 time: {:.4f}".format(time.time() - t3))

            t4 = time.time()
            NDCG = NDCG_at_k(self, dataset, 10, sample_user=1000)
            print("\t NDCG@{}: {:.4f}".format(10, NDCG))
            print("\t NDCG@10 time: {:.4f}".format(time.time() - t4))
            print()

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
                if self.lower_bound and self.upper_bound:
                    pred = np.clip(pred, self.lower_bound, self.upper_bound)
                return pred
            except ZeroDivisionError:
                print("user %d sim item is zero" % u)
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
                if self.lower_bound is not None and self.upper_bound is not None:
                    pred = np.clip(pred, self.lower_bound, self.upper_bound)
                return pred
            except ZeroDivisionError:
                print("user %d sim item is zero" % u)
                return self.default_prediction

    def recommend_user(self, u, n_rec, like_score=4.0, random_rec=False):
        rank = set()
    #    neighbors = [(self.sim[u, v], v) for v in range(len(self.sim))]
    #    k_neighbors = sorted(neighbors, key=itemgetter(0), reverse=True)[1:k+1]  # exclude u
    #    for _, n in k_neighbors:

        k_neighbors = np.argsort(self.sim[u])[::-1][1: self.k + 1]
        for n in k_neighbors:
            n_items = np.array(list(self.train_user[n].items()))
            if self.task == "rating":
                n_items = [(i, r) for i, r in n_items if r >= like_score]
    #        elif self.task == "ranking":
    #            n_items = [(i, r) for i, r in n_items]

            for j, _ in n_items:
                if j in self.train_user[u]:
                    continue
                pred = self.predict(u, j)
                rank.add((j, pred))
        rank = list(rank)

        if random_rec:
            item_pred_dict = {j: pred for j, pred in rank if pred >= like_score}
            if len(item_pred_dict) == 0:
                print("not enough candidates, try raising the like_score")
                sys.exit(1)

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









