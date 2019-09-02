import time, os, sys
from operator import itemgetter
import numpy as np
from ..utils.similarities import *
from ..utils.baseline_estimates import baseline_als, baseline_sgd
from .Base import BasePure
try:
    from ..utils.similarities_cy import cosine_cy, pearson_cy
except ImportError:
    pass
import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(format=LOG_FORMAT)
logging.warning("User KNN method requires huge memory for constructing similarity matrix. \n"
                "\tFor large num of users or items, consider using sklearn sim_option, "
                "which provides sparse similarity matrix. \n")


class userKNN(BasePure):
    def __init__(self, sim_option="pearson", k=50, min_support=1, baseline=True, task="rating", neg_sampling=False):
        self.k = k
        self.min_support = min_support
        self.baseline = baseline
        self.task = task
        self.neg_sampling = neg_sampling
        if sim_option == "cosine":
            self.sim_option = cosine_cy  # cosine_sim
        elif sim_option == "pearson":
            self.sim_option = pearson_cy  # pearson_cy
        elif sim_option == "sklearn":
          self.sim_option = sk_sim
        else:
            raise ValueError("sim_option %s not allowed" % sim_option)
        super(userKNN, self).__init__()

    def fit(self, dataset, verbose=1, **kwargs):
        self.dataset = dataset
        self.global_mean = dataset.global_mean
        self.train_user = dataset.train_user
        self.train_item = dataset.train_item
        if dataset.lower_upper_bound is not None:
            self.lower_bound = dataset.lower_upper_bound[0]
            self.upper_bound = dataset.lower_upper_bound[1]
        else:
            self.lower_bound = None
            self.upper_bound = None


        t0 = time.time()
        if self.sim_option == sk_sim:
            self.sim = self.sim_option(self.train_item, dataset.n_users, dataset.n_items,
                                       min_support=self.min_support, sparse=True)
        else:
            item_user_list = {k: list(v.items()) for k, v in dataset.train_item.items()}
            self.sim = self.sim_option(dataset.n_users, item_user_list, min_support=self.min_support)
            self.sim = np.array(self.sim)

        #    n = len(self.train_user)
        #    ids = list(self.train_user.keys())
        #    self.sim = get_sim(self.train_user, self.sim_option, n, ids, min_support=self.min_support)
        print(self.sim)
    #    print(self.sim[(self.sim != 0) & (self.sim != 1)])
        print("sim time: {:.4f}, sim shape: {}".format(time.time() - t0, self.sim.shape))
        if issparse(self.sim):
            print("sim num_elements: {}".format(self.sim.getnnz()))
        if self.baseline and self.task == "rating":
            self.bu, self.bi = baseline_als(dataset)

        if verbose > 0:
            print("training_time: {:.2f}".format(time.time() - t0))
            metrics = kwargs.get("metrics", self.metrics)
            if hasattr(self, "sess"):
                self.print_metrics_tf(dataset, 1, **metrics)
            else:
                self.print_metrics(dataset, 1, **metrics)
            print()

        return self

    def predict(self, u, i):
        if self.sim_option == sk_sim:
            u_nonzero_neighbors = set(self.sim.rows[u])
        else:
            u_nonzero_neighbors = set(np.where(self.sim[u] != 0.0)[0])
        try:
            neighbors = [(v, self.sim[u, v], r) for (v, r) in self.train_item[i].items()
                         if v in u_nonzero_neighbors and u != v]
            k_neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)[:self.k]
            if self.baseline and self.task == "rating":
                bui = self.global_mean + self.bu[u] + self.bi[i]
        except IndexError:
            return self.global_mean if self.task == "rating" else 0.0
        if len(neighbors) == 0:
            return self.global_mean if self.task == "rating" else 0.0

        if self.task == "rating":
            sim_ratings = 0
            sim_sums = 0
            for (v, sim, r) in k_neighbors:
                if sim > 0 and self.baseline:
                    bvi = self.global_mean + self.bu[v] + self.bi[i]
                    sim_ratings += sim * (r - bvi)
                    sim_sums += sim
                elif sim > 0:
                    sim_ratings += sim * r
                    sim_sums += sim
            try:
                if self.baseline:
                    pred = bui + sim_ratings / sim_sums
                else:
                    pred = sim_ratings / sim_sums

                if self.lower_bound is not None and self.upper_bound is not None:
                    pred = np.clip(pred, self.lower_bound, self.upper_bound)
                return pred
            except ZeroDivisionError:
                print("user %d sim user is zero" % u)
                return self.global_mean

        elif self.task == "ranking":
            sim_sums = 0
            for (_, sim, r) in k_neighbors:
                if sim > 0:
                    sim_sums += sim
            return sim_sums

    def recommend_user(self, u, n_rec, like_score=4.0, random_rec=False):
        if self.sim_option == sk_sim and len(self.sim.rows[u]) <= 1:  # no neighbors, just herself
            return -1
        elif self.sim_option != sk_sim and len(np.where(self.sim[u] != 0.0)[0]) <= 1:
            return -1

        rank = set()
        if self.sim_option == sk_sim:
            indices = np.argsort(self.sim[u].data[0])[::-1][1: self.k + 1]
            k_neighbors = np.array(self.sim.rows[u])[indices]
        else:
            k_neighbors = np.argsort(self.sim[u])[::-1][1: self.k + 1]

        for n in k_neighbors:
            n_items = np.array(list(self.train_user[n].items()))
            if self.task == "rating":
                n_items = [(i, r) for i, r in n_items if r >= like_score]

            for j, _ in n_items:
                if j in self.train_user[u]:
                    continue
                pred = self.predict(u, j)
                rank.add((int(j), pred))
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









