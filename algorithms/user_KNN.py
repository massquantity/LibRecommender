import time, os
import pickle
from operator import itemgetter
import numpy as np
import faiss
from ..utils.similarities import *
from ..utils.baseline_estimates import baseline_als, baseline_sgd


class userKNN_567:
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
        t0 = time.time()
        self.sim = get_sim(self.train_user, self.sim_option, n, ids, min_support=self.min_support)
    #    with open(os.path.join(os.path.abspath(os.path.pardir), "sim_matrix.pkl")) as f:
    #        pickle.dump(self.sim, f)
    #    self.sim = invert_sim(self.train_item, n, min_support=self.min_support)
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
                pred = np.clip(pred, 1, 5)
            #    pred = min(5, pred)
            #    pred = max(1, pred)
                return pred
            except ZeroDivisionError:
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
                pred = np.clip(pred, 1, 5)
                return pred
            except ZeroDivisionError:
                return self.default_prediction

    def topN(self, u, k, n_rec, random_rec=False):
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
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items

        self.m = np.zeros((dataset.n_users, dataset.n_items)).astype(np.float32)
        for u, i_ratings in dataset.train_user.items():
            for i, r in i_ratings.items():
                self.m[u, i] = r

        self.sim_model = faiss.IndexFlatL2(dataset.n_items)
        self.sim_model.add(self.m)
        return self

    def predict(self, u, i):
        i_users = np.array(list(set(self.train_item[i]) - set([u])))  # exclude u
        index_dict = dict(zip(np.arange(len(i_users)), i_users))
        sim_model = faiss.IndexFlatL2(self.n_items)
        try:
            sim_model.add(self.m[i_users])
        except IndexError:
            return self.default_prediction

        sims, neighbors_indices_map = sim_model.search(self.m[u].reshape(1, -1), self.k)
        sims = sims.flatten() / max(sims.flatten())  # normalize to [0, 1]
        neighbors_indices_map = neighbors_indices_map.flatten()
        neighbors_indices_map = neighbors_indices_map[neighbors_indices_map != -1]
        valid_indices = np.where(neighbors_indices_map)
        sims = sims[valid_indices]

        neighbors_indices = np.vectorize(index_dict.get)(neighbors_indices_map)
        neighbors_ratings = self.m[neighbors_indices, i]

        sim_ratings = 0
        sim_sums = 0
        for sim, r in zip(sims, neighbors_ratings):
            if sim > 0 and r != 0:
                sim_ratings += sim * r
                sim_sums += sim
        try:
            pred = sim_ratings / sim_sums
            pred = np.clip(pred, 1, 5)
            return pred
        except ZeroDivisionError:
            return self.default_prediction

    def evaluate(self, dataset, num):
        """
        user_indices = dataset.train_user_indices[:num]
        item_indices = dataset.train_item_indices[:num]
        ratings = dataset.train_labels[:num]
        pred = []
        for u, i in zip(user_indices, item_indices):
            p = self.predict(u, i)
            pred.append(p)
        score = np.sqrt(np.mean(np.power(pred - ratings, 2)))
        return score
        """

        user_indices = dataset.train_user_indices[:num]
        item_indices = dataset.train_item_indices[:num]
        ratings = dataset.train_labels[:num]
        t2 = time.time()
        sims_all, neighbors_indices_all = self.sim_model.search(self.m, self.n_users)  # add min_support
        print("faiss time: ", time.time() - t2, "sim shape: ", sims_all.shape)
        '''
        pred = []
        for user, item in zip(user_indices, item_indices):
            sims = sims_all[user]  # [1:]
            neighbors_indices = neighbors_indices_all[user]  #[1:]

            i_users = np.array(list(self.train_item[item]))
            sim_indices = np.where(np.in1d(neighbors_indices, i_users))[0]
        #    sim_indices = np.searchsorted(neighbors_indices, i_users)
            sims = sims[sim_indices]
            neighbors_indices = neighbors_indices[sim_indices]

            sim_indices = np.argsort(sims)[1: self.k + 1]
            if len(sim_indices) == 0:
                pred.append(self.default_prediction)
                continue
            sims = sims[sim_indices]
            sims = sims / max(sims)
            neighbors_indices = neighbors_indices[sim_indices]
            neighbors_ratings = self.m[neighbors_indices, item]

            p = np.sum(np.multiply(sims, neighbors_ratings)) / np.sum(sims)
            p = np.clip(p, 1, 5)
            pred.append(p)
        score = np.sqrt(np.mean(np.power(pred - ratings, 2)))
        return score
        '''

        pred = []
        for user, item in zip(user_indices, item_indices):
            sims = sims_all[user]  # [1:]
            neighbors_indices = neighbors_indices_all[user]  # [1:]

            i_users = np.array(list(self.train_item[item]))
            sim_indices = np.where(np.in1d(neighbors_indices, i_users))[0]
            if sims[sim_indices][0] == 0.0:
                sim_indices = sim_indices[1: self.k + 1]
            else:
                sim_indices = sim_indices[:self.k]
            #    sim_indices = np.searchsorted(neighbors_indices, i_users)
            if len(sim_indices) == 0:
                pred.append(self.default_prediction)
                continue
            sims = sims[sim_indices]
        #    sims = 1.0 - sims / max(sims)
        #    sims = 1 / (sims / len(sims))  # use L2 norm  Sklearn normalize
        #   sims_all, neighbors_indices_all = self.sim_model.search(np.zeros(self.m.shape[0]), self.n_users)

            neighbors_indices = neighbors_indices[sim_indices]
            for j, (sim, vser) in enumerate(zip(sims, neighbors_indices)):
            #    u_rating_index = np.where(self.m[user] != 0)[0]
            #    v_rating_index = np.where(self.m[vser] != 0)[0]
            #    num = len(np.intersect1d(u_rating_index, v_rating_index))
                num = len(self.train_user[user]) + len(self.train_user[vser])
                sims[j] = 1 / (sims[j] / num)

            neighbors_ratings = self.m[neighbors_indices, item]

            p = np.sum(np.multiply(sims, neighbors_ratings)) / np.sum(sims)
            p = np.clip(p, 1, 5)
            pred.append(p)
        score = np.sqrt(np.mean(np.power(pred - ratings, 2)))
        return score


    def topN(self, u, k, n_rec, random_rec=False):
        rank = set()
        sims, neighbors_indices = self.sim_model.search(self.m[u].reshape(1, -1), k)
        sims = sims.flatten()
        neighbors_indices = neighbors_indices.flatten()
        neighbors = list(zip(sims, neighbors_indices))
        k_neighbors = sorted(neighbors, key=itemgetter(0), reverse=True)[1:k+1]  # exclude u
        for _, n in k_neighbors:
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






