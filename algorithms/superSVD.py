import time
from operator import itemgetter
import numpy as np
from ..evaluate import rmse_svd
from ..utils.similarities import *
from ..utils.baseline_estimates import baseline_als, baseline_sgd


class superSVD:
    def __init__(self, n_factors=100, n_epochs=20, lr=0.01, reg=5.0,
                 batch_size=256, batch_training=True, k=50,
                 sim_option="pearson", min_support=1, seed=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.batch_training = batch_training
        self.seed = seed
        self.k = k
        self.min_support = min_support
        if sim_option == "cosine":
            self.sim_option = cosine_sim
        elif sim_option == "msd":
            self.sim_option = msd_sim
        elif sim_option == "pearson":
            self.sim_option = pearson_sim
        else:
            raise ValueError("sim_option %s not allowed" % sim_option)


    def fit(self, dataset):
        np.random.seed(self.seed)
        self.dataset = dataset
        self.global_mean = dataset.global_mean
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.train_user = dataset.train_user
        self.train_item = dataset.train_item
        self.train_user_indices = dataset.train_user_indices
        self.train_item_indices = dataset.train_item_indices
        self.train_ratings = dataset.train_ratings
        self.test_user_indices = dataset.test_user_indices
        self.test_item_indices = dataset.test_item_indices
        self.test_ratings = dataset.test_ratings
        self.bbu, self.bbi = baseline_als(dataset)

        self.bu = np.zeros((self.n_users,))
        self.bi = np.zeros((self.n_items,))
        self.pu = np.random.normal(loc=0.0, scale=0.1,
                                   size=(self.n_users, self.n_factors))
        self.qi = np.random.normal(loc=0.0, scale=0.1,
                                   size=(self.n_items, self.n_factors))
        self.yj = np.random.normal(loc=0.0, scale=0.1,
                                   size=(self.n_items, self.n_factors))
        self.w = np.random.normal(loc=0.0, scale=0.1,
                                  size=(self.n_items, self.n_items))
        self.c = np.random.normal(loc=0.0, scale=0.1,
                                  size=(self.n_items, self.n_items))

        if not self.batch_training:
            for epoch in range(1, self.n_epochs + 1):
                t0 = time.time()
                for u, i, r in zip(self.train_user_indices,
                                   self.train_item_indices,
                                   self.train_ratings):
                    u_items = list(self.train_user[u].keys())
                    nu_sqrt = np.sqrt(len(u_items))
                    nui = np.sum(self.yj[u_items], axis=0) / nu_sqrt
                    dot = np.dot(self.qi[i], self.pu[u] + nui)
                    intersect_user_item, intersect_user_item_test = self.get_intersect()
                    intersect_items, index_u = intersect_user_item[(u, i)]

                    if len(intersect_items) == 0:
                        err = r - (self.global_mean + self.bu[u] + self.bi[i] + dot)
                        self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                        self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                        self.pu[u] += self.lr * (err * self.qi[i] - self.reg * self.pu[u])
                        self.qi[i] += self.lr * (err * (self.pu[u] + nui) - self.reg * self.qi[i])
                        self.yj[u_items] += self.lr * (err * self.qi[u_items] / nu_sqrt -
                                                       self.reg * self.yj[u_items])

                    else:
                        u_ratings = np.array(list(self.train_user[u].values()))[index_u]
                        base_neighbor = self.global_mean + self.bbu[u] + self.bbi[intersect_items]
                        user_sqrt = np.sqrt(len(intersect_items))
                        ru = np.sum((u_ratings - base_neighbor) * self.w[i][intersect_items]) / user_sqrt
                        nu = np.sum(self.c[i][intersect_items]) / user_sqrt
                        err = r - (self.global_mean + self.bu[u] + self.bi[i] + dot + ru + nu)

                        self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                        self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                        self.pu[u] += self.lr * (err * self.qi[i] - self.reg * self.pu[u])
                        self.qi[i] += self.lr * (err * (self.pu[u] + nui) - self.reg * self.qi[i])
                        self.yj[u_items] += self.lr * (err * self.qi[u_items] / nu_sqrt -
                                                       self.reg * self.yj[u_items])
                        self.w[i][intersect_items] += \
                            self.lr * (err * (u_ratings - base_neighbor) / user_sqrt -
                                                                 self.reg * self.w[i][intersect_items])
                        self.c[i][intersect_items] += self.lr * (err / user_sqrt -
                                                                 self.reg * self.c[i][intersect_items])

                if epoch % 1 == 0:
                    print("Epoch {} time: {:.4f}".format(epoch, time.time() - t0))
                    print("training rmse: ", self.rmse(dataset, "train"))
                    print("test rmse: ", self.rmse(dataset, "test"))

    def get_intersect(self):
        n = len(self.train_item)
        ids = list(self.train_item.keys())
        sim_matrix = get_sim(self.train_item, self.sim_option, n, ids, min_support=self.min_support)
        print("sim matrix shape: ", sim_matrix.shape)

        sim_whole = {}
        for i in range(self.n_items):
            sim_whole[i] = np.argsort(sim_matrix[i])[::-1][:self.k]

        intersect_user_item_train = {}
        for u, i in zip(self.train_user_indices, self.train_item_indices):
            u_items = list(self.train_user[u].keys())
            sim_items = sim_whole[i]
            intersect_items, index_u, _ = np.intersect1d(
                u_items, sim_items, assume_unique=True, return_indices=True)
            intersect_user_item_train[(u, i)] = (intersect_items, index_u)

        intersect_user_item_test = {}
        for user, item in zip(self.test_user_indices, self.test_item_indices):
            u_items = list(self.train_user[u].keys())
            sim_items = sim_whole[item]
            intersect_items, index_u, _ = np.intersect1d(
                u_items, sim_items, assume_unique=True, return_indices=True)
            intersect_user_item_test[(user, item)] = (intersect_items, index_u)

        return intersect_user_item_train, intersect_user_item_test



















