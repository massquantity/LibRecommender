# cython: profile=True
cimport cython
import numpy as np
cimport numpy as np
import time
import math
import pickle
from ..evaluate import rmse
from ..utils.similarities import *
from ..utils.similarities_cy import cosine_cy, cosine_cym
from ..utils.intersect import get_intersect, get_intersect_tf
from ..utils.baseline_estimates import baseline_als, baseline_sgd
from libc.math cimport sqrt


class superSVD_cy:
    def __init__(self, n_factors=100, n_epochs=20, lr=0.01, reg=5.0,
                 batch_training=True, k=50, min_support=1,
                 sim_option="pearson", seed=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
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
        self.train_user_indices = list(dataset.train_user_indices)
        self.train_item_indices = list(dataset.train_item_indices)
        self.train_labels = list(dataset.train_labels)
        self.test_user_indices = dataset.test_user_indices
        self.test_item_indices = dataset.test_item_indices
        self.test_labels = dataset.test_labels
        self.bbu, self.bbi = baseline_als(dataset)

        time_sim = time.time()
    #    self.intersect_user_item_train = get_intersect(dataset, self.sim_option,
    #                                                   self.min_support, self.k, load=True)
        self.intersect_items_all, self.intersect_indices_all, self.u_labels_all = \
            get_intersect(dataset, self.sim_option, self.min_support, self.k, load=True)
        print("sim intersect time: {:.4f}".format(time.time() - time_sim))

        self.user_item_list = []
        self.user_label_list = []
        self.user_item_length = []
        for u in range(dataset.n_users):
            self.user_item_list.append(list(dataset.train_user[u].keys()))
            self.user_label_list.append(list(dataset.train_user[u].values()))
            self.user_item_length.append(len(list(dataset.train_user[u])))

    #    self.bbu = list(self.bbu)
    #    self.bbi = list(self.bbi)

    #    dd = []
    #    for i in [self.global_mean, self.bbu, self.bbi, self.n_users, self.n_items, self.train_user_indices, 
    #            self.train_item_indices, self.train_labels, self.u_items_all, self.intersect_items_all, 
    #            self.intersect_indices_all, self.u_labels_all]:
    #        dd.append(i)
    #    with open("dd.pkl", "wb") as f:
    #        pickle.dump(dd, f)

        self.sgd(dataset)

    #    if epoch % 1 == 0:
    #        print("Epoch {} time: {:.4f}".format(epoch, time.time() - t0))
    #        print("training rmse: ", rmse(self, dataset, "train"))
    #        print("test rmse: ", rmse(self, dataset, "test"))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def sgd(self, dataset):   
        cdef int u, i, f, j, p, iis
        cdef double r, err, dot, puf, qif
        cdef double ru, nu, nu_sqrt, user_sqrt
        cdef double global_mean = self.global_mean
        cdef double lr = self.lr
        cdef double reg = self.reg
        cdef double base_neighbor

        cdef np.ndarray[np.double_t] bbu = self.bbu
        cdef np.ndarray[np.double_t] bbi = self.bbi
        cdef np.ndarray[np.double_t] bu
        cdef np.ndarray[np.double_t] bi
        cdef np.ndarray[np.double_t, ndim=2] pu
        cdef np.ndarray[np.double_t, ndim=2] qi
        cdef np.ndarray[np.double_t, ndim=2] yj
        cdef np.ndarray[np.double_t] nui
        cdef np.ndarray[np.double_t, ndim=2] w
        cdef np.ndarray[np.double_t, ndim=2] c
#        cdef np.ndarray[np.double_t] u_labels
#        cdef np.ndarray[np.double_t] base_neighbor
#        cdef np.ndarray[np.int_t] index_u
#        cdef np.ndarray[np.int_t] intersect_items
#        cdef np.ndarray[np.int_t, ndim=2] intersect_user_item_train = self.intersect_user_item_train
#        cdef np.ndarray[np.int_t] train_user_indices = self.train_user_indices
#        cdef np.ndarray[np.int_t] train_item_indices = self.train_item_indices
#        cdef np.ndarray[np.double_t] train_labels = self.train_labels

        bu = np.zeros((self.n_users,), np.double)
        bi = np.zeros((self.n_items,), np.double)
        pu = np.random.normal(loc=0.0, scale=0.1,
                            size=(self.n_users, self.n_factors))
        qi = np.random.normal(loc=0.0, scale=0.1,
                            size=(self.n_items, self.n_factors))
        yj = np.random.normal(loc=0.0, scale=0.1,
                            size=(self.n_items, self.n_factors))
        w = np.random.normal(loc=0.0, scale=0.1,
                            size=(self.n_items, self.n_items))
        c = np.random.normal(loc=0.0, scale=0.1,
                            size=(self.n_items, self.n_items))
    #    nui = np.zeros((self.n_factors,), np.double)

        for epoch in range(self.n_epochs):
            t0 = time.time()
            for p in range(len(self.train_labels)):
                u = self.train_user_indices[p]
                i = self.train_item_indices[p]
                r = self.train_labels[p]

            #    u_items = [j for j in dataset.train_user[u]]
            #    u_items = self.u_items_all[p]
                nu_sqrt = sqrt(self.user_item_length[u])

                nui = np.zeros((self.n_factors,), np.double)
                for j in self.user_item_list[u]:
                    for f in range(self.n_factors):
                        nui[f] += yj[j, f] / nu_sqrt

                dot = 0.0
                for f in range(self.n_factors):
                    dot += qi[i, f] * (pu[u, f] + nui[f])

        #        nui = np.sum(self.yj[u_items], axis=0) / nu_sqrt
        #        dot = np.dot(self.qi[i], self.pu[u] + nui)
        #        intersect_items, index_u = intersect_user_item_train[(u, i)]
                intersect_items = self.intersect_items_all[p]
                index_u = self.intersect_indices_all[p]

                if len(intersect_items) == 0:
                    err = r - (global_mean + bu[u] + bi[i] + dot)
                    bu[u] += lr * (err - reg * bu[u])
                    bi[i] += lr * (err - reg * bi[i])

                    for f in range(self.n_factors):
                        pu[u, f] += lr * (err * qi[i, f] - reg * pu[u, f])
                        qi[i, f] += lr * (err * (pu[u, f] + nui[f]) - reg * qi[i, f])
                        for j in self.user_item_list[u]:
                            yj[j, f] += lr * (err * qi[i, f] / nu_sqrt - reg * yj[j, f])

            #        self.pu[u] += self.lr * (err * self.qi[i] - self.reg * self.pu[u])
            #        self.qi[i] += self.lr * (err * (self.pu[u] + nui) - self.reg * self.qi[i])
            #        self.yj[u_items] += self.lr * (err * self.qi[u_items] / nu_sqrt -
            #                                        self.reg * self.yj[u_items])

                else:
                    user_sqrt = sqrt(len(intersect_items))
                #    u_labels = np.zeros((self.n_items,), np.double)
                #    base_neighbor = np.zeros((self.n_items,), np.double)
                #    user_values = [v for v in self.train_user[u].values()]
                    u_labels = self.u_labels_all[p]
                #    base_neighbor = self.base_neighbor_all[p]
                    ru = 0.0
                    nu = 0.0
                    for j in range(len(intersect_items)):
                        iis = intersect_items[j]
                        base_neighbor = global_mean + bbu[u] + bbi[iis]
                        ru += (u_labels[j] - base_neighbor) * w[i, iis]
                        nu += c[i, iis]
                    #    u_labels[intersect_items[j]] = user_values[index_u[j]]
                    #    base_neighbor[j] = global_mean + bbu[u] + bbi[intersect_items[j]]
                    #    ru += (u_labels[intersect_items[j]] - base_neighbor[intersect_items[j]]) * w[i, j]
                    #    nu += c[i, j]

                    ru /= user_sqrt
                    nu /= user_sqrt

            #        u_labels = np.array(list(self.train_user[u].values()))[index_u]
            #        base_neighbor = global_mean + bbu[u] + bbi[intersect_items]
            #        user_sqrt = sqrt(len(intersect_items))
            #        ru = np.sum((u_labels - base_neighbor) * self.w[i][intersect_items]) / user_sqrt
            #        nu = np.sum(self.c[i][intersect_items]) / user_sqrt

                    err = r - (global_mean + bu[u] + bi[i] + dot + ru + nu)
                    bu[u] += lr * (err - reg * bu[u])
                    bi[i] += lr * (err - reg * bi[i])

                    for f in range(self.n_factors):
                        pu[u, f] += lr * (err * qi[i, f] - reg * pu[u, f])
                        qi[i, f] += lr * (err * (pu[u, f] + nui[f]) - reg * qi[i, f])
                        for j in self.user_item_list[u]:
                            yj[j, f] += lr * (err * qi[i, f] / nu_sqrt - reg * yj[j, f])

                    for j in range(len(intersect_items)):
                        iis = intersect_items[j]
                        base_neighbor = global_mean + bbu[u] + bbi[iis]
                        w[i, iis] += lr * (err * (u_labels[j] - base_neighbor) / user_sqrt - reg * w[i, iis])
                        c[i, iis] += lr * (err / user_sqrt - reg * c[i, iis])


            #        self.pu[u] += self.lr * (err * self.qi[i] - self.reg * self.pu[u])
            #        self.qi[i] += self.lr * (err * (self.pu[u] + nui) - self.reg * self.qi[i])
            #        self.yj[u_items] += self.lr * (err * self.qi[u_items] / nu_sqrt -
            #                                        self.reg * self.yj[u_items])
            #        self.w[i][intersect_items] += \
            #            self.lr * (err * (u_labels - base_neighbor) / user_sqrt -
            #                                                    self.reg * self.w[i][intersect_items])
            #        self.c[i][intersect_items] += self.lr * (err / user_sqrt -
            #                                                    self.reg * self.c[i][intersect_items])

            print("Epoch {} time: {:.4f}".format(epoch, time.time() - t0))
        
        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi
        self.yj = yj
        self.w = w
        self.c = c


    def predict(self, u, i):
        try:
            u_items = list(self.train_user[u].keys())
            nui = np.sum(self.yj[u_items], axis=0) / sqrt(len(u_items))
            pred = self.global_mean + self.bu[u] + self.bi[i] + np.dot(self.pu[u] + nui, self.qi[i])

            try:
                intersect_items, index_u = self.intersect_user_item_train[(u, i)]
            except KeyError:
                intersect_items, index_u = [], -1

            if len(intersect_items) == 0:
                pass
            else:
                u_labels = np.array(list(self.train_user[u].values()))[index_u]
                base_neighbor = self.global_mean + self.bbu[u] + self.bbi[intersect_items]
                user_sqrt = sqrt(len(intersect_items))
                ru = np.sum((u_labels - base_neighbor) * self.w[i][intersect_items]) / user_sqrt
                nu = np.sum(self.c[i][intersect_items]) / user_sqrt
                pred += (ru + nu)

            pred = np.clip(pred, 1, 5)

        except IndexError:
            pred = self.global_mean
        return pred


class superSVD_cy_909:
    def __init__(self, n_factors=100, n_epochs=20, lr=0.01, reg=5.0,
                 batch_training=True, k=50, min_support=1,
                 sim_option="pearson", seed=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
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
        self.train_user_indices = list(dataset.train_user_indices)
        self.train_item_indices = list(dataset.train_item_indices)
        self.train_labels = list(dataset.train_labels)
        self.test_user_indices = dataset.test_user_indices
        self.test_item_indices = dataset.test_item_indices
        self.test_labels = dataset.test_labels
        self.bbu, self.bbi = baseline_als(dataset)

        time_sim = time.time()
    #    self.intersect_user_item_train = get_intersect(dataset, self.sim_option,
    #                                                   self.min_support, self.k, load=True)
        self.intersect_items_all, self.intersect_indices_all, self.u_labels_all = \
            get_intersect(dataset, self.sim_option, self.min_support, self.k, load=True)
        print("sim intersect time: {:.4f}".format(time.time() - time_sim))

        self.u_items_all = []
        for u in self.train_user_indices:
            self.u_items_all.append(list(dataset.train_user[u]))

        self.sgd(dataset)

    #    if epoch % 1 == 0:
    #        print("Epoch {} time: {:.4f}".format(epoch, time.time() - t0))
    #        print("training rmse: ", rmse(self, dataset, "train"))
    #        print("test rmse: ", rmse(self, dataset, "test"))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def sgd(self, dataset):   
        cdef int u, i, f, j, p, iis
        cdef double r, err, dot, puf, qif
        cdef double ru, nu, nu_sqrt, user_sqrt
        cdef double global_mean = self.global_mean
        cdef double lr = self.lr
        cdef double reg = self.reg
        cdef double base_neighbor

        cdef double[:] bbu = self.bbu
        cdef double[:] bbi = self.bbi
        cdef double[:] bu = np.zeros((self.n_users,), np.double)
        cdef double[:] bi = np.zeros((self.n_items,), np.double)
        cdef double[:, :] pu = np.random.normal(loc=0.0, scale=0.1,
                            size=(self.n_users, self.n_factors))
        cdef double[:, :] qi = np.random.normal(loc=0.0, scale=0.1,
                            size=(self.n_items, self.n_factors))
        cdef double[:, :] yj = np.random.normal(loc=0.0, scale=0.1,
                            size=(self.n_items, self.n_factors))
        cdef double[:] nui = np.zeros((self.n_factors,), np.double)
        cdef double[:, :] w = np.random.normal(loc=0.0, scale=0.1,
                            size=(self.n_items, self.n_items))
        cdef double[:, :] c = np.random.normal(loc=0.0, scale=0.1,
                            size=(self.n_items, self.n_items))
#        cdef np.ndarray[np.double_t] u_labels
#        cdef np.ndarray[np.double_t] base_neighbor
#        cdef np.ndarray[np.int_t] index_u
#        cdef np.ndarray[np.int_t] intersect_items
#        cdef np.ndarray[np.int_t, ndim=2] intersect_user_item_train = self.intersect_user_item_train
#        cdef np.ndarray[np.int_t] train_user_indices = self.train_user_indices
#        cdef np.ndarray[np.int_t] train_item_indices = self.train_item_indices
#        cdef np.ndarray[np.double_t] train_labels = self.train_labels


    #    nui = np.zeros((self.n_factors,), np.double)

        for epoch in range(self.n_epochs):
            t0 = time.time()
            for p in range(len(self.train_labels)):
                u = self.train_user_indices[p]
                i = self.train_item_indices[p]
                r = self.train_labels[p]

            #    u_items = [j for j in dataset.train_user[u]]
                u_items = self.u_items_all[p]
                nu_sqrt = sqrt(len(u_items))

                nui = np.zeros((self.n_factors,), np.double)
                for j in u_items:
                    for f in range(self.n_factors):
                        nui[f] += yj[j, f] / nu_sqrt

                dot = 0.0
                for f in range(self.n_factors):
                    dot += qi[i, f] * (pu[u, f] + nui[f])

        #        nui = np.sum(self.yj[u_items], axis=0) / nu_sqrt
        #        dot = np.dot(self.qi[i], self.pu[u] + nui)
        #        intersect_items, index_u = intersect_user_item_train[(u, i)]
                intersect_items = self.intersect_items_all[p]
                index_u = self.intersect_indices_all[p]

                if len(intersect_items) == 0:
                    err = r - (global_mean + bu[u] + bi[i] + dot)
                    bu[u] += lr * (err - reg * bu[u])
                    bi[i] += lr * (err - reg * bi[i])

                    for f in range(self.n_factors):
                        pu[u, f] += lr * (err * qi[i, f] - reg * pu[u, f])
                        qi[i, f] += lr * (err * (pu[u, f] + nui[f]) - reg * qi[i, f])
                        for j in u_items:
                            yj[j, f] += lr * (err * qi[i, f] / nu_sqrt - reg * yj[j, f])

            #        self.pu[u] += self.lr * (err * self.qi[i] - self.reg * self.pu[u])
            #        self.qi[i] += self.lr * (err * (self.pu[u] + nui) - self.reg * self.qi[i])
            #        self.yj[u_items] += self.lr * (err * self.qi[u_items] / nu_sqrt -
            #                                        self.reg * self.yj[u_items])

                else:
                    user_sqrt = sqrt(len(intersect_items))
                #    u_labels = np.zeros((self.n_items,), np.double)
                #    base_neighbor = np.zeros((self.n_items,), np.double)
                #    user_values = [v for v in self.train_user[u].values()]
                    u_labels = self.u_labels_all[p]
                #    base_neighbor = self.base_neighbor_all[p]
                    ru = 0.0
                    nu = 0.0
                    for j in range(len(intersect_items)):
                        iis = intersect_items[j]
                        base_neighbor = global_mean + bbu[u] + bbi[iis]
                        ru += (u_labels[j] - base_neighbor) * w[i, iis]
                        nu += c[i, iis]
                    #    u_labels[intersect_items[j]] = user_values[index_u[j]]
                    #    base_neighbor[j] = global_mean + bbu[u] + bbi[intersect_items[j]]
                    #    ru += (u_labels[intersect_items[j]] - base_neighbor[intersect_items[j]]) * w[i, j]
                    #    nu += c[i, j]

                    ru /= user_sqrt
                    nu /= user_sqrt

            #        u_labels = np.array(list(self.train_user[u].values()))[index_u]
            #        base_neighbor = global_mean + bbu[u] + bbi[intersect_items]
            #        user_sqrt = sqrt(len(intersect_items))
            #        ru = np.sum((u_labels - base_neighbor) * self.w[i][intersect_items]) / user_sqrt
            #        nu = np.sum(self.c[i][intersect_items]) / user_sqrt

                    err = r - (global_mean + bu[u] + bi[i] + dot + ru + nu)
                    bu[u] += lr * (err - reg * bu[u])
                    bi[i] += lr * (err - reg * bi[i])

                    for f in range(self.n_factors):
                        pu[u, f] += lr * (err * qi[i, f] - reg * pu[u, f])
                        qi[i, f] += lr * (err * (pu[u, f] + nui[f]) - reg * qi[i, f])
                        for j in u_items:
                            yj[j, f] += lr * (err * qi[i, f] / nu_sqrt - reg * yj[j, f])

                    for j in range(len(intersect_items)):
                        iis = intersect_items[j]
                        base_neighbor = global_mean + bbu[u] + bbi[iis]
                        w[i, iis] += lr * (err * (u_labels[j] - base_neighbor) / user_sqrt - reg * w[i, iis])
                        c[i, iis] += lr * (err / user_sqrt - reg * c[i, iis])


            #        self.pu[u] += self.lr * (err * self.qi[i] - self.reg * self.pu[u])
            #        self.qi[i] += self.lr * (err * (self.pu[u] + nui) - self.reg * self.qi[i])
            #        self.yj[u_items] += self.lr * (err * self.qi[u_items] / nu_sqrt -
            #                                        self.reg * self.yj[u_items])
            #        self.w[i][intersect_items] += \
            #            self.lr * (err * (u_labels - base_neighbor) / user_sqrt -
            #                                                    self.reg * self.w[i][intersect_items])
            #        self.c[i][intersect_items] += self.lr * (err / user_sqrt -
            #                                                    self.reg * self.c[i][intersect_items])

            print("Epoch {} time: {:.4f}".format(epoch, time.time() - t0))
        
        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi
        self.yj = yj
        self.w = w
        self.c = c