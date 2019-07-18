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
from ..utils.intersect import get_intersect, get_intersect_tf, get_sim
from ..utils.baseline_estimates import baseline_als, baseline_sgd
from libc.math cimport sqrt


class superSVD_cys:
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
        self.train_user_indices = dataset.train_user_indices
        self.train_item_indices = dataset.train_item_indices
        self.train_labels = dataset.train_labels
        self.test_user_indices = dataset.test_user_indices
        self.test_item_indices = dataset.test_item_indices
        self.test_labels = dataset.test_labels
        self.bbu, self.bbi = baseline_als(dataset)

        self.sim_matrix = get_sim(dataset)

        self.user_item_list = []
        self.user_label_list = []
        self.user_item_length = []
        for u in range(dataset.n_users):
            self.user_item_list.append(list(dataset.train_user[u].keys()))
            self.user_label_list.append(list(dataset.train_user[u].values()))
            self.user_item_length.append(len(list(dataset.train_user[u])))

        '''
        self.user_item_list = []
        self.user_item_indices = [0]
        self.user_label_list = []
        self.user_item_length = []
        for u in range(dataset.n_users):
            self.user_item_list.extend(list(dataset.train_user[u].keys()))
            self.user_item_indices.append(len(list(dataset.train_user[u])) + self.user_item_indices[-1])
            self.user_label_list.extend(list(dataset.train_user[u].values()))
            self.user_item_length.append(len(list(dataset.train_user[u])))

        user_item_list = np.array(self.user_item_list, dtype=np.intc)
        user_item_indices = np.array(self.user_item_indices, dtype=np.intc)
        user_label_list = np.array(self.user_label_list, dtype=np.double)
        user_item_length = np.array(self.user_item_length, dtype=np.intc)
        '''
        self.sgd(dataset)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def sgd(self, dataset):   
        cdef int u, i, f, j, p, k, g, h
        cdef double r, err, dot, puf, qif, rg
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
#        cdef np.ndarray[np.double_t, ndim=2] sim_matrix = self.sim_matrix
#        cdef np.ndarray[np.double_t] u_labels
#        cdef np.ndarray[np.double_t] base_neighbor
#        cdef np.ndarray[np.int_t] index_u
#        cdef np.ndarray[np.int_t] intersect_items
#        cdef np.ndarray[np.int_t, ndim=2] intersect_user_item_train = self.intersect_user_item_train
#        cdef np.ndarray[np.int_t] train_user_indices = self.train_user_indices
#        cdef np.ndarray[np.int_t] train_item_indices = self.train_item_indices
#        cdef np.ndarray[np.double_t] train_labels = self.train_labels
        cdef int hlist[40]
        cdef double rglist[40]
#        cdef int user_item_list[len(self.user_item_list)]  #######################################

        bu = np.zeros((self.n_users,), np.double)
        bi = np.zeros((self.n_items,), np.double)
        pu = np.random.normal(loc=0.0, scale=0.1,
                            size=(self.n_users, self.n_factors)).astype(np.double)
        qi = np.random.normal(loc=0.0, scale=0.1,
                            size=(self.n_items, self.n_factors)).astype(np.double)
        yj = np.random.normal(loc=0.0, scale=0.1,
                            size=(self.n_items, self.n_factors)).astype(np.double)
        w = np.random.normal(loc=0.0, scale=0.1,
                            size=(self.n_items, self.n_items)).astype(np.double)
        c = np.random.normal(loc=0.0, scale=0.1,
                            size=(self.n_items, self.n_items)).astype(np.double)
    #    nui = np.zeros((self.n_factors,), np.double)

        for epoch in range(self.n_epochs):
            t0 = time.time()
            for p in range(len(self.train_labels)):
                u = self.train_user_indices[p]
                i = self.train_item_indices[p]
                r = self.train_labels[p]

                nu_sqrt = sqrt(self.user_item_length[u])

                nui = np.zeros((self.n_factors,), np.double)
                for j in self.user_item_list[u]:
                    for f in range(self.n_factors):
                        nui[f] += yj[j, f] / nu_sqrt

        #        for j in range(user_item_indices[u], user_item_indices[u+1]):
        #            h = user_item_list[j]
        #            for f in range(self.n_factors):
        #                nui[f] += yj[h, f] / nu_sqrt
                

                dot = 0.0
                for f in range(self.n_factors):
                    dot += qi[i, f] * (pu[u, f] + nui[f])

                k = 0
                ru = 0.0
                nu = 0.0
                for j in range(len(self.user_item_list[u])):
                    g = self.user_item_list[u][j]
                    for h in self.sim_matrix[i]:
                        if g == h:
                            hlist[k] = h
                            rg = self.user_label_list[u][j]
                            rglist[k] = rg - (global_mean + bbu[u] + bbi[h])
                            ru += (rg - (global_mean + bbu[u] + bbi[h])) * w[i, h]
                            nu += c[i, h]
                            k += 1

        #        for j in range(user_item_indices[u], user_item_indices[u+1]):
        #            g = user_item_list[j]
        #            for h in self.sim_matrix[i]:
        #                if g == h:
        #                    rg = user_label_list[j]
        #                    hlist[k] = h
        #                    rglist[k] = rg - (global_mean + bbu[u] + bbi[h])
        #                    ru += (rg - (global_mean + bbu[u] + bbi[h])) * w[i, h]
        #                    nu += c[i, h]
        #                    k += 1


                if k == 0:
                    err = r - (global_mean + bu[u] + bi[i] + dot)
                    bu[u] += lr * (err - reg * bu[u])
                    bi[i] += lr * (err - reg * bi[i])

                    for f in range(self.n_factors):
                        pu[u, f] += lr * (err * qi[i, f] - reg * pu[u, f])
                        qi[i, f] += lr * (err * (pu[u, f] + nui[f]) - reg * qi[i, f])
                        for j in self.user_item_list[u]:
                            yj[j, f] += lr * (err * qi[i, f] / nu_sqrt - reg * yj[j, f])
            #            for j in range(user_item_indices[u], user_item_indices[u+1]):
            #                h = user_item_list[j]
            #                yj[h, f] += lr * (err * qi[i, f] / nu_sqrt - reg * yj[h, f])

                else:
                    user_sqrt = sqrt(k)
                    ru /= user_sqrt
                    nu /= user_sqrt
                    err = r - (global_mean + bu[u] + bi[i] + dot + ru + nu)
                    bu[u] += lr * (err - reg * bu[u])
                    bi[i] += lr * (err - reg * bi[i])

                    for f in range(self.n_factors):
                        pu[u, f] += lr * (err * qi[i, f] - reg * pu[u, f])
                        qi[i, f] += lr * (err * (pu[u, f] + nui[f]) - reg * qi[i, f])
                        for j in self.user_item_list[u]:
                            yj[j, f] += lr * (err * qi[i, f] / nu_sqrt - reg * yj[j, f])
            #            for j in range(user_item_indices[u], user_item_indices[u+1]):
            #                h = self.user_item_list[j]
            #                yj[h, f] += lr * (err * qi[i, f] / nu_sqrt - reg * yj[h, f])

            #        for g, rg in zip(self.user_item_list[u], self.user_label_list[u]):
            #            for h in self.sim_matrix[i]:
            #                if g == h:
            #                    ru = rg - (global_mean + bbu[u] + bbi[h])
            #                    w[i, h] += lr * (err * ru / user_sqrt - reg * w[i, h])
            #                    c[i, h] += lr * (err / user_sqrt - reg * c[i, h])

                    for j in range(k):
                        h = hlist[j]
                        ru = rglist[j]
                        w[i, h] += lr * (err * ru / user_sqrt - reg * w[i, h])
                        c[i, h] += lr * (err / user_sqrt - reg * c[i, h])

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

