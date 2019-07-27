import time
import math
from operator import itemgetter
import numpy as np
from ..evaluate import rmse
from ..utils.similarities import *
try:
    from ..utils.similarities_cy import cosine_cy, cosine_cym
except ImportError:
    pass
from ..utils.intersect import get_intersect, get_intersect_tf
from ..utils.baseline_estimates import baseline_als, baseline_sgd
try:
    import tensorflow as tf
#    tf.enable_eager_execution()
#    tfe = tf.contrib.eager
except ModuleNotFoundError:
    print("you need tensorflow-eager for tf-version of this model")


class superSVD_909:
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
        time_sim = time.time()
        self.intersect_user_item_train = get_intersect(dataset, self.sim_option,
                                                       self.min_support, self.k, load=True)
        print("sim intersect time: {:.4f}".format(time.time() - time_sim))
        if not self.batch_training:
            for epoch in range(1, self.n_epochs + 1):
                t0 = time.time()
                for u, i, r in zip(self.train_user_indices,
                                   self.train_item_indices,
                                   self.train_labels):
                    u_items = list(self.train_user[u].keys())
                    nu_sqrt = math.sqrt(len(u_items))
                    nui = np.sum(self.yj[u_items], axis=0) / nu_sqrt
                    dot = np.dot(self.qi[i], self.pu[u] + nui)
                    intersect_items, index_u = self.intersect_user_item_train[(u, i)]

                    if len(intersect_items) == 0:
                        err = r - (self.global_mean + self.bu[u] + self.bi[i] + dot)
                        self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                        self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                        self.pu[u] += self.lr * (err * self.qi[i] - self.reg * self.pu[u])
                        self.qi[i] += self.lr * (err * (self.pu[u] + nui) - self.reg * self.qi[i])
                        self.yj[u_items] += self.lr * (err * self.qi[u_items] / nu_sqrt -
                                                       self.reg * self.yj[u_items])

                    else:
                        u_labels = np.array(list(self.train_user[u].values()))[index_u]
                        base_neighbor = self.global_mean + self.bbu[u] + self.bbi[intersect_items]
                        user_sqrt = np.sqrt(len(intersect_items))
                        ru = np.sum((u_labels - base_neighbor) * self.w[i][intersect_items]) / user_sqrt
                        nu = np.sum(self.c[i][intersect_items]) / user_sqrt
                        err = r - (self.global_mean + self.bu[u] + self.bi[i] + dot + ru + nu)

                        self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                        self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                        self.pu[u] += self.lr * (err * self.qi[i] - self.reg * self.pu[u])
                        self.qi[i] += self.lr * (err * (self.pu[u] + nui) - self.reg * self.qi[i])
                        self.yj[u_items] += self.lr * (err * self.qi[u_items] / nu_sqrt -
                                                       self.reg * self.yj[u_items])
                        self.w[i][intersect_items] += \
                            self.lr * (err * (u_labels - base_neighbor) / user_sqrt -
                                                                 self.reg * self.w[i][intersect_items])
                        self.c[i][intersect_items] += self.lr * (err / user_sqrt -
                                                                 self.reg * self.c[i][intersect_items])

                if epoch % 1 == 0:
                    print("Epoch {} time: {:.4f}".format(epoch, time.time() - t0))
                    print("training rmse: ", self.rmse(dataset, "train"))
                    print("test rmse: ", self.rmse(dataset, "test"))

        else:
            for epoch in range(1, self.n_epochs + 1):
                t0 = time.time()
                random_users = np.random.permutation(list(self.train_user.keys()))
                for u in random_users:
                    u_items = list(self.train_user[u].keys())
                    u_labels = np.array(list(self.train_user[u].values()))
                    nu_sqrt = math.sqrt(len(u_items))
                    nui = np.sum(self.yj[u_items], axis=0) / nu_sqrt
                    dot = np.dot(self.qi[u_items], self.pu[u] + nui)

                    ru = []
                    nu = []
                    for single_item in u_items:
                        intersect_items, index_u = self.intersect_user_item_train[(u, single_item)]
                        if len(intersect_items) == 0:
                            ru.append(0)
                            nu.append(0)
                        else:
                            u_labels_intersect = np.array(list(self.train_user[u].values()))[index_u]
                            base_neighbor = self.global_mean + self.bbu[u] + self.bbi[intersect_items]
                            user_sqrt = np.sqrt(len(intersect_items))
                            ru_single = np.sum((u_labels_intersect - base_neighbor) *
                                               self.w[single_item][intersect_items]) / user_sqrt
                            nu_single = np.sum(self.c[single_item][intersect_items]) / user_sqrt
                            ru.append(ru_single)
                            nu.append(nu_single)
                    ru = np.array(ru)
                    nu = np.array(nu)

                    err = u_labels - (self.global_mean + self.bu[u] + self.bi[u_items] + dot + ru + nu)
                    err = err.reshape(len(u_items), 1)
                    self.bu[u] += self.lr * (np.sum(err) - self.reg * self.bu[u])
                    self.bi[u_items] += self.lr * (err.flatten() - self.reg * self.bi[u_items])
                    self.qi[u_items] += self.lr * (err * (self.pu[u] + nui) - self.reg * self.qi[u_items])
                    self.pu[u] += self.lr * (np.sum(err * self.qi[u_items], axis=0) - self.reg * self.pu[u])
                    self.yj[u_items] += self.lr * (err * self.qi[u_items] / nu_sqrt - self.reg * self.yj[u_items])

                    for single_item, error in zip(u_items, err):
                        intersect_items, index_u = self.intersect_user_item_train[(u, single_item)]
                        if len(intersect_items) == 0:
                            continue
                        else:
                            u_labels = np.array(list(self.train_user[u].values()))[index_u]
                            base_neighbor = self.global_mean + self.bbu[u] + self.bbi[intersect_items]
                            user_sqrt = np.sqrt(len(intersect_items))
                            self.w[single_item, intersect_items] += self.lr * (
                                    error.flatten() * (u_labels - base_neighbor) / user_sqrt -
                                    self.reg * self.w[single_item, intersect_items])
                            self.c[single_item, intersect_items] += self.lr * (error.flatten() / user_sqrt -
                                                                    self.reg * self.c[single_item, intersect_items])

                if epoch % 1 == 0:
                    print("Epoch {} time: {:.4f}".format(epoch, time.time() - t0))
                    print("training rmse: ", self.rmse(dataset, "train"))
                    print("test rmse: ", self.rmse(dataset, "test"))

    def predict(self, u, i):
        try:
            u_items = list(self.train_user[u].keys())
            nui = np.sum(self.yj[u_items], axis=0) / np.sqrt(len(u_items))
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
                user_sqrt = np.sqrt(len(intersect_items))
                ru = np.sum((u_labels - base_neighbor) * self.w[i][intersect_items]) / user_sqrt
                nu = np.sum(self.c[i][intersect_items]) / user_sqrt
                pred += (ru + nu)

            pred = np.clip(pred, 1, 5)

        except IndexError:
            pred = self.global_mean
        return pred

    def rmse(self, dataset, mode="train"):
        if mode == "train":
            user_indices = dataset.train_user_indices
            item_indices = dataset.train_item_indices
            labels = dataset.train_labels
        elif mode == "test":
            user_indices = dataset.test_user_indices
            item_indices = dataset.test_item_indices
            labels = dataset.test_labels

        pred = []
        for u, i in zip(user_indices, item_indices):
            p = self.predict(u, i)
            pred.append(p)
        score = np.sqrt(np.mean(np.power(pred - labels, 2)))
        return score


class superSVD_tf:
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
        start_time = time.time()
        tf.set_random_seed(self.seed)
        train_user_indices = dataset.train_user_indices
        train_item_indices = dataset.train_item_indices
        test_user_indices = dataset.test_user_indices
        test_item_indices = dataset.test_item_indices
        train_labels = dataset.train_labels
        test_labels = dataset.test_labels
        global_mean = dataset.global_mean

        bu = tf.Variable(tf.zeros([dataset.n_users]))
        bi = tf.Variable(tf.zeros([dataset.n_items]))
        pu = tf.Variable(tf.random_normal([dataset.n_users, self.n_factors], 0.0, 0.01))
        qi = tf.Variable(tf.random_normal([dataset.n_items, self.n_factors], 0.0, 0.01))
        yj = tf.Variable(tf.random_normal([dataset.n_items, self.n_factors], 0.0, 0.01))
        w = tf.Variable(tf.random_normal([dataset.n_items, dataset.n_items], 0.0, 0.01))
        c = tf.Variable(tf.random_normal([dataset.n_items, dataset.n_items], 0.0, 0.01))
        bbu, bbi = baseline_als(dataset)

    #    optimizer = tf.train.GradientDescentOptimizer(self.lr)
        optimizer = tf.train.AdamOptimizer(self.lr)

        self.intersect_user_item_train = get_intersect(dataset, self.sim_option,
                                                       self.min_support, self.k)

        def compute_loss():
            pred_whole = []
        #    for u, i in zip(dataset.train_user_indices, dataset.train_item_indices):
            data = tf.data.Dataset.from_tensor_slices((dataset.train_user_indices,
                                                       dataset.train_item_indices,
                                                       dataset.train_labels))
            for one_element in tfe.Iterator(data):
                u = one_element[0].numpy()
                i = one_element[1].numpy()
                u_items = np.array(list(dataset.train_user[u].keys()))
                nui = tf.reduce_sum(tf.gather(yj, u_items), axis=0) / \
                           tf.sqrt(tf.cast(tf.size(u_items), tf.float32))

                dot = tf.reduce_sum(tf.multiply(tf.gather(pu, u) + nui, tf.gather(qi, i)))
                pred = global_mean + tf.gather(bu, u) + tf.gather(bi, i) + dot

                try:
                    intersect_items, index_u = self.intersect_user_item_train[(u, i)]
                except KeyError:
                    intersect_items, index_u = [], -1

                if len(intersect_items) == 0:
                    pass
                else:
                    u_labels = np.array(list(dataset.train_user[u].values()))[index_u]
                    base_neighbor = global_mean + bbu[u] + bbi[intersect_items]
                    user_sqrt = tf.sqrt(tf.cast(tf.size(intersect_items), tf.float32))
                    ru = tf.cast(tf.reduce_sum(
                            (u_labels - base_neighbor) *
                                tf.gather(tf.gather(w, i), intersect_items)), tf.float32) / user_sqrt
                    nu = tf.cast(tf.reduce_sum(tf.gather(tf.gather(c, i), intersect_items)), tf.float32) / user_sqrt
                    pred += ru + nu
                pred_whole.append(pred)

            pred_whole = tf.convert_to_tensor(np.array(pred_whole))
            score = tf.reduce_sum(tf.pow(pred_whole - dataset.train_labels, 2)) + \
                        self.reg * (tf.nn.l2_loss(bu) + tf.nn.l2_loss(bi) + tf.nn.l2_loss(pu) +
                           tf.nn.l2_loss(qi) + tf.nn.l2_loss(yj) + tf.nn.l2_loss(w) + tf.nn.l2_loss(c))
            return score

        for epoch in range(1, self.n_epochs + 1):
            t0 = time.time()
            for u, i in zip(train_user_indices, train_item_indices):
                with tf.GradientTape() as tape:
                    variables = [bu, bi, pu, qi, yj, w, c]
                    loss = compute_loss()
                    grads = tape.gradient(loss, variables)
                    optimizer.apply_gradients(zip(grads, variables))

            train_loss = compute_loss().numpy()
            print("Epoch: ", epoch + 1, "\ttrain loss: {}".format(train_loss))
            print("Epoch {}, training time: {:.4f}".format(epoch + 1, time.time() - t0))

        self.pu = pu.numpy()
        self.qi = qi.numpy()
        self.yj = yj.numpy()
        self.bu = bu.numpy()
        self.bi = bi.numpy()
        self.w = w.numpy()
        self.c = c.numpy()
        self.bbu = bbu
        self.bbi = bbi
        self.global_mean = global_mean
        self.dataset = dataset

    def predict(self, u, i):
        try:
            u_items = list(self.dataset.train_user[u].keys())
            nui = np.sum(self.yj[u_items], axis=0) / np.sqrt(len(u_items))
            pred = self.global_mean + self.bu[u] + self.bi[i] + np.dot(self.pu[u] + nui, self.qi[i])

            try:
                intersect_items, index_u = self.intersect_user_item_train[(u, i)]
            except KeyError:
                intersect_items, index_u = [], -1

            if len(intersect_items) == 0:
                pass
            else:
                u_labels = np.array(list(self.dataset.train_user[u].values()))[index_u]
                base_neighbor = self.global_mean + self.bbu[u] + self.bbi[intersect_items]
                user_sqrt = np.sqrt(len(intersect_items))
                ru = np.sum((u_labels - base_neighbor) * self.w[i][intersect_items]) / user_sqrt
                nu = np.sum(self.c[i][intersect_items]) / user_sqrt
                pred += (ru + nu)

            pred = np.clip(pred, 1, 5)

        except IndexError:
            pred = self.global_mean
        return pred

class superSVD_tf_test:
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
        start_time = time.time()
        tf.set_random_seed(self.seed)
        train_user_indices = dataset.train_user_indices
        train_item_indices = dataset.train_item_indices
        test_user_indices = dataset.test_user_indices
        test_item_indices = dataset.test_item_indices
        train_labels = dataset.train_labels
        test_labels = dataset.test_labels
        global_mean = dataset.global_mean

        bu = tf.Variable(tf.zeros([dataset.n_users]))
        bi = tf.Variable(tf.zeros([dataset.n_items]))
        pu = tf.Variable(tf.random_normal([dataset.n_users, self.n_factors], 0.0, 0.01))
        qi = tf.Variable(tf.random_normal([dataset.n_items, self.n_factors], 0.0, 0.01))
        yj = tf.Variable(tf.random_normal([dataset.n_items, self.n_factors], 0.0, 0.01))
        w = tf.Variable(tf.random_normal([dataset.n_items * dataset.n_items, 1], 0.0, 0.01))
        c = tf.Variable(tf.random_normal([dataset.n_items * dataset.n_items, 1], 0.0, 0.01))
    #    bbu, bbi = baseline_als(dataset)

        labels = tf.placeholder(tf.int32, shape=[None])
        user_indices = tf.placeholder(tf.int32, shape=[None])
        item_indices = tf.placeholder(tf.int32, shape=[None])
        bias_user = tf.nn.embedding_lookup(bu, user_indices)
        bias_item = tf.nn.embedding_lookup(bi, item_indices)

        user_split_items = [[] for u in range(dataset.n_users)]
        for u, i in zip(train_user_indices, train_item_indices):
            user_split_items[u].append(i)

        sparse_dict = {'indices': [], 'values': []}
        for i, user in enumerate(user_split_items):
            for j, item in enumerate(user):
                sparse_dict['indices'].append((i, j))
                sparse_dict['values'].append(item)
        sparse_dict['dense_shape'] = (dataset.n_users, dataset.n_items)
        implicit_feedback = tf.SparseTensor(**sparse_dict)
        yjs = tf.nn.embedding_lookup_sparse(yj, implicit_feedback, sp_weights=None, combiner="sqrtn")
        nu = tf.gather(yjs, np.arange(dataset.n_users))

        wc_sparse, sparse_weight = get_intersect_tf(dataset)
        ws = tf.nn.embedding_lookup_sparse(w, wc_sparse, sp_weights=sparse_weight, combiner="sqrtn")
        cs = tf.nn.embedding_lookup_sparse(c, wc_sparse, sp_weights=None, combiner="sqrtn")

        pn = pu + nu
        embed_user = tf.nn.embedding_lookup(pn, user_indices)
        embed_item = tf.nn.embedding_lookup(qi, item_indices)
        embed_w = tf.nn.embedding_lookup(ws, user_indices)
        embed_c = tf.nn.embedding_lookup(cs, user_indices)

        pred = global_mean + bias_user + bias_item + \
               tf.reduce_sum(tf.multiply(embed_user, embed_item), axis=1) + \
               embed_w + embed_c

        loss = tf.reduce_sum(
            tf.square(
                tf.subtract(
                    tf.cast(labels, tf.float32), pred)))

        reg_pu = tf.contrib.layers.l2_regularizer(self.reg)(pu)
        reg_qi = tf.contrib.layers.l2_regularizer(self.reg)(qi)
        reg_bu = tf.contrib.layers.l2_regularizer(self.reg)(bu)
        reg_bi = tf.contrib.layers.l2_regularizer(self.reg)(bi)
        total_loss = tf.add_n([loss, reg_pu, reg_qi, reg_bu, reg_bi])

        optimizer = tf.train.AdamOptimizer(self.lr)
        #    optimizer = tf.train.GradientDescentOptimizer(self.lr)
        training_op = optimizer.minimize(total_loss)
        init = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init)
        for epoch in range(self.n_epochs):
            t0 = time.time()
            self.sess.run(training_op, feed_dict={labels: train_labels,
                                                  user_indices: train_user_indices,
                                                  item_indices: train_item_indices})

            train_loss = self.sess.run(total_loss,
                                       feed_dict={labels: train_labels,
                                                  user_indices: train_user_indices,
                                                  item_indices: train_item_indices})
            print("Epoch: ", epoch + 1, "\ttrain loss: {}".format(train_loss))
            print("Epoch {}, training time: {:.4f}".format(epoch + 1, time.time() - t0))

    def predict(self, u, i):
        try:
            u_items = list(self.dataset.train_user[u].keys())
            nui = np.sum(self.yj[u_items], axis=0) / np.sqrt(len(u_items))
            pred = self.global_mean + self.bu[u] + self.bi[i] + np.dot(self.pu[u] + nui, self.qi[i])

            try:
                intersect_items, index_u = self.intersect_user_item_train[(u, i)]
            except KeyError:
                intersect_items, index_u = [], -1

            if len(intersect_items) == 0:
                pass
            else:
                u_labels = np.array(list(self.dataset.train_user[u].values()))[index_u]
                base_neighbor = self.global_mean + self.bbu[u] + self.bbi[intersect_items]
                user_sqrt = np.sqrt(len(intersect_items))
                ru = np.sum((u_labels - base_neighbor) * self.w[i][intersect_items]) / user_sqrt
                nu = np.sum(self.c[i][intersect_items]) / user_sqrt
                pred += (ru + nu)

            pred = np.clip(pred, 1, 5)

        except IndexError:
            pred = self.global_mean
        return pred

class superSVD:
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
        time_sim = time.time()
        self.intersect_user_item_train = get_intersect(dataset, self.sim_option,
                                                       self.min_support, self.k, load=True)
        print("sim intersect time: {:.4f}".format(time.time() - time_sim))
        if not self.batch_training:
            for epoch in range(1, self.n_epochs + 1):
                t0 = time.time()
                for u, i, r in zip(self.train_user_indices,
                                   self.train_item_indices,
                                   self.train_labels):
                    u_items = list(self.train_user[u].keys())
                    nu_sqrt = math.sqrt(len(u_items))


            #        nui = np.zeros(self.n_factors)
            #        for j in u_items:
            #            for f in range(self.n_factors):
            #                nui[f] += self.yj[j, f] / nu_sqrt


                    nui = np.sum(self.yj[u_items], axis=0) / nu_sqrt
                    dot = np.dot(self.qi[i], self.pu[u] + nui)
                    intersect_items, index_u = self.intersect_user_item_train[(u, i)]

                    if len(intersect_items) == 0:
                        err = r - (self.global_mean + self.bu[u] + self.bi[i] + dot)
                        self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                        self.bi[i] += self.lr * (err - self.reg * self.bi[i])

                        for f in range(self.n_factors):
                            puf = self.pu[u, f]
                            qif = self.qi[i, f]
                            self.pu[u, f] += self.lr * (err * qif - self.reg * puf)
                            self.qi[i, f] += self.lr * (err * (puf + nui[f]) - self.reg * qif)
                #            for j in u_items:
                #                self.yj[j, f] += self.lr * (err * qif / nu_sqrt - self.reg * self.yj[j, f])

                #        self.pu[u] += self.lr * (err * self.qi[i] - self.reg * self.pu[u])
                #        self.qi[i] += self.lr * (err * (self.pu[u] + nui) - self.reg * self.qi[i])
                        self.yj[u_items] += self.lr * (err * self.qi[u_items] / nu_sqrt -
                                                       self.reg * self.yj[u_items])

                    else:
                        u_labels = np.array(list(self.train_user[u].values()))[index_u]
                        base_neighbor = self.global_mean + self.bbu[u] + self.bbi[intersect_items]
                        user_sqrt = math.sqrt(len(intersect_items))
                        ru = np.sum((u_labels - base_neighbor) * self.w[i][intersect_items]) / user_sqrt
                        nu = np.sum(self.c[i][intersect_items]) / user_sqrt
                        err = r - (self.global_mean + self.bu[u] + self.bi[i] + dot + ru + nu)

                        self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                        self.bi[i] += self.lr * (err - self.reg * self.bi[i])

                        for f in range(self.n_factors):
                            puf = self.pu[u, f]
                            qif = self.qi[i, f]
                            self.pu[u, f] += self.lr * (err * qif - self.reg * puf)
                            self.qi[i, f] += self.lr * (err * (puf + nui[f]) - self.reg * qif)
                #            for j in u_items:
                #                self.yj[j, f] += self.lr * (err * qif / nu_sqrt - self.reg * self.yj[j, f])


                #        self.pu[u] += self.lr * (err * self.qi[i] - self.reg * self.pu[u])
                #        self.qi[i] += self.lr * (err * (self.pu[u] + nui) - self.reg * self.qi[i])
                        self.yj[u_items] += self.lr * (err * self.qi[u_items] / nu_sqrt -
                                                       self.reg * self.yj[u_items])
                        self.w[i][intersect_items] += \
                            self.lr * (err * (u_labels - base_neighbor) / user_sqrt -
                                                                 self.reg * self.w[i][intersect_items])
                        self.c[i][intersect_items] += self.lr * (err / user_sqrt -
                                                                 self.reg * self.c[i][intersect_items])

                if epoch % 1 == 0:
                    print("Epoch {} time: {:.4f}".format(epoch, time.time() - t0))
                    print("training rmse: ", rmse(self, dataset, "train"))
                    print("test rmse: ", rmse(self, dataset, "test"))

        else:
            for epoch in range(1, self.n_epochs + 1):
                t0 = time.time()
                random_users = np.random.permutation(list(self.train_user.keys()))
                for u in random_users:
                    u_items = list(self.train_user[u].keys())
                    u_labels = np.array(list(self.train_user[u].values()))
                    nu_sqrt = math.sqrt(len(u_items))
                    nui = np.sum(self.yj[u_items], axis=0) / nu_sqrt
                    dot = np.dot(self.qi[u_items], self.pu[u] + nui)

                    ru = []
                    nu = []
                    for single_item in u_items:
                        intersect_items, index_u = self.intersect_user_item_train[(u, single_item)]
                        if len(intersect_items) == 0:
                            ru.append(0)
                            nu.append(0)
                        else:
                            u_labels_intersect = np.array(list(self.train_user[u].values()))[index_u]
                            base_neighbor = self.global_mean + self.bbu[u] + self.bbi[intersect_items]
                            user_sqrt = math.sqrt(len(intersect_items))
                            ru_single = np.sum((u_labels_intersect - base_neighbor) *
                                               self.w[single_item][intersect_items]) / user_sqrt
                            nu_single = np.sum(self.c[single_item][intersect_items]) / user_sqrt
                            ru.append(ru_single)
                            nu.append(nu_single)
                    ru = np.array(ru)
                    nu = np.array(nu)

                    err = u_labels - (self.global_mean + self.bu[u] + self.bi[u_items] + dot + ru + nu)
                    err = err.reshape(len(u_items), 1)
                    self.bu[u] += self.lr * (np.sum(err) - self.reg * self.bu[u])
                    self.bi[u_items] += self.lr * (err.flatten() - self.reg * self.bi[u_items])
                    self.qi[u_items] += self.lr * (err * (self.pu[u] + nui) - self.reg * self.qi[u_items])
                    self.pu[u] += self.lr * (np.sum(err * self.qi[u_items], axis=0) - self.reg * self.pu[u])
                    self.yj[u_items] += self.lr * (err * self.qi[u_items] / nu_sqrt - self.reg * self.yj[u_items])

                    for single_item, error in zip(u_items, err):
                        intersect_items, index_u = self.intersect_user_item_train[(u, single_item)]
                        if len(intersect_items) == 0:
                            continue
                        else:
                            u_labels = np.array(list(self.train_user[u].values()))[index_u]
                            base_neighbor = self.global_mean + self.bbu[u] + self.bbi[intersect_items]
                            user_sqrt = math.sqrt(len(intersect_items))
                            self.w[single_item, intersect_items] += self.lr * (
                                    error.flatten() * (u_labels - base_neighbor) / user_sqrt -
                                    self.reg * self.w[single_item, intersect_items])
                            self.c[single_item, intersect_items] += self.lr * (error.flatten() / user_sqrt -
                                                                    self.reg * self.c[single_item, intersect_items])

                if epoch % 1 == 0:
                    print("Epoch {} time: {:.4f}".format(epoch, time.time() - t0))
                    print("training rmse: ", rmse(self, dataset, "train"))
                    print("test rmse: ", rmse(self, dataset, "test"))

    def predict(self, u, i):
        try:
            u_items = list(self.train_user[u].keys())
            nui = np.sum(self.yj[u_items], axis=0) / np.sqrt(len(u_items))
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
                user_sqrt = np.sqrt(len(intersect_items))
                ru = np.sum((u_labels - base_neighbor) * self.w[i][intersect_items]) / user_sqrt
                nu = np.sum(self.c[i][intersect_items]) / user_sqrt
                pred += (ru + nu)

            pred = np.clip(pred, 1, 5)

        except IndexError:
            pred = self.global_mean
        return pred

