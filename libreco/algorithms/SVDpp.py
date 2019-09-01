"""

References: Yehuda Koren "Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model"
            (https://dl.acm.org/citation.cfm?id=1401944)

author: massquantity

"""
import time
import itertools
import numpy as np
from .Base import BasePure
from libreco.utils.initializers import truncated_normal
from ..evaluate import precision_tf
from ..utils import NegativeSampling
import tensorflow as tf


class SVDpp(BasePure):
    def __init__(self, n_factors=100, n_epochs=20, lr=0.01, reg=1e-3,
                 batch_size=256, seed=42, task="rating", neg_sampling=False):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.seed = seed
        self.task = task
        self.neg_sampling = neg_sampling
        super(SVDpp, self).__init__()

    def build_model(self, dataset):
        tf.set_random_seed(self.seed)
        self.dataset = dataset
        self.global_mean = dataset.global_mean
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.train_user_indices = dataset.train_user_indices
        self.train_item_indices = dataset.train_item_indices
        implicit_feedback = self.get_implicit_feedback(dataset)

        if dataset.lower_upper_bound is not None:
            self.lower_bound = dataset.lower_upper_bound[0]
            self.upper_bound = dataset.lower_upper_bound[1]
        else:
            self.lower_bound = None
            self.upper_bound = None

        self.user_indices = tf.placeholder(tf.int32, shape=[None])
        self.item_indices = tf.placeholder(tf.int32, shape=[None])
        self.labels = tf.placeholder(tf.float32, shape=[None])

        self.bu = tf.Variable(tf.zeros([dataset.n_users]))
        self.bi = tf.Variable(tf.zeros([dataset.n_items]))
        self.pu = tf.Variable(tf.random_normal([dataset.n_users, self.n_factors], 0.0, 0.01))
        self.qi = tf.Variable(tf.random_normal([dataset.n_items, self.n_factors], 0.0, 0.01))
        self.yj = tf.Variable(tf.random_normal([dataset.n_items, self.n_factors], 0.0, 0.01))  # qi, yj share embedding ???

    #    yjs = tf.nn.embedding_lookup_sparse(self.yj, implicit_feedback, sp_weights=None, combiner="sqrtn")
    #    self.nu = tf.gather(yjs, np.arange(dataset.n_users))
        self.nu = tf.nn.embedding_lookup_sparse(self.yj, implicit_feedback, sp_weights=None, combiner="sqrtn")
        self.pn = self.pu + self.nu

        self.bias_user = tf.nn.embedding_lookup(self.bu, self.user_indices)
        self.bias_item = tf.nn.embedding_lookup(self.bi, self.item_indices)
        self.embed_user = tf.nn.embedding_lookup(self.pn, self.user_indices)
        self.embed_item = tf.nn.embedding_lookup(self.qi, self.item_indices)

        if self.task == "rating":
            self.pred = tf.cast(self.global_mean, tf.float32) + self.bias_user + self.bias_item + \
                        tf.reduce_sum(tf.multiply(self.embed_user, self.embed_item), axis=1)

            self.loss = tf.losses.mean_squared_error(labels=self.labels,
                                                     predictions=self.pred)

            if self.lower_bound is not None and self.upper_bound is not None:
                self.rmse = tf.sqrt(tf.losses.mean_squared_error(labels=self.labels,
                                predictions=tf.clip_by_value(self.pred, self.lower_bound, self.upper_bound)))
            else:
                self.rmse = self.loss

        elif self.task == "ranking":
            self.logits = self.bias_user + self.bias_item + \
                          tf.reduce_sum(tf.multiply(self.embed_user, self.embed_item), axis=1)
            self.logits = tf.reshape(self.logits, [-1])
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
            self.y_prob = tf.sigmoid(self.logits)
            self.pred = tf.where(self.y_prob >= 0.5,
                                 tf.fill(tf.shape(self.logits), 1.0),
                                 tf.fill(tf.shape(self.logits), 0.0))
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.labels), tf.float32))
            self.precision = precision_tf(self.pred, self.labels)

        self.reg_pu = self.reg * tf.nn.l2_loss(self.pu)
        self.reg_qi = self.reg * tf.nn.l2_loss(self.qi)
        self.reg_bu = self.reg * tf.nn.l2_loss(self.bu)
        self.reg_bi = self.reg * tf.nn.l2_loss(self.bi)
        self.reg_yj = self.reg * tf.nn.l2_loss(self.yj)
        self.total_loss = tf.add_n([self.loss, self.reg_pu, self.reg_qi, self.reg_bu, self.reg_bi, self.reg_yj])
    #    self.total_loss = self.loss

    def fit(self, dataset, verbose=1, **kwargs):
        self.build_model(dataset)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.training_op = optimizer.minimize(self.total_loss)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        with self.sess.as_default():
            if self.task == "rating" or (self.task == "ranking" and not self.neg_sampling):
                for epoch in range(1, self.n_epochs + 1):
                    t0 = time.time()
                    n_batches = int(np.ceil(len(dataset.train_labels) / self.batch_size))
                    for n in range(n_batches):
                        end = min(len(dataset.train_labels), (n + 1) * self.batch_size)
                        r = dataset.train_labels[n * self.batch_size: end]
                        u = dataset.train_user_indices[n * self.batch_size: end]
                        i = dataset.train_item_indices[n * self.batch_size: end]
                        self.sess.run(self.training_op, feed_dict={self.labels: r,
                                                                   self.user_indices: u,
                                                                   self.item_indices: i})

                    if verbose > 0:
                        print("Epoch {}, training_time: {:.2f}".format(epoch, time.time() - t0))
                        metrics = kwargs.get("metrics", self.metrics)
                        if hasattr(self, "sess"):
                            self.print_metrics_tf(dataset, epoch, **metrics)
                        else:
                            self.print_metrics(dataset, epoch, **metrics)
                        print()

            elif self.task == "ranking":
                for epoch in range(1, self.n_epochs + 1):
                    t0 = time.time()
                    neg = NegativeSampling(dataset, dataset.num_neg, self.batch_size)
                    n_batches = int(np.ceil(len(dataset.train_label_implicit) / self.batch_size))
                    for n in range(n_batches):
                        user_batch, item_batch, label_batch = neg.next_batch()
                        self.sess.run([self.training_op], feed_dict={self.labels: label_batch,
                                                                     self.user_indices: user_batch,
                                                                     self.item_indices: item_batch})

                    if verbose > 0:
                        print("Epoch {}: training time: {:.4f}".format(epoch, time.time() - t0))
                        metrics = kwargs.get("metrics", self.metrics)
                        if hasattr(self, "sess"):
                            self.print_metrics_tf(dataset, epoch, **metrics)
                        else:
                            self.print_metrics(dataset, epoch, **metrics)
                        print()

    def predict(self, u, i):
        try:
            target = self.pred if self.task == "rating" else self.y_prob
            pred = self.sess.run(target, feed_dict={self.user_indices: [u],
                                                    self.item_indices: [i]})
            if self.lower_bound is not None and self.upper_bound is not None:
                pred = np.clip(pred, self.lower_bound, self.upper_bound) if self.task == "rating" else pred[0]
        except tf.errors.InvalidArgumentError:
            pred = self.dataset.global_mean if self.task == "rating" else 0.0
        return pred

    def recommend_user(self, u, n_rec):
        items = np.arange(self.dataset.n_items)
        consumed = self.dataset.train_user[u]
        count = n_rec + len(consumed)
        target = self.pred if self.task == "rating" else self.y_prob

        preds = self.sess.run(target, feed_dict={self.user_indices: [u],
                                                 self.item_indices: items})
        ids = np.argpartition(preds, -count)[-count:]
        rank = sorted(zip(ids, preds[ids]), key=lambda x: -x[1])
        return list(itertools.islice((rec for rec in rank if rec[0] not in consumed), n_rec))

    def get_implicit_feedback(self, data):
        user_split_items = [[] for u in range(data.n_users)]
        for u, i in zip(data.train_user_indices, data.train_item_indices):
            user_split_items[u].append(i)

        sparse_dict = {'indices': [], 'values': []}
        for i, user in enumerate(user_split_items):
            for j, item in enumerate(user):
                sparse_dict['indices'].append((i, j))
                sparse_dict['values'].append(item)
        sparse_dict['dense_shape'] = (data.n_users, data.n_items)
        implicit_feedback = tf.SparseTensor(**sparse_dict)
        return implicit_feedback



class sss:
    def __init__(self, n_factors=100, n_epochs=20, lr=0.01, reg=5.0,
                 batch_size=256, seed=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.seed = seed

    def fit(self, dataset, verbose=1):
        np.random.seed(self.seed)
        self.dataset = dataset
        self.global_mean = dataset.global_mean
        self.bu = np.zeros((dataset.n_users))
        self.bi = np.zeros((dataset.n_items))
        self.pu = truncated_normal(mean=0.0, scale=0.1,
                                   shape=(dataset.n_users, self.n_factors))
        self.qi = truncated_normal(mean=0.0, scale=0.1,
                                   shape=(dataset.n_items, self.n_factors))
        self.yj = truncated_normal(mean=0.0, scale=0.1,
                                   shape=(dataset.n_items, self.n_factors))

        for epoch in range(1, self.n_epochs + 1):
            t0 = time.time()
            neg = NegativeSampling(dataset, dataset.num_neg, self.batch_size)
            n_batches = int(np.ceil(len(dataset.train_label_implicit) / self.batch_size))
        #    n_batches = len(dataset.train_labels) // self.batch_size
            for n in range(n_batches):
                users, items, labels = neg.next_batch()
                nu = []
                u_items = []
                u_sqrt = []
        #        t0 = time.time()
                for u in users:
                    u_items_single = list(dataset.train_user[u].keys())
                    nu_sqrt = np.sqrt(len(u_items_single))
                    nu_single = np.sum(self.yj[u_items_single], axis=0) / nu_sqrt
                    nu.append(nu_single)
                    u_items.append(u_items_single)
                    u_sqrt.append(nu_sqrt)
         #       print("1: ", time.time() - t0)
         #       t3 = time.time()
                self.pn = self.pu[users] + np.array(nu)
        #        print("4: ", time.time() - t3)
        #        t1 = time.time()
                dot = np.sum(np.multiply(self.pn, self.qi[items]), axis=1)
                err = labels - (self.global_mean + self.bu[users] + self.bi[items] + dot)
                self.bu[users] += self.lr * (err - self.reg * self.bu[users])
                self.bi[items] += self.lr * (err - self.reg * self.bi[items])
                self.pu[users] += self.lr * (err.reshape(-1, 1) * self.qi[items] - self.reg * self.pu[users])
                self.qi[items] += self.lr * (err.reshape(-1, 1) * self.pn - self.reg * self.qi[items])
        #        print("2: ", time.time() - t1)
        #        t2 = time.time()
                err /= nu_sqrt
                for k, i in enumerate(u_items):
                    self.yj[i] += self.lr * (err[k] * self.qi[i] - self.reg * self.yj[i])
        #        print("3: ", time.time() - t2)

            if verbose > 0:
                print("Epoch {}: training time: {:.4f}".format(epoch, time.time() - t0))
                t1 = time.time()
                test_loss, test_prob = binary_cross_entropy(self, dataset.test_user_implicit,
                                                            dataset.test_item_implicit,
                                                            dataset.test_label_implicit)

                test_auc = roc_auc_score(dataset.test_label_implicit, test_prob)
                test_ap = average_precision_score(dataset.test_label_implicit, test_prob)
                precision_test, recall_test, _ = precision_recall_curve(dataset.test_label_implicit,
                                                                        test_prob)
                test_pr_auc = auc(recall_test, precision_test)
                print("\t test loss: {:.4f}".format(test_loss))
                print("\t test roc auc: {:.2f} "
                      "\n\t test average precision: {:.2f}"
                      "\n\t test pr auc: {:.2f}".format(test_auc, test_ap, test_pr_auc))
                print("\t auc, etc. time: {:.4f}".format(time.time() - t1))

                t4 = time.time()
                mean_average_precision_10 = MAP_at_k(self, self.dataset, 10, sample_user=1000)
                print("\t MAP@{}: {:.4f}".format(10, mean_average_precision_10))
                print("\t MAP@10 time: {:.4f}".format(time.time() - t4))

                t5 = time.time()
                mean_average_recall_50 = MAR_at_k(self, self.dataset, 50, sample_user=1000)
                print("\t MAR@{}: {:.4f}".format(50, mean_average_recall_50))
                print("\t MAR@50 time: {:.4f}".format(time.time() - t5))

                t6 = time.time()
                NDCG = NDCG_at_k(self, self.dataset, 10, sample_user=1000)
                print("\t NDCG@{}: {:.4f}".format(10, NDCG))
                print("\t NDCG@10 time: {:.4f}".format(time.time() - t6))
                print()

    def fit_7890(self, dataset, verbose=1):
        np.random.seed(self.seed)
        self.dataset = dataset
        self.global_mean = dataset.global_mean
        self.bu = np.zeros((dataset.n_users))
        self.bi = np.zeros((dataset.n_items))
        self.pu = truncated_normal(mean=0.0, scale=0.1,
                                   shape=(dataset.n_users, self.n_factors))
        self.qi = truncated_normal(mean=0.0, scale=0.1,
                                   shape=(dataset.n_items, self.n_factors))
        self.yj = truncated_normal(mean=0.0, scale=0.1,
                                   shape=(dataset.n_items, self.n_factors))

        for epoch in range(1, self.n_epochs + 1):
            t0 = time.time()
            neg = NegativeSampling(dataset, dataset.num_neg, self.batch_size)
            n_batches = int(np.ceil(len(dataset.train_label_implicit) / self.batch_size))
        #    n_batches = len(dataset.train_labels) // self.batch_size
            for n in range(n_batches):
                users, items, labels = neg.next_batch()
                nu = []
                u_items = []
                u_sqrt = []
                items_dict = dict()
            #    t0 = time.time()
                for u in users:
                    u_items_single = list(dataset.train_user[u].keys())
                    for i in u_items_single:
                        if i in items_dict:
                            items_dict[i] += 1
                        else:
                            items_dict[i] = 0

                    nu_sqrt = np.sqrt(len(u_items_single))
                    nu_single = np.sum(self.yj[u_items_single], axis=0) / nu_sqrt
                    nu.append(nu_single)
                    u_items.append(u_items_single)
                    u_sqrt.append(nu_sqrt)
            #    print("1: ", time.time() - t0)
            #   t3 = time.time()
                self.pn = self.pu[users] + np.array(nu)
            #    print("4: ", time.time() - t3)
            #    t1 = time.time()
                dot = np.sum(np.multiply(self.pn, self.qi[items]), axis=1)
                err = labels - (self.global_mean + self.bu[users] + self.bi[items] + dot)
                self.bu[users] += self.lr * (err - self.reg * self.bu[users])
                self.bi[items] += self.lr * (err - self.reg * self.bi[items])
                self.pu[users] += self.lr * (err.reshape(-1, 1) * self.qi[items] - self.reg * self.pu[users])
                self.qi[items] += self.lr * (err.reshape(-1, 1) * self.pn - self.reg * self.qi[items])
            #    print("2: ", time.time() - t1)
                t2 = time.time()
                err /= nu_sqrt
                for k, i in enumerate(items_dict):
                    err_i = np.sum(err[np.where(items == i)])
                    self.yj[i] += self.lr * (err_i * self.qi[i] - items_dict[i] * self.reg * self.yj[i])
                print("3: ", time.time() - t2)

            if verbose > 0:
                print("Epoch {}: training time: {:.4f}".format(epoch, time.time() - t0))
                t1 = time.time()
                test_loss, test_prob = binary_cross_entropy(self, dataset.test_user_implicit,
                                                            dataset.test_item_implicit,
                                                            dataset.test_label_implicit)

                test_auc = roc_auc_score(dataset.test_label_implicit, test_prob)
                test_ap = average_precision_score(dataset.test_label_implicit, test_prob)
                precision_test, recall_test, _ = precision_recall_curve(dataset.test_label_implicit,
                                                                        test_prob)
                test_pr_auc = auc(recall_test, precision_test)
                print("\t test loss: {:.4f}".format(test_loss))
                print("\t test roc auc: {:.2f} "
                      "\n\t test average precision: {:.2f}"
                      "\n\t test pr auc: {:.2f}".format(test_auc, test_ap, test_pr_auc))
                print("\t auc, etc. time: {:.4f}".format(time.time() - t1))

                t4 = time.time()
                mean_average_precision_10 = MAP_at_k(self, self.dataset, 10, sample_user=1000)
                print("\t MAP@{}: {:.4f}".format(10, mean_average_precision_10))
                print("\t MAP@10 time: {:.4f}".format(time.time() - t4))

                t5 = time.time()
                mean_average_recall_50 = MAR_at_k(self, self.dataset, 50, sample_user=1000)
                print("\t MAR@{}: {:.4f}".format(50, mean_average_recall_50))
                print("\t MAR@50 time: {:.4f}".format(time.time() - t5))

                t6 = time.time()
                NDCG = NDCG_at_k(self, self.dataset, 10, sample_user=1000)
                print("\t NDCG@{}: {:.4f}".format(10, NDCG))
                print("\t NDCG@10 time: {:.4f}".format(time.time() - t6))
                print()

    def predict(self, u, i):
        try:
            u_items = list(self.dataset.train_user[u].keys())
            nu = np.sum(self.yj[u_items], axis=0) / np.sqrt(len(u_items))
            logits = self.global_mean + self.bu[u] + self.bi[i] + np.dot(self.pu[u] + nu, self.qi[i])
            prob = 1.0 / (1.0 + np.exp(-logits))
            pred = 1.0 if prob >= 0.5 else 0.0
        except IndexError:
            pred, prob = 0.0, 0.0
        return prob, pred

    def recommend_user(self, u, n_rec):
        consumed = self.dataset.train_user[u]
        count = n_rec + len(consumed)

        u_items = list(self.dataset.train_user[u].keys())
        nu = np.sum(self.yj[u_items], axis=0) / np.sqrt(len(u_items))
        preds = self.global_mean + self.bu[u] + self.bi + np.dot(self.pu[u] + nu, self.qi.T)
        ids = np.argpartition(preds, -count)[-count:]
        rank = sorted(zip(ids, preds[ids]), key=lambda x: -x[1])
        return list(itertools.islice((rec for rec in rank if rec[0] not in consumed), n_rec))


