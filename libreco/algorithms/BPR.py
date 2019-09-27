"""

References: Steffen Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback"
            (https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf)

author: massquantity

"""
import time
import itertools
import logging
import numpy as np
from scipy.sparse import lil_matrix
import tensorflow as tf
from ..utils.sampling import PairwiseSampling
from .Base import BasePure
from ..evaluate.evaluate import binary_cross_entropy, MAP_at_k, MAR_at_k, NDCG_at_k, precision_tf
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss


class Bpr(BasePure):
    def __init__(self, n_factors=16, lr=0.01, n_epochs=20, reg=0.0,
                 batch_size=64, seed=42, k=20, method="mf", neg_sampling=False):
        self.n_factors = n_factors
        self.lr = lr
        self.n_epochs = n_epochs
        self.reg = reg
        self.batch_size = batch_size
        self.seed = seed
        self.k = k
        self.method = method
        self.neg_sampling = neg_sampling
        print("using FTRL............ \n")
        super(Bpr, self).__init__()

    def build_model(self, dataset):
        if self.method == "mf":
            tf.set_random_seed(self.seed)
            self.dataset = dataset
            self.n_users = dataset.n_users
            self.n_items = dataset.n_items
            self.user = tf.placeholder(tf.int32, shape=[None], name="user")
            self.item_i = tf.placeholder(tf.int32, shape=[None], name="item_i")
            self.item_j = tf.placeholder(tf.int32, shape=[None], name="item_j")

            self.pu = tf.Variable(tf.truncated_normal([dataset.n_users, self.n_factors], mean=0.0, stddev=0.01))
            self.qi = tf.Variable(tf.truncated_normal([dataset.n_items, self.n_factors], mean=0.0, stddev=0.01))
            self.embed_user = tf.nn.embedding_lookup(self.pu, self.user)
            self.embed_item_i = tf.nn.embedding_lookup(self.qi, self.item_i)
            self.embed_item_j = tf.nn.embedding_lookup(self.qi, self.item_j)

            self.x_ui = tf.reduce_sum(tf.multiply(self.embed_user, self.embed_item_i), axis=1)
            self.x_uj = tf.reduce_sum(tf.multiply(self.embed_user, self.embed_item_j), axis=1)
            self.x_uij = self.x_ui - self.x_uj

            self.reg_user = self.reg * tf.nn.l2_loss(self.embed_user)  #######################
            self.reg_item_i = self.reg * tf.nn.l2_loss(self.embed_item_i)
            self.reg_item_j = self.reg * tf.nn.l2_loss(self.embed_item_j)
            self.loss = - tf.reduce_sum(
                tf.log(1 / (1 + tf.exp(-self.x_uij))) - self.reg_user - self.reg_item_i - self.reg_item_j)

            self.item_t = tf.placeholder(tf.int32, shape=[None])
            self.labels = tf.placeholder(tf.float32, shape=[None])
            self.embed_item_t = tf.nn.embedding_lookup(self.qi, self.item_t)
            self.logits = tf.reduce_sum(tf.multiply(self.embed_user, self.embed_item_t), axis=1)
            self.test_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
            self.prob = tf.sigmoid(self.logits)
            self.pred = tf.where(self.prob >= 0.5,
                                 tf.fill(tf.shape(self.logits), 1.0),
                                 tf.fill(tf.shape(self.logits), 0.0))
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.labels), tf.float32))
            self.precision = precision_tf(self.pred, self.labels)

        elif self.method == "knn":
            LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
            logging.basicConfig(format=LOG_FORMAT)
            logging.warning("knn method requires huge memory for constructing similarity matrix")
            self.dataset = dataset
            self.n_users = dataset.n_users
            self.n_items = dataset.n_items
            self.sim_matrix = lil_matrix((dataset.n_items, dataset.n_items))

        else:
            raise ValueError("method name must be one of these: [mf, knn]")

    def fit(self, dataset, verbose=1, **kwargs):
        if verbose > 0:
            sampling = PairwiseSampling(dataset)
            self.test_user, self.test_item, self.test_label = sampling(mode="test")

        if self.method == "mf":
            self.build_model(dataset)
        #    self.optimizer = tf.train.AdamOptimizer(self.lr)
            self.optimizer = tf.train.FtrlOptimizer(learning_rate=0.1, l1_regularization_strength=1e-3)
            self.training_op = self.optimizer.minimize(self.loss)
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)
            print(tf.trainable_variables())
            with self.sess.as_default():
                for epoch in range(1, self.n_epochs + 1):
                    t0 = time.time()
                    sampling = PairwiseSampling(self.dataset, batch_size=self.batch_size)
                    n_batches = int(np.ceil(len(dataset.train_user_indices) / self.batch_size))
                    for n in range(n_batches):
                        batch_user, \
                        batch_item_i, \
                        batch_item_j = sampling.next_mf_tf()

                        self.sess.run(self.training_op, feed_dict={self.user: batch_user,
                                                                   self.item_i: batch_item_i,
                                                                   self.item_j: batch_item_j})

                    if verbose > 0:
                        print("Epoch {}: training time: {:.4f}".format(epoch, time.time() - t0))
                        metrics = kwargs.get("metrics", self.metrics)
                        if hasattr(self, "sess"):
                            self.print_metrics_tf(dataset, epoch, **metrics)
                        else:
                            self.print_metrics(dataset, epoch, **metrics)
                        print()

        elif self.method == "knn":
            self.build_model(dataset)
            for epoch in range(1, self.n_epochs + 1):
                t0 = time.time()
                sampling = PairwiseSampling(self.dataset, batch_size=1)
                for _ in range(len(self.dataset.train_user_indices)):
                    user, item_i, item_i_nei, item_j, item_j_nei, x_uij = \
                        sampling.next_knn(self.sim_matrix, k=self.k)

                    sigmoid = 1.0 / (1.0 + np.exp(x_uij))
                    sigmoid = np.repeat(sigmoid, len(item_i_nei))
                    self.sim_matrix[item_i, item_i_nei] += self.lr * (
                            sigmoid - self.reg * self.sim_matrix[item_i, item_i_nei])
                    self.sim_matrix[item_i_nei, item_i] = self.sim_matrix[item_i, item_i_nei]
                    self.sim_matrix[item_i, item_i] = 1.0
                    self.sim_matrix[item_j, item_j_nei] += self.lr * (
                            - sigmoid - self.reg * self.sim_matrix[item_j, item_j_nei])
                    self.sim_matrix[item_j_nei, item_j] = self.sim_matrix[item_j, item_j_nei]
                    self.sim_matrix[item_j, item_j] = 1.0

                if verbose > 0:
                    print("Epoch {}: training time: {:.4f}".format(epoch, time.time() - t0))
                    metrics = kwargs.get("metrics", self.metrics)
                    if hasattr(self, "sess"):
                        self.print_metrics_tf(dataset, epoch, **metrics)
                    else:
                        self.print_metrics(dataset, epoch, **metrics)
                    print()

                    '''
                    print("Epoch {}, train time: {:.4f}".format(epoch, time.time() - t0))
                    t1 = time.time()
                    test_loss, test_prob = binary_cross_entropy(self, test_user, test_item, test_label)
                    test_roc_auc = roc_auc_score(test_label, test_prob)
                    test_pr_auc = average_precision_score(test_label, test_prob)
                    print("evaluate time: {:.2f}".format(time.time() - t1))
                    print("test loss: {:.4f}, test auc: {:.4f}, test pr auc: {:.4f}".format(
                        test_loss, test_roc_auc, test_pr_auc))

                    t2 = time.time()
                    mean_average_precision_10 = MAP_at_k(self, self.dataset, 10, sample_user=1000)
                    print("\t MAP@{}: {:.4f}".format(10, mean_average_precision_10))
                    print("\t MAP@10 time: {:.4f}".format(time.time() - t2))

                    t3 = time.time()
                    mean_average_recall_50 = MAR_at_k(self, self.dataset, 50, sample_user=1000)
                    print("\t MAR@{}: {:.4f}".format(50, mean_average_recall_50))
                    print("\t MAR@50 time: {:.4f}".format(time.time() - t3))

                    t4 = time.time()
                    NDCG = NDCG_at_k(self, self.dataset, 10, sample_user=1000)
                    print("\t NDCG@{}: {:.4f}".format(10, NDCG))
                    print("\t NDCG@10 time: {:.4f}".format(time.time() - t4))
                    print()
                    '''
    def predict(self, u, i):
        if self.method == "mf":
            try:
                y_prob = self.sess.run([self.prob], feed_dict={self.user: np.array([u]),
                                                               self.item_t: np.array([i])})
            except tf.errors.InvalidArgumentError:
                y_prob = [0.0]
            return y_prob[0]

        elif self.method == "knn":
            try:
                u_items = np.array(list(self.dataset.train_user[u]))
                k_neightbors = self.sim_matrix[i, u_items]
                logits = np.sum(k_neightbors)
                prob = 1.0 / (1.0 + np.exp(-logits))
            except IndexError:
                prob = 0.0
            return prob

    def recommend_user(self, u, n_rec):
        if self.method == "mf":
            user_indices = np.full(self.n_items, u)
            item_indices = np.arange(self.n_items)
            preds = self.sess.run(self.prob, feed_dict={self.user: user_indices,
                                                        self.item_t: item_indices})
            preds = preds.ravel()
            consumed = self.dataset.train_user[u]
            count = n_rec + len(consumed)
            ids = np.argpartition(preds, -count)[-count:]
            rank = sorted(zip(ids, preds[ids]), key=lambda x: -x[1])
            return list(itertools.islice((rec for rec in rank if rec[0] not in consumed), n_rec))

        elif self.method == "knn":
            u_items = np.array(list(self.dataset.train_user[u]))
            k_neightbors_all_items = self.sim_matrix[:, u_items]  # self.sim_matrix.todense()
            preds = np.array(np.sum(k_neightbors_all_items, axis=1)).ravel()
            count = n_rec + len(u_items)
            ids = np.argpartition(preds, -count)[-count:]
            rank = sorted(zip(ids, preds[ids]), key=lambda x: -x[1])
            return list(itertools.islice((rec for rec in rank if rec[0] not in u_items), n_rec))




