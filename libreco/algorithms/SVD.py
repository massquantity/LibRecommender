import time
from operator import itemgetter
import itertools
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, auc
from ..evaluate import rmse, accuracy, precision_tf, MAR_at_k, MAP_at_k, NDCG_at_k
from ..utils import NegativeSampling
import tensorflow as tf


class SVD:
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

    def build_model(self, dataset):
        tf.set_random_seed(self.seed)
        self.dataset = dataset
        self.global_mean = dataset.global_mean
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

        self.bias_user = tf.nn.embedding_lookup(self.bu, self.user_indices)
        self.bias_item = tf.nn.embedding_lookup(self.bi, self.item_indices)
        self.embed_user = tf.nn.embedding_lookup(self.pu, self.user_indices)
        self.embed_item = tf.nn.embedding_lookup(self.qi, self.item_indices)

        if self.task == "rating":
            self.pred = self.global_mean + self.bias_user + self.bias_item + \
                        tf.reduce_sum(tf.multiply(self.embed_user, self.embed_item), axis=1)

            self.loss = tf.losses.mean_squared_error(labels=self.labels,
                                                     predictions=self.pred)

            if self.lower_bound is not None and self.upper_bound is not None:
                self.rmse = tf.sqrt(tf.losses.mean_squared_error(labels=self.labels,
                                predictions=tf.clip_by_value(self.pred, self.lower_bound, self.upper_bound)))
            else:
                self.rmse = self.loss

        elif self.task == "ranking":
            self.logits = self.global_mean + self.bias_user + self.bias_item + \
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
        self.total_loss = tf.add_n([self.loss, self.reg_pu, self.reg_qi, self.reg_bu, self.reg_bi])

    def fit(self, dataset, verbose=1):
        """
        :param dataset:
        :param verbose: whether to print train & test metrics, could be slow...
        :return:
        """
        self.build_model(dataset)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        #    self.optimizer = tf.train.FtrlOptimizer(learning_rate=0.1, l1_regularization_strength=1e-3)
        self.training_op = self.optimizer.minimize(self.total_loss)
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

                    if verbose > 0 and self.task == "rating":
                        test_loss, test_rmse = self.sess.run([self.total_loss, self.rmse],
                                                   feed_dict={self.labels: dataset.test_labels,
                                                              self.user_indices: dataset.test_user_indices,
                                                              self.item_indices: dataset.test_item_indices})

                        print("Epoch {}, training_time: {:.2f}".format(epoch, time.time() - t0))
                        print("Epoch {}, test_loss: {:.4f}, test_rmse: {:.4f}".format(
                            epoch, test_loss, test_rmse))
                        print()

                    elif verbose > 0 and self.task == "ranking":
                        print("Epoch {}: training time: {:.4f}".format(epoch, time.time() - t0))
                        t3 = time.time()
                        test_loss, test_accuracy, test_precision = \
                            self.sess.run([self.loss, self.accuracy, self.precision],
                                          feed_dict={self.labels: dataset.test_labels,
                                                     self.user_indices: dataset.test_user_indices,
                                                     self.item_indices: dataset.test_item_indices})

                        print("\ttest loss: {:.4f}".format(test_loss))
                        print("\ttest accuracy: {:.4f}".format(test_accuracy))
                        print("\ttest precision: {:.4f}".format(test_precision))
                        print("\tloss time: {:.4f}".format(time.time() - t3))

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
                        t3 = time.time()
                        test_loss, test_accuracy, test_precision, test_prob = \
                            self.sess.run([self.loss, self.accuracy, self.precision, self.y_prob],
                                feed_dict={self.labels: dataset.test_label_implicit,
                                           self.user_indices: dataset.test_user_implicit,
                                           self.item_indices: dataset.test_item_implicit})

                        print("\ttest loss: {:.4f}".format(test_loss))
                        print("\ttest accuracy: {:.4f}".format(test_accuracy))
                        print("\ttest precision: {:.4f}".format(test_precision))
                        print("\tloss time: {:.4f}".format(time.time() - t3))

                        t1 = time.time()
                        test_auc = roc_auc_score(dataset.test_label_implicit, test_prob)
                        test_ap = average_precision_score(dataset.test_label_implicit, test_prob)
                        precision_test, recall_test, _ = precision_recall_curve(dataset.test_label_implicit,
                                                                                test_prob)
                        test_pr_auc = auc(recall_test, precision_test)
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
            target = self.pred if self.task == "rating" else self.y_prob
            pred = self.sess.run(target, feed_dict={self.user_indices: [u],
                                                    self.item_indices: [i]})
            if self.lower_bound is not None and self.upper_bound is not None:
                pred = np.clip(pred, self.lower_bound, self.upper_bound) if self.task == "rating" else pred[0]
        except tf.errors.InvalidArgumentError:
            pred = self.dataset.global_mean
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







