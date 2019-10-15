"""

Reference: Steffen Rendle "Factorization Machines" (https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)

author: massquantity

"""
import os
import time
import itertools
from collections import OrderedDict
import numpy as np
import tensorflow as tf
from .Base import BasePure, BaseFeat
from ..evaluate.evaluate import precision_tf, MAP_at_k, MAR_at_k, recall_at_k, NDCG_at_k
from ..utils.sampling import NegativeSampling, NegativeSamplingFeat


class FmPure(BasePure):
    def __init__(self, lr, n_epochs=20, n_factors=100, reg=0.0, batch_size=256,
                 seed=42, task="rating", neg_sampling=False):
        self.lr = lr
        self.n_epochs = n_epochs
        self.n_factors = n_factors
        self.reg = reg
        self.batch_size = batch_size
        self.seed = seed
        self.task = task
        self.neg_sampling = neg_sampling
        super(FmPure, self).__init__()

    @staticmethod
    def build_sparse_data(data, user_indices, item_indices):
        first_dim = np.tile(np.arange(len(user_indices)), 2).reshape(-1, 1)
        second_dim = np.concatenate([user_indices, item_indices + data.n_users], axis=0).reshape(-1, 1)
        indices = np.concatenate([first_dim, second_dim], axis=1)
        indices = indices.astype(np.int64)
        values = np.ones(len(user_indices) * 2, dtype=np.float32)
        shape = [len(user_indices), data.n_users + data.n_items]
        return indices, values, shape

    def build_model(self, dataset):
        tf.set_random_seed(self.seed)
        self.dataset = dataset
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.dim = dataset.n_users + dataset.n_items
        if dataset.lower_upper_bound is not None:
            self.lower_bound = dataset.lower_upper_bound[0]
            self.upper_bound = dataset.lower_upper_bound[1]
        else:
            self.lower_bound = None
            self.upper_bound = None

        self.w = tf.Variable(tf.truncated_normal([self.dim, 1], 0.0, 0.01))
        self.v = tf.Variable(tf.truncated_normal([self.dim, self.n_factors], 0.0, 0.01))

        self.x = tf.sparse_placeholder(tf.float32, [None, self.dim])
        self.labels = tf.placeholder(tf.float32, shape=[None], name="labels")
        self.linear_term = tf.sparse_tensor_dense_matmul(self.x, self.w)

        self.pairwise_term = 0.5 * tf.subtract(
                tf.square(tf.sparse_tensor_dense_matmul(self.x, self.v)),
                tf.sparse_tensor_dense_matmul(tf.square(self.x), tf.square(self.v)))

        self.concat = tf.concat([self.linear_term, self.pairwise_term], axis=1)

        if self.task == "rating":
            self.pred = tf.layers.dense(inputs=self.concat, units=1, name="pred")
            self.loss = tf.losses.mean_squared_error(labels=tf.reshape(self.labels, [-1, 1]),
                                                     predictions=self.pred)

            if self.lower_bound is not None and self.upper_bound is not None:
                self.rmse = tf.sqrt(tf.losses.mean_squared_error(labels=tf.reshape(self.labels, [-1, 1]),
                                predictions=tf.clip_by_value(self.pred, self.lower_bound, self.upper_bound)))
            else:
                self.rmse = self.loss

        #    reg_w = self.reg * tf.nn.l2_loss(self.w)
            reg_v = self.reg * tf.nn.l2_loss(self.v)
            self.total_loss = tf.add_n([self.loss, reg_v])

        elif self.task == "ranking":
            self.logits = tf.layers.dense(inputs=self.concat, units=1, name="logits")
            self.logits = tf.reshape(self.logits, [-1])
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits))

            self.y_prob = tf.sigmoid(self.logits)
            self.pred = tf.where(self.y_prob >= 0.5,
                                 tf.fill(tf.shape(self.logits), 1.0),
                                 tf.fill(tf.shape(self.logits), 0.0))
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.labels), tf.float32))
            self.precision = precision_tf(self.pred, self.labels)

        #    reg_w = self.reg * tf.nn.l2_loss(self.w)
            reg_v = self.reg * tf.nn.l2_loss(self.v)
            self.total_loss = tf.add_n([self.loss, reg_v])

    def fit(self, dataset, verbose=1, **kwargs):
        self.build_model(dataset)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
    #    self.optimizer = tf.train.FtrlOptimizer(learning_rate=0.1, l1_regularization_strength=1e-3)
        self.training_op = self.optimizer.minimize(self.total_loss)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.sess.run(tf.local_variables_initializer())
        with self.sess.as_default():
            if self.task == "rating" or (self.task == "ranking" and not self.neg_sampling):
                for epoch in range(1, self.n_epochs + 1):
                    t0 = time.time()
                    n_batches = int(np.ceil(len(dataset.train_labels) / self.batch_size))
                    for n in range(n_batches):
                        end = min(len(dataset.train_labels), (n + 1) * self.batch_size)
                        user_batch = dataset.train_user_indices[n * self.batch_size: end]
                        item_batch = dataset.train_item_indices[n * self.batch_size: end]
                        label_batch = dataset.train_labels[n * self.batch_size: end]

                        indices_batch, values_batch, shape_batch = FmPure.build_sparse_data(dataset,
                                                                                            user_batch,
                                                                                            item_batch)
                        self.sess.run(self.training_op, feed_dict={self.x: tf.SparseTensorValue(indices_batch,
                                                                                                values_batch,
                                                                                                shape_batch),
                                                                   self.labels: label_batch})

                    if verbose > 0:
                        print("Epoch {}, training time: {:.2f}".format(epoch, time.time() - t0))
                        indices_test, values_test, shape_test = FmPure.build_sparse_data(
                                                                    dataset,
                                                                    dataset.test_user_indices,
                                                                    dataset.test_item_indices)
                        test_loss, test_rmse = self.sess.run([self.total_loss, self.rmse],
                                                             feed_dict={self.x: (indices_test,
                                                                                 values_test,
                                                                                 shape_test),
                                                                        self.labels: dataset.test_labels})

                        print("Epoch {}, test_loss: {:.4f}, test_rmse: {:.4f}".format(
                            epoch, test_loss, test_rmse))

            elif self.task == "ranking":
                for epoch in range(1, self.n_epochs + 1):
                    t0 = time.time()
                    neg = NegativeSampling(dataset, dataset.num_neg, self.batch_size)
                    n_batches = int(np.ceil(len(dataset.train_labels) / self.batch_size))
                    for n in range(n_batches):
                        user_batch, item_batch, label_batch = neg.next_batch()
                        indices_batch, values_batch, shape_batch = FmPure.build_sparse_data(dataset,
                                                                                            user_batch,
                                                                                            item_batch)
                        self.sess.run(self.training_op, feed_dict={self.x: tf.SparseTensorValue(indices_batch,
                                                                                                values_batch,
                                                                                                shape_batch),
                                                                   self.labels: label_batch})

                    if verbose > 0:
                        indices_test, values_test, shape_test = FmPure.build_sparse_data(
                                                                    dataset,
                                                                    dataset.test_user_implicit,
                                                                    dataset.test_item_implicit)
                        test_loss, test_accuracy, test_precision = \
                            self.sess.run([self.loss, self.accuracy, self.precision],
                                          feed_dict={self.x: (indices_test, values_test, shape_test),
                                                     self.labels: dataset.test_label_implicit})

                        print("Epoch {}, training time: {:.2f}".format(epoch, time.time() - t0))
                        print("Epoch {}, test loss: {:.4f}, test accuracy: {:.4f}, test precision: {:.4f}".format(
                                epoch, test_loss, test_accuracy, test_precision))

                        t2 = time.time()
                        mean_average_precision = MAP_at_k(self, self.dataset, 20, sample_user=1000)
                        print("\t MAP@{}: {:.4f}".format(20, mean_average_precision))
                        print("\t MAP time: {:.4f}".format(time.time() - t2))

                        t3 = time.time()
                        recall = recall_at_k(self, self.dataset, 50, sample_user=1000)
                        print("\t MAR@{}: {:.4f}".format(50, recall))
                        print("\t MAR time: {:.4f}".format(time.time() - t3))

                        t4 = time.time()
                        ndcg = NDCG_at_k(self, self.dataset, 20, sample_user=1000)
                        print("\t NDCG@{}: {:.4f}".format(20, ndcg))
                        print("\t NDCG time: {:.4f}".format(time.time() - t4))
                        print()

    def predict(self, u, i):
        index, value, shape = FmPure.build_sparse_data(self.dataset, np.array([u]), np.array([i]))
        try:
            pred = self.sess.run(self.pred, feed_dict={self.x: (index, value, shape)})
            if self.lower_bound is not None and self.upper_bound is not None:
                pred = np.clip(pred, self.lower_bound, self.upper_bound) if self.task == "rating" else pred[0]
        except tf.errors.InvalidArgumentError:
            pred = self.dataset.global_mean if self.task == "rating" else 0.0
        return pred

    def recommend_user(self, u, n_rec):
        consumed = self.dataset.train_user[u]
        count = n_rec + len(consumed)
        target = self.pred if self.task == "rating" else self.y_prob

        user_indices = np.full(self.n_items, u)
        item_indices = np.arange(self.n_items)
        index, value, shape = FmPure.build_sparse_data(self.dataset, user_indices, item_indices)
        preds = self.sess.run(target, feed_dict={self.x: tf.SparseTensorValue(index, value, shape)})

        ids = np.argpartition(preds, -count)[-count:]
        rank = sorted(zip(ids, preds[ids]), key=lambda x: -x[1])
        return list(itertools.islice((rec for rec in rank if rec[0] not in consumed), n_rec))


class FmFeat(BaseFeat):
    def __init__(self, lr, n_epochs=20, n_factors=100, reg=0.0, batch_size=256, seed=42,
                 task="rating", neg_sampling=False):
        self.lr = lr
        self.n_epochs = n_epochs
        self.n_factors = n_factors
        self.reg = reg
        self.batch_size = batch_size
        self.seed = seed
        self.task = task
        self.neg_sampling = neg_sampling
        super(FmFeat, self).__init__()

    def build_model(self, dataset):
        tf.set_random_seed(self.seed)
        self.dataset = dataset
        self.field_size = dataset.train_feat_indices.shape[1]
        self.feature_size = dataset.feature_size
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.global_mean = dataset.global_mean
        self.total_items_unique = self.item_info
        if dataset.lower_upper_bound is not None:
            self.lower_bound = dataset.lower_upper_bound[0]
            self.upper_bound = dataset.lower_upper_bound[1]
        else:
            self.lower_bound = None
            self.upper_bound = None

        self.feature_indices = tf.placeholder(tf.int32, shape=[None, self.field_size])
        self.feature_values = tf.placeholder(tf.float32, shape=[None, self.field_size])
        self.labels = tf.placeholder(tf.float32, shape=[None])

        self.w = tf.Variable(tf.truncated_normal([self.feature_size + 1, 1], 0.0, 0.01))  # feature_size + 1####
        self.v = tf.Variable(tf.truncated_normal([self.feature_size + 1, self.n_factors], 0.0, 0.01))
        self.feature_values_reshape = tf.reshape(self.feature_values, shape=[-1, self.field_size, 1])

        self.linear_embedding = tf.nn.embedding_lookup(self.w, self.feature_indices)   # N * F * 1
        self.linear_term = tf.reduce_sum(tf.multiply(self.linear_embedding, self.feature_values_reshape), 2)

        self.feature_embedding = tf.nn.embedding_lookup(self.v, self.feature_indices)  # N * F * K
        self.feature_embedding = tf.multiply(self.feature_embedding, self.feature_values_reshape)

        self.pairwise_term = 0.5 * tf.subtract(
            tf.square(tf.reduce_sum(self.feature_embedding, axis=2)), # axis=1 ?
            tf.reduce_sum(tf.square(self.feature_embedding), axis=2))

        self.concat = tf.concat([self.linear_term, self.pairwise_term], axis=1)

        if self.task == "rating":
            self.pred = tf.layers.dense(inputs=self.concat, units=1)
            self.loss = tf.losses.mean_squared_error(labels=tf.reshape(self.labels, [-1, 1]),
                                                     predictions=self.pred)

            if self.lower_bound is not None and self.upper_bound is not None:
                self.rmse = tf.sqrt(tf.losses.mean_squared_error(abels=tf.reshape(self.labels, [-1, 1]),
                                predictions=tf.clip_by_value(self.pred, self.lower_bound, self.upper_bound)))
            else:
                self.rmse = self.loss

        #    reg_w = self.reg * tf.nn.l2_loss(self.w)
            reg_v = self.reg * tf.nn.l2_loss(self.v)
            self.total_loss = tf.add_n([self.loss, reg_v])

        elif self.task == "ranking":
            self.logits = tf.layers.dense(inputs=self.concat, units=1, name="logits")
            self.logits = tf.reshape(self.logits, [-1])
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits))

            self.y_prob = tf.sigmoid(self.logits)
            self.pred = tf.where(self.y_prob >= 0.5,
                                 tf.fill(tf.shape(self.logits), 1.0),
                                 tf.fill(tf.shape(self.logits), 0.0))
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.labels), tf.float32))
            self.precision = precision_tf(self.pred, self.labels)

        #    reg_w = self.reg * tf.nn.l2_loss(self.w)
            reg_v = self.reg * tf.nn.l2_loss(self.v)
            self.total_loss = tf.add_n([self.loss, reg_v])

    def fit(self, dataset, verbose=1, pre_sampling=True, **kwargs):
        self.build_model(dataset)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
    #    self.optimizer = tf.train.FtrlOptimizer(learning_rate=0.1, l1_regularization_strength=1e-3)
        self.training_op = self.optimizer.minimize(self.total_loss)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.sess.run(tf.local_variables_initializer())
        with self.sess.as_default():
            if self.task == "rating" or (self.task == "ranking" and not self.neg_sampling):
                for epoch in range(1, self.n_epochs + 1):
                    t0 = time.time()
                    n_batches = int(np.ceil(len(dataset.train_labels) / self.batch_size))
                    for n in range(n_batches):
                        end = min(len(dataset.train_labels), (n + 1) * self.batch_size)
                        indices_batch = dataset.train_feat_indices[n * self.batch_size: end]
                        values_batch = dataset.train_feat_values[n * self.batch_size: end]
                        labels_batch = dataset.train_labels[n * self.batch_size: end]

                        self.sess.run(self.training_op, feed_dict={self.feature_indices: indices_batch,
                                                                   self.feature_values: values_batch,
                                                                   self.labels: labels_batch})

                    if verbose == 1:
                        print("Epoch {}, training_time: {:.2f}".format(epoch, time.time() - t0))
                    elif verbose > 1:
                        print("Epoch {}, training_time: {:.2f}".format(epoch, time.time() - t0))
                        metrics = kwargs.get("metrics", self.metrics)
                        if hasattr(self, "sess"):
                            self.print_metrics_tf(dataset, epoch, **metrics)
                        else:
                            self.print_metrics(dataset, epoch, **metrics)
                        print()

            elif self.task == "ranking" and self.neg_sampling:
                for epoch in range(1, self.n_epochs + 1):
                    t0 = time.time()
                    neg = NegativeSamplingFeat(dataset, dataset.num_neg, self.batch_size, pre_sampling=pre_sampling)
                    n_batches = int(np.ceil(len(dataset.train_labels_implicit) / self.batch_size))
                    for n in range(n_batches):
                        indices_batch, values_batch, labels_batch = neg.next_batch()
                        self.sess.run(self.training_op, feed_dict={self.feature_indices: indices_batch,
                                                                   self.feature_values: values_batch,
                                                                   self.labels: labels_batch})

                    if verbose == 1:
                        print("Epoch {}, training_time: {:.2f}".format(epoch, time.time() - t0))
                    elif verbose > 1:
                        print("Epoch {}: training time: {:.4f}".format(epoch, time.time() - t0))
                        metrics = kwargs.get("metrics", self.metrics)
                        if hasattr(self, "sess"):
                            self.print_metrics_tf(dataset, epoch, **metrics)
                        else:
                            self.print_metrics(dataset, epoch, **metrics)
                        print()

    def predict(self, user, item):
        feat_indices, feat_value = self.get_predict_indices_and_values(self.dataset, user, item)
        try:
            target = self.pred if self.task == "rating" else self.y_prob
            pred = self.sess.run(target, feed_dict={self.feature_indices: feat_indices,
                                                    self.feature_values: feat_value})
            pred = pred.flatten()
            if self.lower_bound is not None and self.upper_bound is not None:
                pred = np.clip(pred, self.lower_bound, self.upper_bound) if self.task == "rating" else pred[0]
        except tf.errors.InvalidArgumentError:
            pred = self.dataset.global_mean if self.task == "rating" else 0.0
        return pred

    def recommend_user(self, u, n_rec):
        consumed = self.dataset.train_user[u]
        count = n_rec + len(consumed)
        target = self.pred if self.task == "rating" else self.y_prob

        feat_indices, feat_values = self.get_recommend_indices_and_values(self.dataset, u, self.total_items_unique)
        preds = self.sess.run(target, feed_dict={self.feature_indices: feat_indices,
                                                 self.feature_values: feat_values})

        preds = preds.flatten()
        ids = np.argpartition(preds, -count)[-count:]
        rank = sorted(zip(ids, preds[ids]), key=lambda x: -x[1])
        return list(itertools.islice((rec for rec in rank if rec[0] not in consumed), n_rec))

    @property
    def item_info(self):
        item_col = self.dataset.train_feat_indices.shape[1] - 1
        item_cols = [item_col] + self.dataset.item_feature_cols
        return np.unique(self.dataset.train_feat_indices[:, item_cols], axis=0)











