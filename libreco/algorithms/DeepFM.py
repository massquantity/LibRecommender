"""

Reference: Huifeng Guo et al. "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction"
                              (https://arxiv.org/pdf/1703.04247.pdf)

author: massquantity

"""
import time
import itertools
import numpy as np
import tensorflow as tf
from .Base import BasePure, BaseFeat
from ..utils.sampling import NegativeSamplingFeat, NegativeSampling
from ..evaluate.evaluate import precision_tf, MAP_at_k, recall_at_k, NDCG_at_k, MAR_at_k
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, auc


class DeepFmPure(BasePure):
    def __init__(self, lr, embed_size=32, n_epochs=20, reg=0.0, batch_size=64,
                 dropout_rate=0.0, seed=42, task="rating", neg_sampling=False, network_size=None):
        self.lr = lr
        self.embed_size = embed_size
        self.n_epochs = n_epochs
        self.reg = reg
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.task = task
        self.neg_sampling = neg_sampling
        if network_size is not None:
            assert len(network_size) == 3, "network size does not match true model size"
            self.network_size = network_size
        else:
            self.network_size = [embed_size * 2, embed_size * 2, embed_size]
        super(DeepFmPure, self).__init__()

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

        self.x = tf.sparse_placeholder(tf.float32, [None, self.dim])
        self.user_indices = tf.placeholder(tf.int32, shape=[None], name="user_indices")
        self.item_indices = tf.placeholder(tf.int32, shape=[None], name="item_indices")
        self.labels = tf.placeholder(tf.float32, shape=[None], name="labels")
        self.dropout_switch = tf.placeholder_with_default(False, shape=[], name="training")

        self.w = tf.Variable(tf.truncated_normal([self.dim, 1], 0.0, 0.01))
        self.user_weights = tf.get_variable(initializer=tf.variance_scaling_initializer,
                                            shape=[self.n_users, self.embed_size],
                                            name="user_weights")
        self.item_weights = tf.get_variable(initializer=tf.variance_scaling_initializer,
                                            shape=[self.n_items, self.embed_size],
                                            name="item_weights")
        self.FM_embedding = tf.concat([self.user_weights, self.item_weights], axis=0)

        self.linear_term = tf.sparse_tensor_dense_matmul(self.x, self.w)
        self.pairwise_term = 0.5 * tf.reduce_sum(
            tf.subtract(
                tf.square(tf.sparse_tensor_dense_matmul(self.x, self.FM_embedding)),
                tf.sparse_tensor_dense_matmul(tf.square(self.x), tf.square(self.FM_embedding))),
            axis=1, keepdims=True)

        self.user_embedding = tf.nn.embedding_lookup(self.user_weights, self.user_indices)
        self.item_embedding = tf.nn.embedding_lookup(self.item_weights, self.item_indices)
        self.MLP_embedding = tf.concat([self.user_embedding, self.item_embedding], axis=1)

        self.MLP_layer_one = tf.layers.dense(inputs=self.MLP_embedding,
                                             units=self.network_size[0],
                                             activation=tf.nn.relu,
                                             kernel_initializer=tf.variance_scaling_initializer)
        self.MLP_layer_one = tf.layers.dropout(self.MLP_layer_one, rate=self.dropout_rate, training=self.dropout_switch)
        self.MLP_layer_two = tf.layers.dense(inputs=self.MLP_layer_one,
                                             units=self.network_size[1],
                                             activation=tf.nn.relu,
                                             kernel_initializer=tf.variance_scaling_initializer)
        self.MLP_layer_two = tf.layers.dropout(self.MLP_layer_two, rate=self.dropout_rate, training=self.dropout_switch)
        self.MLP_layer_three = tf.layers.dense(inputs=self.MLP_layer_two,
                                               units=self.network_size[2],
                                               activation=tf.nn.relu,
                                               kernel_initializer=tf.variance_scaling_initializer)

        self.concat_layer = tf.concat([self.linear_term, self.pairwise_term, self.MLP_layer_three], axis=1)

        if self.task == "rating":
            self.pred = tf.layers.dense(inputs=self.concat_layer, units=1, name="pred")
            self.loss = tf.losses.mean_squared_error(labels=tf.reshape(self.labels, [-1, 1]),
                                                     predictions=self.pred)

            if self.lower_bound is not None and self.upper_bound is not None:
                self.rmse = tf.sqrt(tf.losses.mean_squared_error(labels=self.labels,
                                                                 predictions=tf.clip_by_value(self.pred,
                                                                                              self.lower_bound,
                                                                                              self.upper_bound)))
            else:
                self.rmse = self.loss

        elif self.task == "ranking":
            self.logits = tf.layers.dense(inputs=self.concat_layer, units=1, name="logits")
            self.logits = tf.reshape(self.logits, [-1])
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
            self.y_prob = tf.sigmoid(self.logits)
            self.pred = tf.where(self.y_prob >= 0.5,
                                 tf.fill(tf.shape(self.logits), 1.0),
                                 tf.fill(tf.shape(self.logits), 0.0))
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.labels), tf.float32))
            self.precision = precision_tf(self.pred, self.labels)

    def fit(self, dataset, verbose=1, **kwargs):
        self.build_model(dataset)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        #    self.optimizer = tf.train.FtrlOptimizer(learning_rate=0.1, l1_regularization_strength=1e-3)
        self.training_op = self.optimizer.minimize(self.loss)
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
                        user_batch = dataset.train_user_indices[n * self.batch_size: end]
                        item_batch = dataset.train_item_indices[n * self.batch_size: end]
                        label_batch = dataset.train_labels[n * self.batch_size: end]

                        indices_batch, values_batch, shape_batch = DeepFmPure.build_sparse_data(dataset,
                                                                                                user_batch,
                                                                                                item_batch)
                        self.sess.run(self.training_op, feed_dict={self.x: tf.SparseTensorValue(indices_batch,
                                                                                                values_batch,
                                                                                                shape_batch),
                                                                   self.labels: label_batch,
                                                                   self.user_indices: user_batch,
                                                                   self.item_indices: item_batch,
                                                                   self.dropout_switch: True})

                    if verbose > 0:
                        print("Epoch {}, training time: {:.2f}".format(epoch, time.time() - t0))
                        indices_test, values_test, shape_test = DeepFmPure.build_sparse_data(
                            dataset,
                            dataset.test_user_indices,
                            dataset.test_item_indices)
                        test_loss, test_rmse = self.sess.run([self.loss, self.rmse],
                                                             feed_dict={self.x: (indices_test,
                                                                                 values_test,
                                                                                 shape_test),
                                                                        self.labels: dataset.test_labels,
                                                                        self.user_indices: dataset.test_user_indices,
                                                                        self.item_indices: dataset.test_item_indices})

                        print("Epoch {}, test_loss: {:.4f}, test_rmse: {:.4f}".format(
                            epoch, test_loss, test_rmse))

            elif self.task == "ranking":
                for epoch in range(1, self.n_epochs + 1):
                    t0 = time.time()
                    neg = NegativeSampling(dataset, dataset.num_neg, self.batch_size)
                    n_batches = int(np.ceil(len(dataset.train_labels) / self.batch_size))
                    for n in range(n_batches):
                        user_batch, item_batch, label_batch = neg.next_batch()
                        indices_batch, values_batch, shape_batch = DeepFmPure.build_sparse_data(dataset,
                                                                                                user_batch,
                                                                                                item_batch)
                        self.sess.run(self.training_op, feed_dict={self.x: tf.SparseTensorValue(indices_batch,
                                                                                                values_batch,
                                                                                                shape_batch),
                                                                   self.labels: label_batch,
                                                                   self.user_indices: user_batch,
                                                                   self.item_indices: item_batch,
                                                                   self.dropout_switch: True})

                    if verbose > 0:
                        indices_test, values_test, shape_test = DeepFmPure.build_sparse_data(
                            dataset,
                            dataset.test_user_implicit,
                            dataset.test_item_implicit)
                        test_loss, test_accuracy, test_precision = \
                            self.sess.run([self.loss, self.accuracy, self.precision],
                                          feed_dict={self.x: (indices_test, values_test, shape_test),
                                                     self.labels: dataset.test_label_implicit,
                                                     self.user_indices: dataset.test_user_implicit,
                                                     self.item_indices: dataset.test_item_implicit})

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
        index, value, shape = DeepFmPure.build_sparse_data(self.dataset, np.array([u]), np.array([i]))
        try:
            pred = self.sess.run(self.pred, feed_dict={self.x: (index, value, shape),
                                                       self.user_indices: [u],
                                                       self.item_indices: [i]})
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
        index, value, shape = DeepFmPure.build_sparse_data(self.dataset, user_indices, item_indices)
        preds = self.sess.run(target, feed_dict={self.x: tf.SparseTensorValue(index, value, shape),
                                                 self.user_indices: user_indices,
                                                 self.item_indices: item_indices})

        ids = np.argpartition(preds, -count)[-count:]
        rank = sorted(zip(ids, preds[ids]), key=lambda x: -x[1])
        return list(itertools.islice((rec for rec in rank if rec[0] not in consumed), n_rec))


class DeepFmFeat(BaseFeat):
    def __init__(self, lr, embed_size=32, n_epochs=20, reg=0.0, batch_size=64,
                 dropout_rate=0.0, seed=42, task="rating", neg_sampling=False, network_size=None):
        self.lr = lr
        self.embed_size = embed_size
        self.n_epochs = n_epochs
        self.reg = reg
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.task = task
        self.seed = seed
        self.neg_sampling = neg_sampling
        if network_size is not None:
            assert len(network_size) == 3, "network size does not match true model size"
            self.network_size = network_size
        else:
            self.network_size = [embed_size * 2, embed_size * 2, embed_size]
        super(DeepFmFeat, self).__init__()

    def build_model(self, dataset):
        tf.set_random_seed(self.seed)
        self.dataset = dataset
        self.field_size = dataset.train_feat_indices.shape[1]
        self.feature_size = dataset.feature_size
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
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
        self.dropout_switch = tf.placeholder_with_default(False, shape=[], name="training")

        self.linear_features = tf.get_variable(initializer=tf.variance_scaling_initializer,
                                               shape=[self.feature_size + 1, 1],
                                               name="linear_features")
        self.pairwise_features = tf.get_variable(initializer=tf.variance_scaling_initializer,
                                                 shape=[self.feature_size + 1, self.embed_size],
                                                 name="pairwise_features")
        self.feature_values_reshape = tf.reshape(self.feature_values, shape=[-1, self.field_size, 1])

        self.linear_embedding = tf.nn.embedding_lookup(self.linear_features, self.feature_indices)  # N * F * 1
        self.linear_term = tf.reduce_sum(tf.multiply(self.linear_embedding, self.feature_values_reshape), 2)

        self.feature_embedding = tf.nn.embedding_lookup(self.pairwise_features, self.feature_indices)  # N * F * K
        self.feature_embedding = tf.multiply(self.feature_embedding, self.feature_values_reshape)

        self.pairwise_term = 0.5 * tf.subtract(
            tf.square(tf.reduce_sum(self.feature_embedding, axis=2)),  # axis=1 ?
            tf.reduce_sum(tf.square(self.feature_embedding), axis=2))

        self.MLP_embedding = tf.reduce_sum(self.feature_embedding, axis=1)  # axis=1 ?
    #    self.MLP_embedding = tf.reshape(self.feature_embedding, [-1, self.field_size * self.embed_size])  # N * (F * K)
        self.MLP_layer_one = tf.layers.dense(inputs=self.MLP_embedding,
                                             units=self.network_size[0],
                                             activation=tf.nn.relu,
                                             kernel_initializer=tf.variance_scaling_initializer)
        self.MLP_layer_one = tf.layers.dropout(self.MLP_layer_one, rate=self.dropout_rate, training=self.dropout_switch)
        self.MLP_layer_two = tf.layers.dense(inputs=self.MLP_layer_one,
                                             units=self.network_size[1],
                                             activation=tf.nn.relu,
                                             kernel_initializer=tf.variance_scaling_initializer)
        self.MLP_layer_two = tf.layers.dropout(self.MLP_layer_two, rate=self.dropout_rate, training=self.dropout_switch)
        self.MLP_layer_three = tf.layers.dense(inputs=self.MLP_layer_two,
                                               units=self.network_size[2],
                                               activation=tf.nn.relu,
                                               kernel_initializer=tf.variance_scaling_initializer)
        #    self.MLP_layer_three = tf.layers.dropout(self.MLP_layer_three, rate=self.dropout)

        self.concat_layer = tf.concat([self.linear_term,
                                       self.pairwise_term,
                                       self.MLP_layer_three], axis=1)

        #    self.concat_layer = self.MLP_layer_three

        if self.task == "rating":
            self.pred = tf.layers.dense(inputs=self.concat_layer, units=1, name="pred")
            self.loss = tf.losses.mean_squared_error(labels=tf.reshape(self.labels, [-1, 1]),
                                                     predictions=self.pred)
            self.rmse = tf.sqrt(tf.losses.mean_squared_error(labels=tf.reshape(self.labels, [-1, 1]),
                                                             predictions=tf.clip_by_value(self.pred, 1, 5)))

            if self.lower_bound is not None and self.upper_bound is not None:
                self.rmse = tf.sqrt(tf.losses.mean_squared_error(labels=self.labels,
                                predictions=tf.clip_by_value(self.pred, self.lower_bound, self.upper_bound)))
            else:
                self.rmse = self.loss

            reg_features = self.reg * tf.nn.l2_loss(self.pairwise_features)
            self.total_loss = tf.add_n([self.loss, reg_features])

        elif self.task == "ranking":
            self.logits = tf.layers.dense(inputs=self.concat_layer, units=1, name="logits")
            self.logits = tf.reshape(self.logits, [-1])
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits))

            self.y_prob = tf.sigmoid(self.logits)
            self.pred = tf.where(self.y_prob >= 0.5,
                                 tf.fill(tf.shape(self.logits), 1.0),
                                 tf.fill(tf.shape(self.logits), 0.0))

            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.labels), tf.float32))
            self.precision = precision_tf(self.pred, self.labels)

            reg_features = self.reg * tf.nn.l2_loss(self.pairwise_features)
            self.total_loss = tf.add_n([self.loss, reg_features])

    def fit(self, dataset, verbose=1, pre_sampling=True, **kwargs):
        self.build_model(dataset)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
    #    self.optimizer = tf.train.FtrlOptimizer(learning_rate=0.1, l1_regularization_strength=1e-3)
        self.training_op = self.optimizer.minimize(self.total_loss)
    #    init = tf.global_variables_initializer()
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess = tf.Session()
        self.sess.run(init)
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
                                                                   self.labels: labels_batch,
                                                                   self.dropout_switch: True})

                    if verbose > 0 and self.task == "rating":
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

                    if verbose > 0:
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

        ids = np.argpartition(preds, -count)[-count:]
        rank = sorted(zip(ids, preds[ids]), key=lambda x: -x[1])
        return list(itertools.islice((rec for rec in rank if rec[0] not in consumed), n_rec))

    @property
    def item_info(self):
        item_col = self.dataset.train_feat_indices.shape[1] - 1
        item_cols = [item_col] + self.dataset.item_feature_cols
        return np.unique(self.dataset.train_feat_indices[:, item_cols], axis=0)
