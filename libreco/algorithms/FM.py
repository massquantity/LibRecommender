import os
import time
import itertools
import operator
from collections import OrderedDict
import numpy as np
import tensorflow as tf
from ..evaluate.evaluate import precision_tf, MAP_at_k, MAR_at_k, NDCG_at_k
from ..utils.sampling import NegativeSampling, NegativeSamplingFeat


class FmPure:
    def __init__(self, lr, n_epochs=20, n_factors=100, reg=0.0, batch_size=256, seed=42, task="rating"):
        self.lr = lr
        self.n_epochs = n_epochs
        self.n_factors = n_factors
        self.reg = reg
        self.batch_size = batch_size
        self.seed = seed
        self.task = task

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
        self.dim = dataset.n_users + dataset.n_items
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

            self.rmse = tf.sqrt(tf.losses.mean_squared_error(labels=tf.reshape(self.labels, [-1, 1]),
                                                             predictions=tf.clip_by_value(self.pred, 1, 5)))

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

    def fit(self, dataset, verbose=1):
        self.build_model(dataset)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
    #    self.optimizer = tf.train.FtrlOptimizer(learning_rate=0.1, l1_regularization_strength=1e-3)
        self.training_op = self.optimizer.minimize(self.total_loss)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.sess.run(tf.local_variables_initializer())
        with self.sess.as_default():
            if self.task == "rating":
                for epoch in range(1, self.n_epochs + 1):
                    t0 = time.time()
                    n_batches = len(dataset.train_labels) // self.batch_size
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
                        indices_train, values_train, shape_train = FmPure.build_sparse_data(
                                                                        dataset,
                                                                        dataset.train_user_indices,
                                                                        dataset.train_item_indices)
                        train_rmse = self.sess.run(self.rmse, feed_dict={self.x: (indices_train,
                                                                                  values_train,
                                                                                  shape_train),
                                                                         self.labels: dataset.train_labels})

                        indices_test, values_test, shape_test = FmPure.build_sparse_data(
                                                                    dataset,
                                                                    dataset.test_user_indices,
                                                                    dataset.test_item_indices)
                        test_rmse = self.sess.run(self.rmse, feed_dict={self.x: (indices_test,
                                                                                 values_test,
                                                                                 shape_test),
                                                                        self.labels: dataset.test_labels})

                        print("Epoch {}, training time: {:.2f}".format(epoch, time.time() - t0))
                        print("Epoch {}, train rmse: {:.4f}".format(epoch, train_rmse))
                        print("Epoch {}, test rmse: {:.4f}".format(epoch, test_rmse))
                        print()

            elif self.task == "ranking":
                for epoch in range(1, self.n_epochs + 1):
                    t0 = time.time()
                    neg = NegativeSampling(dataset, dataset.num_neg, self.batch_size)
                    n_batches = len(dataset.train_user_implicit) // self.batch_size
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
                        indices_train, values_train, shape_train = FmPure.build_sparse_data(
                                                                       dataset,
                                                                       dataset.train_user_implicit,
                                                                       dataset.train_item_implicit)
                        train_loss, train_accuracy, train_precision = \
                            self.sess.run([self.loss, self.accuracy, self.precision],
                                          feed_dict={self.x: (indices_train, values_train, shape_train),
                                                     self.labels: dataset.train_label_implicit})

                        indices_test, values_test, shape_test = FmPure.build_sparse_data(
                                                                    dataset,
                                                                    dataset.test_user_implicit,
                                                                    dataset.test_item_implicit)
                        test_loss, test_accuracy, test_precision = \
                            self.sess.run([self.loss, self.accuracy, self.precision],
                                          feed_dict={self.x: (indices_test, values_test, shape_test),
                                                     self.labels: dataset.test_label_implicit})

                        print("Epoch {}, training time: {:.2f}".format(epoch, time.time() - t0))
                        print("Epoch {}, train loss: {:.4f}, train accuracy: {:.4f}, train precision: {:.4f}".format(
                                epoch, train_loss, train_accuracy, train_precision))
                        print("Epoch {}, test loss: {:.4f}, test accuracy: {:.4f}, test precision: {:.4f}".format(
                                epoch, test_loss, test_accuracy, test_precision))
                        print()


    def predict(self, u, i):
        index, value, shape = FmPure.build_sparse_data(self.dataset, np.array([u]), np.array([i]))
        try:
            pred = self.sess.run(self.pred, feed_dict={self.x: (index, value, shape)})
            pred = np.clip(pred, 1, 5)
        except tf.errors.InvalidArgumentError:
            pred = self.dataset.global_mean
        return pred


class FmFeat:
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

    def build_model(self, dataset):
        tf.set_random_seed(self.seed)
        self.dataset = dataset
        self.field_size = dataset.train_feat_indices.shape[1]
        self.feature_size = dataset.feature_size
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items

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
            tf.square(tf.reduce_sum(self.feature_embedding, axis=1)),
            tf.reduce_sum(tf.square(self.feature_embedding), axis=1))

        self.concat = tf.concat([self.linear_term, self.pairwise_term], axis=1)

        if self.task == "rating":
            self.pred = tf.layers.dense(inputs=self.concat, units=1)
            self.loss = tf.losses.mean_squared_error(labels=tf.reshape(self.labels, [-1, 1]),
                                                     predictions=self.pred)
            self.rmse = tf.sqrt(tf.losses.mean_squared_error(labels=tf.reshape(self.labels, [-1, 1]),
                                                             predictions=tf.clip_by_value(self.pred, 1, 5)))

        #    reg_w = self.reg * tf.nn.l2_loss(self.w)
            reg_v = self.reg * tf.nn.l2_loss(self.v)
            self.total_loss = tf.add_n([self.loss, reg_v])  # reg_w

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

    def fit(self, dataset, verbose=1, pre_sampling=True):
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
                    n_batches = len(dataset.train_labels) // self.batch_size
                    for n in range(n_batches):
                        end = min(len(dataset.train_labels), (n + 1) * self.batch_size)
                        indices_batch = dataset.train_feat_indices[n * self.batch_size: end]
                        values_batch = dataset.train_feat_values[n * self.batch_size: end]
                        labels_batch = dataset.train_labels[n * self.batch_size: end]

                        self.sess.run(self.training_op, feed_dict={self.feature_indices: indices_batch,
                                                                   self.feature_values: values_batch,
                                                                   self.labels: labels_batch})

                    if verbose > 0 and self.task == "rating":
               #         train_rmse = self.rmse.eval(feed_dict={self.feature_indices: dataset.train_feat_indices,
                #                                               self.feature_values: dataset.train_feat_values,
                #                                               self.labels: dataset.train_labels})

                        test_loss, test_rmse = self.sess.run([self.total_loss, self.rmse],
                                                              feed_dict={
                                                                  self.feature_indices: dataset.test_feat_indices,
                                                                  self.feature_values: dataset.test_feat_values,
                                                                  self.labels: dataset.test_labels})

                #        print("Epoch {}, train_rmse: {:.4f}, training_time: {:.2f}".format(
                #                epoch, train_rmse, time.time() - t0))
                        print("Epoch {}, training_time: {:.2f}".format(epoch, time.time() - t0))
                        print("Epoch {}, test_loss: {:.4f}, test_rmse: {:.4f}".format(
                            epoch, test_loss, test_rmse))
                        print()

                    elif verbose > 0 and self.task == "ranking":
                        pass
                        print()

            elif self.task == "ranking" and self.neg_sampling:
                for epoch in range(1, self.n_epochs + 1):
                    t0 = time.time()
                    neg = NegativeSamplingFeat(dataset, dataset.num_neg, self.batch_size, pre_sampling=pre_sampling)
                    n_batches = int(np.ceil(len(dataset.train_labels_implicit) / self.batch_size))
                #    n_batches = len(dataset.train_indices_implicit) // self.batch_size
                    for n in range(n_batches):
                        indices_batch, values_batch, labels_batch = neg.next_batch()
                        self.sess.run(self.training_op, feed_dict={self.feature_indices: indices_batch,
                                                                   self.feature_values: values_batch,
                                                                   self.labels: labels_batch})

                    if verbose > 0:
                        print("Epoch {}: training time: {:.4f}".format(epoch, time.time() - t0))
                        t3 = time.time()
                        test_loss, test_accuracy, test_precision = \
                            self.sess.run([self.loss, self.accuracy, self.precision],
                                feed_dict={self.feature_indices: dataset.test_indices_implicit,
                                           self.feature_values: dataset.test_values_implicit,
                                           self.labels: dataset.test_labels_implicit})

                        print("\ttest loss: {:.4f}".format(test_loss))
                        print("\ttest accuracy: {:.4f}".format(test_accuracy))
                        print("\ttest precision: {:.4f}".format(test_precision))
                        print("\tloss time: {:.4f}".format(time.time() - t3))

                        t4 = time.time()
                        mean_average_precision_10 = MAP_at_k(self, self.dataset, 10, sample_user=1000)
                        print("\t MAP @ {}: {:.4f}".format(10, mean_average_precision_10))
                        print("\t MAP @ 10 time: {:.4f}".format(time.time() - t4))

                        t6 = time.time()
                        mean_average_recall_50 = MAR_at_k(self, self.dataset, 50, sample_user=1000)
                        print("\t MAR @ {}: {:.4f}".format(50, mean_average_recall_50))
                        print("\t MAR @ 50 time: {:.4f}".format(time.time() - t6))

                        t9 = time.time()
                        NDCG = NDCG_at_k(self, self.dataset, 10, sample_user=1000)
                        print("\t NDCG @ {}: {:.4f}".format(10, NDCG))
                        print("\t NDCG time: {:.4f}".format(time.time() - t9))
                        print()

                        '''
                        test_pred, test_loss, test_accuracy, test_precision = \
                            self.sess.run([self.pred, self.loss, self.accuracy, self.precision],
                                          feed_dict={self.feature_indices: dataset.test_indices_implicit,
                                                     self.feature_values: dataset.test_values_implicit,
                                                     self.labels: dataset.test_labels_implicit})

                        print("Epoch {}, training time: {:.2f}".format(epoch, time.time() - t0))
                #        print("Epoch {}, train loss: {:.4f}, train accuracy: {:.4f}, train precision: {:.4f}".format(
                #            epoch, train_loss, train_accuracy, train_precision))
                        print("Epoch {}, test loss: {:.4f}, test accuracy: {:.4f}, test precision: {:.4f}".format(
                            epoch, test_loss, test_accuracy, test_precision))
                        '''
                        print()

    def predict(self, feat_ind, feat_val):
        try:
            pred = self.sess.run(self.pred, feed_dict={self.feature_indices: feat_ind,
                                                       self.feature_values: feat_val})
            pred = np.clip(pred, 1, 5)
        except tf.errors.InvalidArgumentError:
            pred = self.dataset.global_mean
        return pred

    def recommend_user(self, u, n_rec):
        user_repr = u + self.dataset.user_offset
        user_cols = self.dataset.user_feature_cols + [-2]  # -2 is user col
        user_features = self.dataset.train_feat_indices[:, user_cols]
        user = user_features[user_features[:, -1] == user_repr][0]
        user = user[:-1]
    #    print("user: ", user)
        user_reprs = np.tile(user_repr, (self.dataset.n_items, 1))
        users = np.tile(user, (self.dataset.n_items, 1))

    #   np.unique is sorted from starting with the first element, so put item col first
        item_cols = [-1] + self.dataset.item_feature_cols
        total_items_unique = np.unique(self.dataset.train_feat_indices[:, item_cols], axis=0)
    #    print("unique items: ", len(total_items_unique))
        item_reprs = np.expand_dims(total_items_unique[:, 0], -1)
        items = np.delete(total_items_unique, 0, axis=1)

        orig_cols = self.dataset.user_feature_cols + self.dataset.item_feature_cols
        col_reindex = np.array(range(len(orig_cols)))[np.argsort(orig_cols)]

        concat_indices = np.concatenate([users, items], axis=-1)[:, col_reindex]
        concat_indices = np.concatenate([concat_indices, user_reprs], axis=-1)
        concat_indices = np.concatenate([concat_indices, item_reprs], axis=-1)
    #    print(concat_indices[:5])

        if self.dataset.numerical_col is not None:
            numerical_dict = OrderedDict()
            for i in range(len(self.dataset.numerical_col)):
                numerical_map = dict(sorted(zip(self.dataset.train_feat_indices[:, -1],
                                           self.dataset.train_feat_values[:, i]), key=lambda x: x[0]))
                numerical_dict[i] = [v for v in numerical_map.values()]

            print("item repr: ", item_reprs)
            print(numerical_map)
            item_values = [v for v in numerical_map.values()]
            print(item_values)

            feat_values = np.ones(shape=(self.dataset.n_items, concat_indices.shape[1]))
            for k, v in numerical_dict.items():
                feat_values[:, k] = v


    #    feat_values = np.ones(shape=(self.dataset.n_items, concat_indices.shape[1]))
    #    if self.dataset.numerical_col is not None:
    #        for i, col in enumerate(self.dataset.numerical_col):
    #            feat_values[:, i] = self.dataset.train_feat_values[:, col]


        consumed = self.dataset.train_user[u]
        count = n_rec + len(consumed)
        target = self.pred if self.task == "rating" else self.y_prob

        preds = self.sess.run(target, feed_dict={self.feature_indices: concat_indices,
                                                 self.feature_values: feat_values})
        ids = np.argpartition(preds, -count)[-count:]
        rank = sorted(zip(ids, preds[ids]), key=lambda x: -x[1])
        return list(itertools.islice((rec for rec in rank if rec[0] not in consumed), n_rec))

    @property
    def item_info(self):
    #    concat_indices
        pass























