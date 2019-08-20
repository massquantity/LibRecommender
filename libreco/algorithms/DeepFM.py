import time
import itertools
import numpy as np
import tensorflow as tf
from .Base import BasePure, BaseFeat
from ..utils.sampling import NegativeSamplingFeat
from ..evaluate.evaluate import precision_tf, MAP_at_k, HitRatio_at_k, NDCG_at_k, MAR_at_k
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, auc


class DeepFMPure:
    def __init__(self, lr, embed_size=32, n_epochs=20, reg=0.0,
                 batch_size=64, dropout=0.0, seed=42):
        self.lr = lr
        self.embed_size = embed_size
        self.n_epochs = n_epochs
        self.reg = reg
        self.batch_size = batch_size
        self.dropout = dropout
        self.seed = seed

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
        self.w = tf.Variable(tf.truncated_normal([self.dim, 1], 0.0, 0.01))
        self.user_weights = tf.get_variable(initializer=tf.variance_scaling_initializer,
                                            shape=[self.n_users, self.embed_size],
                                            name="user_weights")
        self.item_weights = tf.get_variable(initializer=tf.variance_scaling_initializer,
                                            shape=[self.n_items, self.embed_size],
                                            name="item_weights")
        self.FM_embedding = tf.concat([self.user_weights, self.item_weights], axis=0)

        self.x = tf.sparse_placeholder(tf.float32, [None, self.dim])
        self.user_indices = tf.placeholder(tf.int32, shape=[None], name="user_indices")
        self.item_indices = tf.placeholder(tf.int32, shape=[None], name="item_indices")
        self.labels = tf.placeholder(tf.float32, shape=[None], name="labels")

        self.user_embedding = tf.nn.embedding_lookup(self.user_weights, self.user_indices)
        self.item_embedding = tf.nn.embedding_lookup(self.item_weights, self.item_indices)
        self.MLP_embedding = tf.concat([self.user_embedding, self.item_embedding], axis=1)

        self.linear_term = tf.sparse_tensor_dense_matmul(self.x, self.w)
        self.pairwise_term = 0.5 * tf.reduce_sum(
            tf.subtract(
                tf.square(tf.sparse_tensor_dense_matmul(self.x, self.FM_embedding)),
                tf.sparse_tensor_dense_matmul(tf.square(self.x), tf.square(self.FM_embedding))),
                axis=1, keepdims=True)

        self.MLP_layer1 = tf.layers.dense(inputs=self.MLP_embedding,
                                          units=2,
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.variance_scaling_initializer)
        self.MLP_layer1 = tf.layers.dropout(self.MLP_layer1, rate=self.dropout)
        self.MLP_layer_two = tf.layers.dense(inputs=self.MLP_layer1,
                                          units=2,
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.variance_scaling_initializer)
        self.MLP_layer_two = tf.layers.dropout(self.MLP_layer_two, rate=self.dropout)
        self.MLP_layer_three = tf.layers.dense(inputs=self.MLP_layer_two,
                                          units=2,
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.variance_scaling_initializer)
        self.MLP_layer_three = tf.layers.dropout(self.MLP_layer_three, rate=self.dropout)

        self.concat_layer = tf.concat([self.linear_term, self.pairwise_term, self.MLP_layer_three], axis=1)
        self.pred = tf.layers.dense(inputs=self.concat_layer, units=1, name="pred")

        self.loss = tf.losses.mean_squared_error(labels=tf.reshape(self.labels, [-1, 1]),
                                                 predictions=self.pred)
        self.rmse = tf.sqrt(tf.losses.mean_squared_error(labels=tf.reshape(self.labels, [-1, 1]),
                                                         predictions=tf.clip_by_value(self.pred, 1, 5)))


    def fit(self, dataset):
        self.build_model(dataset)
    #    self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.optimizer = tf.train.FtrlOptimizer(learning_rate=0.1, l1_regularization_strength=1e-3)
        self.training_op = self.optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        with self.sess.as_default():
            for epoch in range(1, self.n_epochs + 1):
                t0 = time.time()
                n_batches = len(dataset.train_labels) // self.batch_size
                for n in range(n_batches):
                    end = min(len(dataset.train_labels), (n + 1) * self.batch_size)
                    user_batch = dataset.train_user_indices[n * self.batch_size: end]
                    item_batch = dataset.train_item_indices[n * self.batch_size: end]
                    label_batch = dataset.train_labels[n * self.batch_size: end]

                    indices_batch, values_batch, shape_batch = DeepFMPure.build_sparse_data(dataset,
                                                                                        user_batch,
                                                                                        item_batch)
                    self.sess.run(self.training_op, feed_dict={self.x: tf.SparseTensorValue(indices_batch,
                                                                                            values_batch,
                                                                                            shape_batch),
                                                               self.user_indices: user_batch,
                                                               self.item_indices: item_batch,
                                                               self.labels: label_batch})
                if epoch % 1 == 0:
                    indices_train, values_train, shape_train = DeepFMPure.build_sparse_data(
                                                                    dataset,
                                                                    dataset.train_user_indices,
                                                                    dataset.train_item_indices)
                    train_rmse = self.sess.run(self.rmse, feed_dict={self.x: (indices_train,
                                                                              values_train,
                                                                              shape_train),
                                                                     self.user_indices: dataset.train_user_indices,
                                                                     self.item_indices: dataset.train_item_indices,
                                                                     self.labels: dataset.train_labels})

                    indices_test, values_test, shape_test = DeepFMPure.build_sparse_data(
                                                                dataset,
                                                                dataset.test_user_indices,
                                                                dataset.test_item_indices)
                    test_rmse = self.sess.run(self.rmse, feed_dict={self.x: (indices_test,
                                                                             values_test,
                                                                             shape_test),
                                                                    self.user_indices: dataset.test_user_indices,
                                                                    self.item_indices: dataset.test_item_indices,
                                                                    self.labels: dataset.test_labels})

                    print("Epoch {}, train_rmse: {:.4f}, training_time: {:.2f}".format(
                        epoch, train_rmse, time.time() - t0))
                    print("Epoch {}, test_rmse: {:.4f}".format(epoch, test_rmse))
                    print()

    def predict(self, u, i):
        index, value, shape = DeepFMPure.build_sparse_data(self.dataset, np.array([u]), np.array([i]))
        try:
            pred = self.sess.run(self.pred, feed_dict={self.x: (index, value, shape),
                                                       self.user_indices: np.array([u]),
                                                       self.item_indices: np.array([i])})
            pred = np.clip(pred, 1, 5)
        except tf.errors.InvalidArgumentError:
            pred = self.dataset.global_mean
        return pred


class DeepFMFeat(BaseFeat):
    def __init__(self, lr, embed_size=32, n_epochs=20, reg=0.0,
                 batch_size=64, dropout=0.0, seed=42, task="rating", neg_sampling=False):
        self.lr = lr
        self.embed_size = embed_size
        self.n_epochs = n_epochs
        self.reg = reg
        self.batch_size = batch_size
        self.dropout = dropout
        self.task = task
        self.seed = seed
        self.neg_sampling = neg_sampling

    def build_model(self, dataset):
        tf.set_random_seed(self.seed)
        self.dataset = dataset
        self.field_size = dataset.train_feat_indices.shape[1]
        self.feature_size = dataset.feature_size
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.total_items_unique = self.item_info

        self.feature_indices = tf.placeholder(tf.int32, shape=[None, self.field_size])
        self.feature_values = tf.placeholder(tf.float32, shape=[None, self.field_size])
        self.labels = tf.placeholder(tf.float32, shape=[None])

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
                                          units=self.embed_size * 2,   # self.embed_size * 2,
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.variance_scaling_initializer)
        self.MLP_layer_one = tf.layers.dropout(self.MLP_layer_one, rate=self.dropout)
        self.MLP_layer_two = tf.layers.dense(inputs=self.MLP_layer_one,
                                          units=self.embed_size,   # self.embed_size,
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.variance_scaling_initializer)
        self.MLP_layer_two = tf.layers.dropout(self.MLP_layer_two, rate=self.dropout)
        self.MLP_layer_three = tf.layers.dense(inputs=self.MLP_layer_two,
                                          units=self.embed_size,   # self.embed_size,
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.variance_scaling_initializer)
        self.MLP_layer_three = tf.layers.dropout(self.MLP_layer_three, rate=self.dropout)

        self.concat_layer = tf.concat([self.linear_term,
                                       self.pairwise_term,
                                       self.MLP_layer_three], axis=1)

        if self.task == "rating":
            self.pred = tf.layers.dense(inputs=self.concat_layer, units=1, name="pred")
            self.loss = tf.losses.mean_squared_error(labels=tf.reshape(self.labels, [-1, 1]),
                                                     predictions=self.pred)
            self.rmse = tf.sqrt(tf.losses.mean_squared_error(labels=tf.reshape(self.labels, [-1, 1]),
                                                             predictions=tf.clip_by_value(self.pred, 1, 5)))

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

    def fit(self, dataset, verbose=1, pre_sampling=True):
        self.build_model(dataset)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
    #   self.optimizer = tf.train.FtrlOptimizer(learning_rate=0.1, l1_regularization_strength=1e-3)
        self.training_op = self.optimizer.minimize(self.total_loss)
    #    init = tf.global_variables_initializer()
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess = tf.Session()
        self.sess.run(init)
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
                        test_loss, test_rmse = self.sess.run([self.total_loss, self.rmse],
                                                             feed_dict={
                                                                 self.feature_indices: dataset.test_feat_indices,
                                                                 self.feature_values: dataset.test_feat_values,
                                                                 self.labels: dataset.test_labels})

                        print("Epoch {}, training_time: {:.2f}".format(epoch, time.time() - t0))
                        print("Epoch {}, test_loss: {:.4f}, test_rmse: {:.4f}".format(
                            epoch, test_loss, test_rmse))
                        print()

                    elif verbose > 0 and self.task == "ranking":
                        print("Epoch {}: training time: {:.4f}".format(epoch, time.time() - t0))
                        t3 = time.time()
                        test_loss, test_accuracy, test_precision = \
                            self.sess.run([self.loss, self.accuracy, self.precision],
                                          feed_dict={self.feature_indices: dataset.test_feat_indices,
                                                     self.feature_values: dataset.test_feat_values,
                                                     self.labels: dataset.test_labels})

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
                        t3 = time.time()
                        test_loss, test_accuracy, test_precision, test_prob = \
                            self.sess.run([self.loss, self.accuracy, self.precision, self.y_prob],
                                feed_dict={self.feature_indices: dataset.test_indices_implicit,
                                           self.feature_values: dataset.test_values_implicit,
                                           self.labels: dataset.test_labels_implicit})

                        print("\ttest loss: {:.4f}".format(test_loss))
                        print("\ttest accuracy: {:.4f}".format(test_accuracy))
                        print("\ttest precision: {:.4f}".format(test_precision))
                        print("\tloss time: {:.4f}".format(time.time() - t3))

                        t1 = time.time()
                        test_auc = roc_auc_score(dataset.test_labels_implicit, test_prob)
                        test_ap = average_precision_score(dataset.test_labels_implicit, test_prob)
                        precision_test, recall_test, _ = precision_recall_curve(dataset.test_labels_implicit,
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

    def predict(self, user, item):
        feat_indices, feat_value = self.get_predict_indices_and_values(self.dataset, user, item)
        try:
            target = self.pred if self.task == "rating" else self.y_prob
            pred = self.sess.run(target, feed_dict={self.feature_indices: feat_indices,
                                                       self.feature_values: feat_value})
            pred = np.clip(pred, 1, 5) if self.task == "rating" else pred[0]
        except tf.errors.InvalidArgumentError:
            pred = self.dataset.global_mean
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