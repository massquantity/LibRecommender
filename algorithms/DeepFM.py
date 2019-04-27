import time
import numpy as np
import tensorflow as tf
from ..utils.negative_sampling import negative_sampling
from ..evaluate.evaluate import precision_tf, AP_at_k, MAP_at_k, HitRatio_at_k, NDCG_at_k


class DeepFM_435:
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
        self.ratings = tf.placeholder(tf.float32, shape=[None], name="ratings")

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
        self.MLP_layer2 = tf.layers.dense(inputs=self.MLP_layer1,
                                          units=2,
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.variance_scaling_initializer)
        self.MLP_layer2 = tf.layers.dropout(self.MLP_layer2, rate=self.dropout)
        self.MLP_layer3 = tf.layers.dense(inputs=self.MLP_layer2,
                                          units=2,
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.variance_scaling_initializer)
        self.MLP_layer3 = tf.layers.dropout(self.MLP_layer3, rate=self.dropout)

        self.concat_layer = tf.concat([self.linear_term, self.pairwise_term, self.MLP_layer3], axis=1)
        self.pred = tf.layers.dense(inputs=self.concat_layer, units=1, name="pred")

        self.loss = tf.losses.mean_squared_error(labels=tf.reshape(self.ratings, [-1, 1]),
                                                 predictions=self.pred)
        self.rmse = tf.sqrt(tf.losses.mean_squared_error(labels=tf.reshape(self.ratings, [-1, 1]),
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
                n_batches = len(dataset.train_ratings) // self.batch_size
                for n in range(n_batches):
                    end = min(len(dataset.train_ratings), (n + 1) * self.batch_size)
                    user_batch = dataset.train_user_indices[n * self.batch_size: end]
                    item_batch = dataset.train_item_indices[n * self.batch_size: end]
                    rating_batch = dataset.train_ratings[n * self.batch_size: end]

                    indices_batch, values_batch, shape_batch = DeepFM.build_sparse_data(dataset,
                                                                                        user_batch,
                                                                                        item_batch)
                    self.sess.run(self.training_op, feed_dict={self.x: tf.SparseTensorValue(indices_batch,
                                                                                            values_batch,
                                                                                            shape_batch),
                                                               self.user_indices: user_batch,
                                                               self.item_indices: item_batch,
                                                               self.ratings: rating_batch})
                if epoch % 1 == 0:
                    indices_train, values_train, shape_train = DeepFM.build_sparse_data(
                                                                    dataset,
                                                                    dataset.train_user_indices,
                                                                    dataset.train_item_indices)
                    train_rmse = self.sess.run(self.rmse, feed_dict={self.x: (indices_train,
                                                                              values_train,
                                                                              shape_train),
                                                                     self.user_indices: dataset.train_user_indices,
                                                                     self.item_indices: dataset.train_item_indices,
                                                                     self.ratings: dataset.train_ratings})

                    indices_test, values_test, shape_test = DeepFM.build_sparse_data(
                                                                dataset,
                                                                dataset.test_user_indices,
                                                                dataset.test_item_indices)
                    test_rmse = self.sess.run(self.rmse, feed_dict={self.x: (indices_test,
                                                                             values_test,
                                                                             shape_test),
                                                                    self.user_indices: dataset.test_user_indices,
                                                                    self.item_indices: dataset.test_item_indices,
                                                                    self.ratings: dataset.test_ratings})

                    print("Epoch {}, train_rmse: {:.4f}, training_time: {:.2f}".format(
                        epoch, train_rmse, time.time() - t0))
                    print("Epoch {}, test_rmse: {:.4f}".format(epoch, test_rmse))
                    print()

    def predict(self, u, i):
        index, value, shape = DeepFM.build_sparse_data(self.dataset, np.array([u]), np.array([i]))
        try:
            pred = self.sess.run(self.pred, feed_dict={self.x: (index, value, shape),
                                                       self.user_indices: np.array([u]),
                                                       self.item_indices: np.array([i])})
            pred = np.clip(pred, 1, 5)
        except tf.errors.InvalidArgumentError:
            pred = self.dataset.global_mean
        return pred


class DeepFM:
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

        self.x = tf.sparse_placeholder(tf.float32, [None, self.dim])
        self.user_indices = tf.placeholder(tf.int32, shape=[None], name="user_indices")
        self.item_indices = tf.placeholder(tf.int32, shape=[None], name="item_indices")
        self.ratings = tf.placeholder(tf.float32, shape=[None], name="ratings")

        #        self.w = tf.Variable(tf.truncated_normal([self.dim, 1], 0.0, 0.01))
        self.user_bias = tf.get_variable(initializer=tf.variance_scaling_initializer,
                                         shape=[self.n_users, 1],
                                         name="user_bias")
        self.item_bias = tf.get_variable(initializer=tf.variance_scaling_initializer,
                                         shape=[self.n_items, 1],
                                         name="item_bias")
        self.user_weights = tf.get_variable(initializer=tf.variance_scaling_initializer,
                                            shape=[self.n_users, self.embed_size],
                                            name="user_weights")
        self.item_weights = tf.get_variable(initializer=tf.variance_scaling_initializer,
                                            shape=[self.n_items, self.embed_size],
                                            name="item_weights")

        self.user_bias_embed = tf.nn.embedding_lookup(self.user_bias, self.user_indices)
        self.item_bias_embed = tf.nn.embedding_lookup(self.item_bias, self.item_indices)
        self.FM_embedding = tf.concat([self.user_weights, self.item_weights], axis=0)

        self.user_embedding = tf.nn.embedding_lookup(self.user_weights, self.user_indices)
        self.item_embedding = tf.nn.embedding_lookup(self.item_weights, self.item_indices)
        self.MLP_embedding = tf.concat([self.user_embedding, self.item_embedding], axis=1)

    #    self.linear_term = tf.sparse_tensor_dense_matmul(self.x, self.w)
        self.pairwise_term = tf.subtract(
                tf.square(tf.sparse_tensor_dense_matmul(self.x, self.FM_embedding)),
                tf.sparse_tensor_dense_matmul(tf.square(self.x), tf.square(self.FM_embedding)))

        self.MLP_layer1 = tf.layers.dense(inputs=self.MLP_embedding,
                                          units=200,   # self.embed_size * 2,
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.variance_scaling_initializer)
        self.MLP_layer1 = tf.layers.dropout(self.MLP_layer1, rate=self.dropout)
        self.MLP_layer2 = tf.layers.dense(inputs=self.MLP_layer1,
                                          units=200,   # self.embed_size,
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.variance_scaling_initializer)
        self.MLP_layer2 = tf.layers.dropout(self.MLP_layer2, rate=self.dropout)
        self.MLP_layer3 = tf.layers.dense(inputs=self.MLP_layer2,
                                          units=200,   # self.embed_size,
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.variance_scaling_initializer)
        self.MLP_layer3 = tf.layers.dropout(self.MLP_layer3, rate=self.dropout)

        self.concat_layer = tf.concat([self.user_bias_embed,
                                       self.item_bias_embed,
                                       self.pairwise_term,
                                       self.MLP_layer3], axis=1)
        self.pred = tf.layers.dense(inputs=self.concat_layer, units=1, name="pred")

        self.loss = tf.losses.mean_squared_error(labels=tf.reshape(self.ratings, [-1, 1]),
                                                 predictions=self.pred)
        self.rmse = tf.sqrt(tf.losses.mean_squared_error(labels=tf.reshape(self.ratings, [-1, 1]),
                                                         predictions=tf.clip_by_value(self.pred, 1, 5)))


    def fit(self, dataset):
        self.build_model(dataset)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
    #    self.optimizer = tf.train.FtrlOptimizer(learning_rate=0.1) # , l1_regularization_strength=1e-3
        self.training_op = self.optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        with self.sess.as_default():
            for epoch in range(1, self.n_epochs + 1):
                t0 = time.time()
                n_batches = len(dataset.train_ratings) // self.batch_size
                for n in range(n_batches):
                    end = min(len(dataset.train_ratings), (n + 1) * self.batch_size)
                    user_batch = dataset.train_user_indices[n * self.batch_size: end]
                    item_batch = dataset.train_item_indices[n * self.batch_size: end]
                    rating_batch = dataset.train_ratings[n * self.batch_size: end]

                    indices_batch, values_batch, shape_batch = DeepFM.build_sparse_data(dataset,
                                                                                        user_batch,
                                                                                        item_batch)
                    self.sess.run(self.training_op, feed_dict={self.x: tf.SparseTensorValue(indices_batch,
                                                                                            values_batch,
                                                                                            shape_batch),
                                                               self.user_indices: user_batch,
                                                               self.item_indices: item_batch,
                                                               self.ratings: rating_batch})
                if epoch % 1 == 0:
                    indices_train, values_train, shape_train = DeepFM.build_sparse_data(
                                                                    dataset,
                                                                    dataset.train_user_indices,
                                                                    dataset.train_item_indices)
                    train_rmse = self.sess.run(self.rmse, feed_dict={self.x: (indices_train,
                                                                              values_train,
                                                                              shape_train),
                                                                     self.user_indices: dataset.train_user_indices,
                                                                     self.item_indices: dataset.train_item_indices,
                                                                     self.ratings: dataset.train_ratings})

                    indices_test, values_test, shape_test = DeepFM.build_sparse_data(
                                                                dataset,
                                                                dataset.test_user_indices,
                                                                dataset.test_item_indices)
                    test_rmse = self.sess.run(self.rmse, feed_dict={self.x: (indices_test,
                                                                             values_test,
                                                                             shape_test),
                                                                    self.user_indices: dataset.test_user_indices,
                                                                    self.item_indices: dataset.test_item_indices,
                                                                    self.ratings: dataset.test_ratings})

                    print("Epoch {}, train_rmse: {:.4f}, training_time: {:.2f}".format(
                        epoch, train_rmse, time.time() - t0))
                    print("Epoch {}, test_rmse: {:.4f}".format(epoch, test_rmse))
                    print()

    def predict(self, u, i):
        index, value, shape = DeepFM.build_sparse_data(self.dataset, np.array([u]), np.array([i]))
        try:
            pred = self.sess.run(self.pred, feed_dict={self.x: (index, value, shape),
                                                       self.user_indices: np.array([u]),
                                                       self.item_indices: np.array([i])})
            pred = np.clip(pred, 1, 5)
        except tf.errors.InvalidArgumentError:
            pred = self.dataset.global_mean
        return pred