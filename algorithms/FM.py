import time
import numpy as np
import tensorflow as tf


class FM:
    def __init__(self, lr, n_epochs=20, n_factors=100, reg=0.0, batch_size=64, seed=42):
        self.lr = lr
        self.n_epochs = n_epochs
        self.n_factors = n_factors  ########## 8
        self.reg = reg
        self.batch_size = batch_size
        self.seed = seed

    @staticmethod
    def build_sparse(data, user_indices, item_indices):
        first_dim = tf.reshape(tf.constant(np.tile(np.arange(user_indices.shape[0])), 2), [-1, 1])
        second_dim = tf.reshape(tf.concat([user_indices, item_indices + data.n_users], axis=0), [-1, 1])
        indices = tf.concat([first_dim, tf.cast(second_dim, tf.int64)], axis=1)
        values = tf.ones(user_indices.shape[0] * 2)
        shape = [user_indices.shape[0], data.n_users + data.n_items]
        return tf.sparse.SparseTensor(indices, values, shape)

    def build_model(self, dataset):
        tf.set_random_seed(self.seed)
        self.dataset = dataset
        self.dim = dataset.n_users + dataset.n_items
        self.user_indices = tf.placeholder(tf.int32, shape=[None], name="user_indices")
        self.item_indices = tf.placeholder(tf.int32, shape=[None], name="item_indices")
        self.ratings = tf.placeholder(tf.int32, shape=[None], name="ratings")

        self.w = tf.Variable(tf.truncated_normal([self.dim], 0.0, 0.1))
        self.v = tf.Variable(tf.truncated_normal([self.dim, self.n_factors], 0.0, 0.1))

    #    self.user_onehot = tf.one_hot(self.user_indices, dataset.n_users)
    #    self.item_onehot = tf.one_hot(self.item_indices, dataset.n_items)
    #    self.x = tf.concat([self.user_onehot, self.item_onehot], axis=1)


        self.x = FM.build_sparse(dataset, self.user_indices, self.item_indices)
        self.w = tf.reshape(self.w, [-1, 1])
    #    self.linear_term = tf.reduce_sum(tf.multiply(self.w, self.x), axis=1, keepdims=True)
        self.linear_term = tf.sparse_tensor_dense_matmul(self.x, self.w)
        print("linear", self.linear_term.shape)
    #    self.linear_term = tf.reduce_sum(self.x.__mul__(self.w), axis=1, keepdims=True)
        self.pairwise_term = 0.5 * tf.reduce_sum(
            tf.subtract(
                tf.square(tf.sparse_tensor_dense_matmul(self.x, self.v)),
                tf.sparse_tensor_dense_matmul(tf.square(self.x), tf.square(self.v))), axis=1, keepdims=True)

        self.pred = tf.add(self.linear_term, self.pairwise_term)
        self.loss = tf.losses.mean_squared_error(labels=tf.reshape(self.ratings, [-1,1]),
                                                 predictions=self.pred)

        self.rmse = tf.sqrt(tf.losses.mean_squared_error(labels=tf.reshape(self.ratings, [-1,1]),
                                                 predictions=tf.clip_by_value(self.pred, 1, 5)))

    #    self.metrics = tf.metrics.root_mean_squared_error(labels=tf.reshape(tf.cast(self.ratings, tf.float32), [-1,1]),
    #                                                      predictions=self.pred)
                                                        # tf.clip_by_value(self.pred, 1, 5)
                                                        # labels=tf.reshape(self.ratings, [-1,1]),

        reg_w = self.reg * tf.nn.l2_loss(self.w)
        reg_v = self.reg * tf.nn.l2_loss(self.v)
        self.total_loss = tf.add_n([self.loss, reg_w, reg_v])

    def fit(self, dataset):
        self.build_model(dataset)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
    #    self.optimizer = tf.train.FtrlOptimizer(learning_rate=0.1)
        self.training_op = self.optimizer.minimize(self.total_loss)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.sess.run(tf.local_variables_initializer())  ######
        with self.sess.as_default():
            for epoch in range(1, self.n_epochs + 1):
                t0 = time.time()
                _, loss, rmse = self.sess.run([self.training_op, self.loss, self.rmse],
                              feed_dict={self.user_indices: dataset.train_user_indices,
                                         self.item_indices: dataset.train_item_indices,
                                         self.ratings: dataset.train_ratings})

                test_loss, test_rmse = self.sess.run([self.loss, self.rmse],
                                                     feed_dict={self.user_indices: dataset.test_user_indices,
                                                                self.item_indices: dataset.test_item_indices,
                                                                self.ratings: dataset.test_ratings})

                print("Epoch {}, loss: {:.4f}, rmse: {:.4f}, training_time: {:.2f}".format(
                    epoch, loss, rmse, time.time() - t0))
                print("Epoch {}, test_loss: {:.4f}, test_rmse: {:.4f}".format(
                    epoch, test_loss, test_rmse))
                print()































