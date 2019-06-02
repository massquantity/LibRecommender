import time
import numpy as np
import tensorflow as tf


class FM_675:
    def __init__(self, lr, n_epochs=20, n_factors=100, reg=0.0, batch_size=256, seed=42):
        self.lr = lr
        self.n_epochs = n_epochs
        self.n_factors = n_factors
        self.reg = reg
        self.batch_size = batch_size
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
        self.dim = dataset.n_users + dataset.n_items
        self.indices = tf.placeholder(tf.int64, shape=[None, 2], name="indices")
        self.values = tf.placeholder(tf.float32, shape=[None], name="values")
        self.shape = tf.placeholder(tf.int64, shape=[2], name="shape")
        self.ratings = tf.placeholder(tf.float32, shape=[None], name="ratings")

        self.w = tf.Variable(tf.truncated_normal([self.dim], 0.0, 0.01))
        self.v = tf.Variable(tf.truncated_normal([self.dim, self.n_factors], 0.0, 0.01))

        #    self.user_onehot = tf.one_hot(self.user_indices, dataset.n_users)
        #    self.item_onehot = tf.one_hot(self.item_indices, dataset.n_items)
        #    self.x = tf.concat([self.user_onehot, self.item_onehot], axis=1)

        self.x = tf.sparse.SparseTensor(self.indices, self.values, self.shape)
        self.w = tf.reshape(self.w, [-1, 1])
    #    self.linear_term = tf.reduce_sum(tf.multiply(self.w, self.x), axis=1, keepdims=True)
    #    self.linear_term = tf.reduce_sum(self.x.__mul__(self.w), axis=1, keepdims=True)
        self.linear_term = tf.sparse_tensor_dense_matmul(self.x, self.w)

        self.pairwise_term = 0.5 * tf.reduce_sum(
            tf.subtract(
                tf.square(tf.sparse_tensor_dense_matmul(self.x, self.v)),
                tf.sparse_tensor_dense_matmul(tf.square(self.x), tf.square(self.v))), axis=1, keepdims=True)

        self.pred = tf.add(self.linear_term, self.pairwise_term)
        self.loss = tf.losses.mean_squared_error(labels=tf.reshape(self.ratings, [-1, 1]),
                                                 predictions=self.pred)

        self.rmse = tf.sqrt(tf.losses.mean_squared_error(labels=tf.reshape(self.ratings, [-1, 1]),
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
    #    self.optimizer = tf.train.FtrlOptimizer(learning_rate=0.1, l1_regularization_strength=1e-3)
        self.training_op = self.optimizer.minimize(self.total_loss)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.sess.run(tf.local_variables_initializer())
        with self.sess.as_default():
            for epoch in range(1, self.n_epochs + 1):
                t0 = time.time()
                n_batches = len(dataset.train_ratings) // self.batch_size
                for n in range(n_batches):
                    end = min(len(dataset.train_ratings), (n + 1) * self.batch_size)
                    user_batch = dataset.train_user_indices[n * self.batch_size: end]
                    item_batch = dataset.train_item_indices[n * self.batch_size: end]
                    rating_batch = dataset.train_ratings[n * self.batch_size: end]

                    indices_batch, values_batch, shape_batch = FM.build_sparse_data(dataset,
                                                                                    user_batch,
                                                                                    item_batch)
                    self.sess.run(self.training_op, feed_dict={self.indices: indices_batch,
                                                               self.values: values_batch,
                                                               self.shape: shape_batch,
                                                               self.ratings: rating_batch})

                if epoch % 1 == 0:
                    indices_train, values_train, shape_train = FM.build_sparse_data(
                                                                    dataset,
                                                                    dataset.train_user_indices,
                                                                    dataset.train_item_indices)
                    train_rmse = self.sess.run(self.rmse, feed_dict={self.indices: indices_train,
                                                                     self.values: values_train,
                                                                     self.shape: shape_train,
                                                                     self.ratings: dataset.train_ratings})

                    indices_test, values_test, shape_test = FM.build_sparse_data(
                                                                dataset,
                                                                dataset.test_user_indices,
                                                                dataset.test_item_indices)
                    test_rmse = self.sess.run(self.rmse, feed_dict={self.indices: indices_test,
                                                                    self.values: values_test,
                                                                    self.shape: shape_test,
                                                                    self.ratings: dataset.test_ratings})

                    print("Epoch {}, train_rmse: {:.4f}, training_time: {:.2f}".format(
                            epoch, train_rmse, time.time() - t0))
                    print("Epoch {}, test_rmse: {:.4f}".format(epoch, test_rmse))
                    print()

    def predict(self, u, i):
        index, value, shape = FM.build_sparse_data(self.dataset, np.array([u]), np.array([i]))
        try:
            pred = self.sess.run(self.pred, feed_dict={self.indices: index,
                                                       self.values: value,
                                                       self.shape: shape})
            pred = np.clip(pred, 1, 5)
        except tf.errors.InvalidArgumentError:
            pred = self.dataset.global_mean
        return pred


class FM:
    def __init__(self, lr, n_epochs=20, n_factors=100, reg=0.0, batch_size=256, seed=42):
        self.lr = lr
        self.n_epochs = n_epochs
        self.n_factors = n_factors
        self.reg = reg
        self.batch_size = batch_size
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
        self.dim = dataset.n_users + dataset.n_items
    #    self.indices = tf.placeholder(tf.int64, shape=[None, 2], name="indices")
    #    self.values = tf.placeholder(tf.float32, shape=[None], name="values")
    #    self.shape = tf.placeholder(tf.int64, shape=[2], name="shape")
    #    self.ratings = tf.placeholder(tf.float32, shape=[None], name="ratings")

        self.w = tf.Variable(tf.truncated_normal([self.dim, 1], 0.0, 0.01))
        self.v = tf.Variable(tf.truncated_normal([self.dim, self.n_factors], 0.0, 0.01))

        #    self.user_onehot = tf.one_hot(self.user_indices, dataset.n_users)
        #    self.item_onehot = tf.one_hot(self.item_indices, dataset.n_items)
        #    self.x = tf.concat([self.user_onehot, self.item_onehot], axis=1)

        self.x = tf.sparse_placeholder(tf.float32, [None, self.dim])
        self.ratings = tf.placeholder(tf.float32, shape=[None], name="ratings")
    #    self.linear_term = tf.reduce_sum(tf.multiply(self.w, self.x), axis=1, keepdims=True)
    #    self.linear_term = tf.reduce_sum(self.x.__mul__(self.w), axis=1, keepdims=True)
        self.linear_term = tf.sparse_tensor_dense_matmul(self.x, self.w)

        self.pairwise_term = 0.5 * tf.reduce_sum(
            tf.subtract(
                tf.square(tf.sparse_tensor_dense_matmul(self.x, self.v)),
                tf.sparse_tensor_dense_matmul(tf.square(self.x), tf.square(self.v))), axis=1, keepdims=True)

        self.pred = tf.add(self.linear_term, self.pairwise_term)
        self.loss = tf.losses.mean_squared_error(labels=tf.reshape(self.ratings, [-1, 1]),
                                                 predictions=self.pred)

        self.rmse = tf.sqrt(tf.losses.mean_squared_error(labels=tf.reshape(self.ratings, [-1, 1]),
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
    #    self.optimizer = tf.train.FtrlOptimizer(learning_rate=0.1, l1_regularization_strength=1e-3)
        self.training_op = self.optimizer.minimize(self.total_loss)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.sess.run(tf.local_variables_initializer())
        with self.sess.as_default():
            for epoch in range(1, self.n_epochs + 1):
                t0 = time.time()
                n_batches = len(dataset.train_ratings) // self.batch_size
                for n in range(n_batches):
                    end = min(len(dataset.train_ratings), (n + 1) * self.batch_size)
                    user_batch = dataset.train_user_indices[n * self.batch_size: end]
                    item_batch = dataset.train_item_indices[n * self.batch_size: end]
                    rating_batch = dataset.train_ratings[n * self.batch_size: end]

                    indices_batch, values_batch, shape_batch = FM.build_sparse_data(dataset,
                                                                                    user_batch,
                                                                                    item_batch)
                    self.sess.run(self.training_op, feed_dict={self.x: tf.SparseTensorValue(indices_batch,
                                                                                            values_batch,
                                                                                            shape_batch),
                                                               self.ratings: rating_batch})

                if epoch % 1 == 0:
                    indices_train, values_train, shape_train = FM.build_sparse_data(
                                                                    dataset,
                                                                    dataset.train_user_indices,
                                                                    dataset.train_item_indices)
                    train_rmse = self.sess.run(self.rmse, feed_dict={self.x: (indices_train,
                                                                              values_train,
                                                                              shape_train),
                                                                     self.ratings: dataset.train_ratings})

                    indices_test, values_test, shape_test = FM.build_sparse_data(
                                                                dataset,
                                                                dataset.test_user_indices,
                                                                dataset.test_item_indices)
                    test_rmse = self.sess.run(self.rmse, feed_dict={self.x: (indices_test,
                                                                             values_test,
                                                                             shape_test),
                                                                    self.ratings: dataset.test_ratings})

                    print("Epoch {}, train_rmse: {:.4f}, training_time: {:.2f}".format(
                            epoch, train_rmse, time.time() - t0))
                    print("Epoch {}, test_rmse: {:.4f}".format(epoch, test_rmse))
                    print()

    def predict(self, u, i):
        index, value, shape = FM.build_sparse_data(self.dataset, np.array([u]), np.array([i]))
        try:
            pred = self.sess.run(self.pred, feed_dict={self.x: (index, value, shape)})
            pred = np.clip(pred, 1, 5)
        except tf.errors.InvalidArgumentError:
            pred = self.dataset.global_mean
        return pred

class FM_67876:
    def __init__(self, lr, n_epochs=20, n_factors=100, reg=0.0, batch_size=256, seed=42):
        self.lr = lr
        self.n_epochs = n_epochs
        self.n_factors = n_factors
        self.reg = reg
        self.batch_size = batch_size
        self.seed = seed

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

        self.w = tf.Variable(tf.truncated_normal([self.feature_size + 1, 1], 0.0, 0.01))
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
        self.pred = tf.layers.dense(inputs=self.concat, units=1)
        self.loss = tf.losses.mean_squared_error(labels=tf.reshape(self.labels, [-1, 1]),
                                                 predictions=self.pred)
        self.rmse = tf.sqrt(tf.losses.mean_squared_error(labels=tf.reshape(self.labels, [-1, 1]),
                                                         predictions=tf.clip_by_value(self.pred, 1, 5)))

    #    reg_w = self.reg * tf.nn.l2_loss(self.w)
        reg_v = self.reg * tf.nn.l2_loss(self.v)
        self.total_loss = tf.add_n([self.loss, reg_v])  # reg_w

    def fit(self, dataset):
        self.build_model(dataset)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
    #    self.optimizer = tf.train.FtrlOptimizer(learning_rate=0.1, l1_regularization_strength=1e-3)
        self.training_op = self.optimizer.minimize(self.total_loss)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.sess.run(tf.local_variables_initializer())
        with self.sess.as_default():
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

                if epoch % 1 == 0:
                #    train_rmse = self.rmse.eval(feed_dict={self.feature_indices: dataset.train_feat_indices,
                #                                           self.feature_values: dataset.train_feat_values,
                #                                           self.labels: dataset.train_labels})

                    test_loss, test_rmse = self.sess.run([self.total_loss, self.rmse],
                                                          feed_dict={
                                                              self.feature_indices: dataset.test_feat_indices,
                                                              self.feature_values: dataset.test_feat_values,
                                                              self.labels: dataset.test_labels})

                #    print("Epoch {}, train_rmse: {:.4f}, training_time: {:.2f}".format(
                #            epoch, train_rmse, time.time() - t0))
                    print("Epoch {}, training_time: {:.2f}".format(epoch, time.time() - t0))
                    print("Epoch {}, test_loss: {:.4f}, test_rmse: {:.4f}".format(
                        epoch, test_loss, test_rmse))
                    print()

    def predict(self, feat_ind, feat_val):
        try:
            pred = self.sess.run(self.pred, feed_dict={self.feature_indices: feat_ind,
                                                       self.feature_values: feat_val})
            pred = np.clip(pred, 1, 5)
        except tf.errors.InvalidArgumentError:
            pred = self.dataset.global_mean
        return pred


class FM_899:
    def __init__(self, lr, n_epochs=20, n_factors=100, reg=0.0, batch_size=256, seed=42):
        self.lr = lr
        self.n_epochs = n_epochs
        self.n_factors = n_factors
        self.reg = reg
        self.batch_size = batch_size
        self.seed = seed

    def build_sparse_indices(self, m, n):  # [[0],[0],[0],[1],[1],[1]...[n],[n],[n]] -> m * n, sample_size * feature_size
        si = []
        for i in range(m):
            si.append(np.zeros([n, 1], dtype=np.int64) + i)
        return np.concatenate(si)

    def build_model(self, dataset):
        tf.set_random_seed(self.seed)
        self.dataset = dataset
        self.field_size = dataset.train_feat_indices.shape[1]
        self.feature_size = dataset.feature_size
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items

        self.feature_indices = tf.placeholder(tf.int32, shape=[None, self.field_size])
        self.feature_values = tf.placeholder(tf.float32, shape=[None, self.field_size])
        self.sparse_indices = tf.placeholder(tf.int64, shape=[None, 1]) # embedding_lookup_sparse indices must be int64
        self.labels = tf.placeholder(tf.float32, shape=[None])

        self.w = tf.Variable(tf.truncated_normal([self.feature_size + 1, 1], 0.0, 0.01))
        self.v = tf.Variable(tf.truncated_normal([self.feature_size + 1, self.n_factors], 0.0, 0.01))
        self.feature_values_reshape = tf.reshape(self.feature_values, shape=[-1, self.field_size, 1])

        sparse = {'indices': self.sparse_indices,
                  'values': tf.reshape(self.feature_indices, [-1]),
                  'dense_shape': [1]}
        sp_ids = tf.SparseTensor(**sparse)

        sparse_sw = {'indices': sparse['indices'],
                     'values': tf.reshape(self.feature_values, [-1]),
                     'dense_shape': sparse['dense_shape']}
        sp_weights = tf.SparseTensor(**sparse_sw)

        self.linear_term = tf.nn.embedding_lookup_sparse(self.w, sp_ids, sp_weights, combiner="sum")

        self.feature_embedding = tf.nn.embedding_lookup_sparse(self.v, sp_ids, sp_weights, combiner="sum")
    #    self.feature_embedding = tf.multiply(self.feature_embedding, self.feature_values_reshape)

        square_embedding = tf.square(self.v)
        sparse_sw_square = {'indices': sparse['indices'],
                            'values': tf.reshape(tf.square(self.feature_values), [-1]),
                            'dense_shape': sparse['dense_shape']}
        sp_weights_square = tf.SparseTensor(**sparse_sw_square)
        self.square_embedding = tf.nn.embedding_lookup_sparse(
            square_embedding, sp_ids, sp_weights_square, combiner="sum")

        self.pairwise_term = 0.5 * tf.subtract(tf.square(self.feature_embedding), self.square_embedding)

        print("linear shape: {}, pairwise shape: {}".format(
            self.linear_term.get_shape(), self.pairwise_term.get_shape()))
        self.concat = tf.concat([self.linear_term, self.pairwise_term], axis=1)
        self.pred = tf.layers.dense(inputs=self.concat, units=1)
        self.loss = tf.losses.mean_squared_error(labels=tf.reshape(self.labels, [-1, 1]),
                                                 predictions=self.pred)
        self.rmse = tf.sqrt(tf.losses.mean_squared_error(labels=tf.reshape(self.labels, [-1, 1]),
                                                         predictions=tf.clip_by_value(self.pred, 1, 5)))

    #    reg_w = self.reg * tf.nn.l2_loss(self.w)
        reg_v = self.reg * tf.nn.l2_loss(self.v)
        self.total_loss = tf.add_n([self.loss, reg_v])  # reg_w

    def fit(self, dataset):
        self.build_model(dataset)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
    #    self.optimizer = tf.train.FtrlOptimizer(learning_rate=0.1, l1_regularization_strength=1e-3)
        self.training_op = self.optimizer.minimize(self.total_loss)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.sess.run(tf.local_variables_initializer())
        with self.sess.as_default():
            for epoch in range(1, self.n_epochs + 1):
                t0 = time.time()
                n_batches = len(dataset.train_labels) // self.batch_size
                m, n = self.batch_size, dataset.train_feat_indices.shape[1]
                sparse_batch = self.build_sparse_indices(m, n)
                for n in range(n_batches):
                    end = min(len(dataset.train_labels), (n + 1) * self.batch_size)
                    indices_batch = dataset.train_feat_indices[n * self.batch_size: end]
                    values_batch = dataset.train_feat_values[n * self.batch_size: end]
                    labels_batch = dataset.train_labels[n * self.batch_size: end]

                    self.sess.run(self.training_op, feed_dict={self.feature_indices: indices_batch,
                                                               self.feature_values: values_batch,
                                                               self.sparse_indices: sparse_batch,
                                                               self.labels: labels_batch})

                if epoch % 1 == 0:
                #    train_rmse = self.rmse.eval(feed_dict={self.feature_indices: dataset.train_feat_indices,
                #                                           self.feature_values: dataset.train_feat_values,
                #                                           self.labels: dataset.train_labels})

                    test_m, test_n = dataset.test_feat_indices.shape[0], dataset.test_feat_indices.shape[1]
                    test_sparse_indices = self.build_sparse_indices(test_m, test_n)

                    test_loss, test_rmse = self.sess.run([self.total_loss, self.rmse],
                                                          feed_dict={
                                                              self.feature_indices: dataset.test_feat_indices,
                                                              self.feature_values: dataset.test_feat_values,
                                                              self.sparse_indices: test_sparse_indices,
                                                              self.labels: dataset.test_labels})

                #    print("Epoch {}, train_rmse: {:.4f}, training_time: {:.2f}".format(
                #            epoch, train_rmse, time.time() - t0))
                    print("Epoch {}, training_time: {:.2f}".format(epoch, time.time() - t0))
                    print("Epoch {}, test_loss: {:.4f}, test_rmse: {:.4f}".format(
                        epoch, test_loss, test_rmse))
                    print()

    def predict(self, feat_ind, feat_val):
        try:
            pred = self.sess.run(self.pred, feed_dict={self.feature_indices: feat_ind,
                                                       self.feature_values: feat_val})
            pred = np.clip(pred, 1, 5)
        except tf.errors.InvalidArgumentError:
            pred = self.dataset.global_mean
        return pred










