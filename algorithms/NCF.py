import time
import numpy as np
import tensorflow as tf
from ..utils.negative_sampling import negative_sampling
from ..evaluate.evaluate import precision_tf, AP_at_k, MAP_at_k


class NCF_9999:
    def __init__(self, embed_size, lr, n_epochs=20, reg=0.0,
                 batch_size=64, dropout=0.0, seed=42):
        self.embed_size = embed_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.reg = reg
        self.batch_size = batch_size
        self.dropout = dropout
        self.seed = seed

    def fit(self, dataset):
        tf.set_random_seed(self.seed)
        self.dataset = dataset
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.global_mean = dataset.global_mean
        regularizer = tf.contrib.layers.l2_regularizer(0.0)
    #    self.pu_GMF = tf.get_variable(name="pu_GMF", initializer=tf.glorot_normal_initializer().__call__(shape=[2,2]))
        self.pu_GMF = tf.get_variable(name="pu_GMF", initializer=tf.variance_scaling_initializer,
                                      regularizer=regularizer,
                                      shape=[self.n_users, self.embed_size])
        self.qi_GMF = tf.get_variable(name="qi_GMF", initializer=tf.variance_scaling_initializer,
                                      regularizer=regularizer,
                                      shape=[self.n_items, self.embed_size])
        self.pu_MLP = tf.get_variable(name="pu_MLP", initializer=tf.variance_scaling_initializer,
                                      regularizer=regularizer,
                                      shape=[self.n_users, self.embed_size])
        self.qi_MLP = tf.get_variable(name="qi_MLP", initializer=tf.variance_scaling_initializer,
                                      regularizer=regularizer,
                                      shape=[self.n_items, self.embed_size])

        self.user_indices = tf.placeholder(tf.int32, shape=[None], name="user_indices")
        self.item_indices = tf.placeholder(tf.int32, shape=[None], name="item_indices")
        self.ratings = tf.placeholder(tf.int32, shape=[None], name="ratings")

        self.pu_GMF_embedding = tf.nn.embedding_lookup(self.pu_GMF, self.user_indices)
        self.qi_GMF_embedding = tf.nn.embedding_lookup(self.qi_GMF, self.item_indices)
        self.pu_MLP_embedding = tf.nn.embedding_lookup(self.pu_MLP, self.user_indices)
        self.qi_MLP_embedding = tf.nn.embedding_lookup(self.qi_MLP, self.item_indices)

        self.GMF_layer = tf.multiply(self.pu_GMF_embedding, self.qi_GMF_embedding)

        self.MLP_input = tf.concat([self.pu_MLP_embedding, self.qi_MLP_embedding], axis=1, name="MLP_input")
        self.MLP_layer1 = tf.layers.dense(inputs=self.MLP_input,
                                          units=self.embed_size * 2,
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.variance_scaling_initializer,
                                          name="MLP_layer1")
    #    self.MLP_layer1 = tf.layers.dropout(self.MLP_layer1, rate=self.dropout)
        self.MLP_layer2 = tf.layers.dense(inputs=self.MLP_layer1,
                                          units=self.embed_size,
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.variance_scaling_initializer,
                                          name="MLP_layer2")
    #    self.MLP_layer2 = tf.layers.dropout(self.MLP_layer2, rate=self.dropout)
        self.MLP_layer3 = tf.layers.dense(inputs=self.MLP_layer2,
                                          units=self.embed_size,
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.variance_scaling_initializer,
                                          name="MLP_layer3")
    #    self.MLP_layer3 = tf.layers.dropout(self.MLP_layer3, rate=self.dropout)

        self.Neu_layer = tf.concat([self.GMF_layer, self.MLP_layer3], axis=1)
        self.pred = tf.layers.dense(inputs=self.Neu_layer,
                                    units=1,
                                    name="pred")
    #    self.loss = tf.reduce_sum(tf.square(tf.cast(self.ratings, tf.float32) - self.pred)) / \
    #                tf.cast(tf.size(self.ratings), tf.float32)
        self.loss = tf.losses.mean_squared_error(labels=tf.reshape(self.ratings, [-1,1]), predictions=self.pred)
        self.metrics = tf.sqrt(
            tf.losses.mean_squared_error(labels=tf.reshape(self.ratings, [-1, 1]),
                                         predictions=tf.clip_by_value(self.pred, 1, 5))
        )

        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.training_op = self.optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init)
        with self.sess.as_default():
            for epoch in range(1, self.n_epochs + 1):
                t0 = time.time()
                n_batches = len(dataset.train_ratings) // self.batch_size
                for n in range(n_batches):
                    end = min(len(dataset.train_ratings), (n+1) * self.batch_size)
                    u = dataset.train_user_indices[n * self.batch_size: end]
                    i = dataset.train_item_indices[n * self.batch_size: end]
                    r = dataset.train_ratings[n * self.batch_size: end]
                    self.sess.run([self.training_op],
                              feed_dict={self.user_indices: u,
                                         self.item_indices: i,
                                         self.ratings: r})

                train_loss = self.sess.run(self.metrics,
                                       feed_dict={self.user_indices: dataset.train_user_indices,
                                                  self.item_indices: dataset.train_item_indices,
                                                  self.ratings: dataset.train_ratings})
                test_loss = self.sess.run(self.metrics,
                                           feed_dict={self.user_indices: dataset.test_user_indices,
                                                      self.item_indices: dataset.test_item_indices,
                                                      self.ratings: dataset.test_ratings})
                print("Epoch: {}\ttrain loss: {:.4f}\ttest loss: {:.4f}".format(epoch, train_loss, test_loss))
                print("Epoch {}, training time: {:.4f}".format(epoch, time.time() - t0))

    def predict(self, u, i):
    #    r = np.zeros(len(u))
        r = -1
        try:
            pred = self.sess.run(self.pred, feed_dict={self.ratings: np.array([r]),
                                                       self.user_indices: np.array([u]),
                                                       self.item_indices: np.array([i])})
            pred = np.clip(pred, 1, 5)
        except tf.errors.InvalidArgumentError:
            pred = self.global_mean
        return pred




class NCF:
    def __init__(self, embed_size, lr, n_epochs=20, reg=0.0,
                 batch_size=64, dropout=0.0, seed=42):
        self.embed_size = embed_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.reg = reg
        self.batch_size = batch_size
        self.dropout = dropout
        self.seed = seed

    def fit(self, dataset):
        tf.set_random_seed(self.seed)
        self.dataset = dataset
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.global_mean = dataset.global_mean
        regularizer = tf.contrib.layers.l2_regularizer(0.0)
    #    self.pu_GMF = tf.get_variable(name="pu_GMF", initializer=tf.glorot_normal_initializer().__call__(shape=[2,2]))
        self.pu_GMF = tf.get_variable(name="pu_GMF", initializer=tf.variance_scaling_initializer,
                                      regularizer=regularizer,
                                      shape=[self.n_users, self.embed_size])
        self.qi_GMF = tf.get_variable(name="qi_GMF", initializer=tf.variance_scaling_initializer,
                                      regularizer=regularizer,
                                      shape=[self.n_items, self.embed_size])
        self.pu_MLP = tf.get_variable(name="pu_MLP", initializer=tf.variance_scaling_initializer,
                                      regularizer=regularizer,
                                      shape=[self.n_users, self.embed_size])
        self.qi_MLP = tf.get_variable(name="qi_MLP", initializer=tf.variance_scaling_initializer,
                                      regularizer=regularizer,
                                      shape=[self.n_items, self.embed_size])

        self.user_indices = tf.placeholder(tf.int32, shape=[None], name="user_indices")
        self.item_indices = tf.placeholder(tf.int32, shape=[None], name="item_indices")
        self.labels = tf.placeholder(tf.float32, shape=[None], name="labels")

        self.pu_GMF_embedding = tf.nn.embedding_lookup(self.pu_GMF, self.user_indices)
        self.qi_GMF_embedding = tf.nn.embedding_lookup(self.qi_GMF, self.item_indices)
        self.pu_MLP_embedding = tf.nn.embedding_lookup(self.pu_MLP, self.user_indices)
        self.qi_MLP_embedding = tf.nn.embedding_lookup(self.qi_MLP, self.item_indices)

        self.GMF_layer = tf.multiply(self.pu_GMF_embedding, self.qi_GMF_embedding)

        self.MLP_input = tf.concat([self.pu_MLP_embedding, self.qi_MLP_embedding], axis=1, name="MLP_input")
        self.MLP_layer1 = tf.layers.dense(inputs=self.MLP_input,
                                          units=self.embed_size * 2,
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.variance_scaling_initializer,
                                          name="MLP_layer1")
    #    self.MLP_layer1 = tf.layers.dropout(self.MLP_layer1, rate=self.dropout)
        self.MLP_layer2 = tf.layers.dense(inputs=self.MLP_layer1,
                                          units=self.embed_size,
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.variance_scaling_initializer,
                                          name="MLP_layer2")
    #    self.MLP_layer2 = tf.layers.dropout(self.MLP_layer2, rate=self.dropout)
        self.MLP_layer3 = tf.layers.dense(inputs=self.MLP_layer2,
                                          units=self.embed_size,
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.variance_scaling_initializer,
                                          name="MLP_layer3")
    #    self.MLP_layer3 = tf.layers.dropout(self.MLP_layer3, rate=self.dropout)

        self.Neu_layer = tf.concat([self.GMF_layer, self.MLP_layer3], axis=1)
        self.logits = tf.layers.dense(inputs=self.Neu_layer,
                                      units=1,
                                      name="logits")
        self.logits = tf.reshape(self.logits, [-1])
        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits))

        self.y_prob = tf.sigmoid(self.logits)
        self.pred = tf.where(self.y_prob >= 0.5,
                             tf.fill(tf.shape(self.logits), 1.0),
                             tf.fill(tf.shape(self.logits), 0.0))

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.labels), tf.float32))
        self.precision = precision_tf(self.pred, self.labels)

        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.training_op = self.optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init)
        with self.sess.as_default():
            for epoch in range(1, self.n_epochs + 1):
                t0 = time.time()
                neg = negative_sampling(dataset, 4, self.batch_size)
                n_batches = len(dataset.train_ratings) // self.batch_size
                for n in range(n_batches):
                    u, i, r = neg.next_batch()
                    self.sess.run([self.training_op],
                              feed_dict={self.user_indices: u,
                                         self.item_indices: i,
                                         self.labels: r})

                train_loss, train_acc, train_precision = \
                    self.sess.run([self.loss, self.accuracy, self.precision],
                                  feed_dict={self.user_indices: dataset.train_user_implicit,
                                             self.item_indices: dataset.train_item_implicit,
                                             self.labels: dataset.train_label_implicit})
                test_loss, test_acc, test_precision = \
                    self.sess.run([self.loss, self.accuracy, self.precision],
                                  feed_dict={self.user_indices: dataset.test_user_implicit,
                                             self.item_indices: dataset.test_item_implicit,
                                             self.labels: dataset.test_label_implicit})
                print("Epoch: {}\ttrain loss: {:.4f}\ttest loss: {:.4f}".format(epoch, train_loss, test_loss))
                print("Epoch: {}\ttrain accuracy: {:.4f}\ttest accuracy: {:.4f}".format(
                    epoch, train_acc, test_acc))
                print("Epoch: {}\ttrain precision: {:.4f}\ttest precision: {:.4f}".format(
                    epoch, train_precision, test_precision))

                mean_average_precision_10 = MAP_at_k(self, self.dataset, 10)
                print("Epoch: {}\t MAP @ {}: {:.4f}".format(epoch, 10, mean_average_precision_10))
                mean_average_precision_100 = MAP_at_k(self, self.dataset, 100)
                print("Epoch: {}\t MAP @ {}: {:.4f}".format(epoch, 100, mean_average_precision_100))

                print("Epoch {}, training time: {:.4f}".format(epoch, time.time() - t0))


    def predict(self, u, i):
    #    r = np.zeros(len(u))
    #    r = -1
        try:
            y_prob, y_pred = self.sess.run([self.y_prob, self.pred],
                                           feed_dict={self.user_indices: np.array([u]),
                                                      self.item_indices: np.array([i])})
        except tf.errors.InvalidArgumentError:
            y_prob, y_pred = [0.0], [0.0]
        return y_prob[0], y_pred[0]

    def predict_user(self, u):
        user_indices = np.full(self.n_items, u)
        item_indices = np.arange(self.n_items)
        y_ranklist = self.sess.run(self.y_prob, feed_dict={self.user_indices: user_indices,
                                                           self.item_indices: item_indices})
        return y_ranklist










