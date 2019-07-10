import time
from operator import itemgetter
import numpy as np
from ..evaluate import rmse, MAP_at_k, accuracy, precision_tf
from ..utils.initializers import truncated_normal
from ..utils import NegativeSampling
try:
    import tensorflow as tf
except ModuleNotFoundError:
    print("you need tensorflow for tf-version of this model")


class SVDBaseline:
    def __init__(self, n_factors=100, n_epochs=20, lr=0.01, reg=5.0,
                 batch_size=256, batch_training=True, seed=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.batch_training = batch_training
        self.seed = seed

    def fit(self, dataset, verbose=1):
        np.random.seed(self.seed)
        self.dataset = dataset
        self.global_mean = dataset.global_mean
        self.bu = np.zeros((dataset.n_users))
        self.bi = np.zeros((dataset.n_items))
        self.pu = truncated_normal(mean=0.0, scale=0.1,
                                   shape=(dataset.n_users, self.n_factors))
        self.qi = truncated_normal(mean=0.0, scale=0.1,
                                   shape=(dataset.n_items, self.n_factors))

        if not self.batch_training:
            for epoch in range(1, self.n_epochs + 1):
                t0 = time.time()
                for u, i, r in zip(dataset.train_user_indices,
                                   dataset.train_item_indices,
                                   dataset.train_labels):
                    dot = np.dot(self.qi[i], self.pu[u])
                    err = r - (self.global_mean + self.bu[u] + self.bi[i] + dot)
                    self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                    self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                    self.pu[u] += self.lr * (err * self.qi[i] - self.reg * self.pu[u])
                    self.qi[i] += self.lr * (err * self.pu[u] - self.reg * self.qi[i])

                if verbose > 0:
                    print("Epoch {} time: {:.4f}".format(epoch, time.time() - t0))
                    print("training rmse: ", rmse(self, dataset, "train"))
                    print("test rmse: ", rmse(self, dataset, "test"))

        else:
            for epoch in range(1, self.n_epochs + 1):
                t0 = time.time()
                n_batches = len(dataset.train_labels) // self.batch_size
                for n in range(n_batches):
                    end = min(len(dataset.train_labels), (n + 1) * self.batch_size)
                    u = dataset.train_user_indices[n * self.batch_size: end]
                    i = dataset.train_item_indices[n * self.batch_size: end]
                    r = dataset.train_labels[n * self.batch_size: end]

                    dot = np.sum(np.multiply(self.pu[u], self.qi[i]), axis=1)
                    err = r - (self.global_mean + self.bu[u] + self.bi[i] + dot)
                    self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                    self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                    self.pu[u] += self.lr * (err.reshape(-1, 1) * self.qi[i] - self.reg * self.pu[u])
                    self.qi[i] += self.lr * (err.reshape(-1, 1) * self.pu[u] - self.reg * self.qi[i])

                if verbose > 0 and epoch % 10 == 0:
                    print("Epoch {} time: {:.4f}".format(epoch, time.time() - t0))
                    print("training rmse: ", rmse(self, dataset, "train"))
                    print("test rmse: ", rmse(self, dataset, "test"))


    def predict(self, u, i):
        try:
            pred = self.global_mean + self.bu[u] + self.bi[i] + np.dot(self.pu[u], self.qi[i])
            pred = np.clip(pred, 1, 5)
        except IndexError:
            pred = self.global_mean
        return pred


class SVD_tf:
    def __init__(self, n_factors=100, n_epochs=20, lr=0.01, reg=1e-3,
                 batch_size=256, seed=42, task="rating"):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.seed = seed
        self.task = task

    def build_model(self, dataset):
        tf.set_random_seed(self.seed)
        self.dataset = dataset
        self.global_mean = dataset.global_mean
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

            self.rmse = tf.sqrt(tf.losses.mean_squared_error(labels=self.labels,
                                                             predictions=tf.clip_by_value(self.pred, 1, 5)))

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

        self.optimizer = tf.train.AdamOptimizer(self.lr)
    #    self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self.training_op = self.optimizer.minimize(self.total_loss)

    def fit(self, dataset, verbose=1):
        """
        :param dataset:
        :param mode: either "placeholder", "structure", "repeat" or "make"
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
            if self.task == "rating":
                for epoch in range(1, self.n_epochs + 1):
                    t0 = time.time()
                    n_batches = int(np.ceil(len(dataset.train_labels) / self.batch_size))
                    for n in range(n_batches):  # batch training
                        end = min(len(dataset.train_labels), (n + 1) * self.batch_size)
                        r = dataset.train_labels[n * self.batch_size: end]
                        u = dataset.train_user_indices[n * self.batch_size: end]
                        i = dataset.train_item_indices[n * self.batch_size: end]
                        self.sess.run([self.training_op], feed_dict={self.labels: r,
                                                                     self.user_indices: u,
                                                                     self.item_indices: i})

                    if verbose > 0:
                        train_rmse = self.sess.run(self.rmse,
                                                   feed_dict={self.labels: dataset.train_labels,
                                                              self.user_indices: dataset.train_user_indices,
                                                              self.item_indices: dataset.train_item_indices})
                        test_rmse = self.sess.run(self.rmse,
                                                   feed_dict={self.labels: dataset.test_labels,
                                                              self.user_indices: dataset.test_user_indices,
                                                              self.item_indices: dataset.test_item_indices})

                        print("Epoch {}, training time: {:.2f}".format(epoch, time.time() - t0))
                        print("Epoch {}, train rmse: {:.4f}".format(epoch, train_rmse))
                        print("Epoch {}, test rmse: {:.4f}".format(epoch, test_rmse))
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
                        train_accuracy, train_precision = \
                            self.sess.run([self.accuracy, self.precision],
                                feed_dict={self.labels: dataset.train_label_implicit,
                                           self.user_indices: dataset.train_user_implicit,
                                           self.item_indices: dataset.train_item_implicit})

                        test_accuracy, test_precision = \
                            self.sess.run([self.accuracy, self.precision],
                                feed_dict={self.labels: dataset.test_label_implicit,
                                           self.user_indices: dataset.test_user_implicit,
                                           self.item_indices: dataset.test_item_implicit})

                        print("Epoch {}, training time: {:.2f}".format(epoch, time.time() - t0))
                        print("Epoch {}, train accuracy: {:.4f}, train precision: {:.4f}".format(
                                epoch, train_accuracy, train_precision))
                        print("Epoch {}, test accuracy: {:.4f}, test precision: {:.4f}".format(
                                epoch, test_accuracy, test_precision))
                        print()

    def predict(self, u, i):
        if self.task == "rating":
            try:
                pred = self.sess.run(self.pred, feed_dict={self.user_indices: [u],
                                                           self.item_indices: [i]})
                pred = np.clip(pred, 1, 5)
            except tf.errors.InvalidArgumentError:
                pred = self.global_mean
            return pred

        elif self.task == "ranking":
            try:
                prob, pred = self.sess.run([self.y_prob, self.pred],
                                            feed_dict={self.user_indices: [u],
                                                       self.item_indices: [i]})
            except tf.errors.InvalidArgumentError:
                prob = 0.5
                pred = self.global_mean
            return prob[0], pred[0]

    def recommend_user(self, u, n_rec):
        items = np.arange(self.dataset.n_items)
        preds = self.sess.run(self.pred, feed_dict={self.user_indices: [u],
                                                     self.item_indices: items})
        rank = list(zip(items, preds))
        rank.sort(key=itemgetter(1), reverse=True)
        return rank[:n_rec]
