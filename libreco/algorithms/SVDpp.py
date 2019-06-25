import time
from operator import itemgetter
import numpy as np
from ..evaluate import rmse_svd
try:
    import tensorflow as tf
except ModuleNotFoundError:
    print("you need tensorflow for tf-version of this model")


class SVDpp:
    def __init__(self, n_factors=100, n_epochs=20, lr=0.01, reg=5.0,
                 batch_size=256, batch_training=True, seed=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.batch_training = batch_training
        self.seed = seed

    def fit(self, dataset):
        np.random.seed(self.seed)
        self.dataset = dataset
        self.global_mean = dataset.global_mean
        self.bu = np.zeros((dataset.n_users))
        self.bi = np.zeros((dataset.n_items))
        self.pu = np.random.normal(loc=0.0, scale=0.1,
                                   size=(dataset.n_users, self.n_factors))
        self.qi = np.random.normal(loc=0.0, scale=0.1,
                                   size=(dataset.n_items, self.n_factors))
        self.yj = np.random.normal(loc=0.0, scale=0.1,
                                   size=(dataset.n_items, self.n_factors))

        if not self.batch_training:
            for epoch in range(1, self.n_epochs + 1):
                t0 = time.time()
                for u, i, r in zip(dataset.train_user_indices,
                                   dataset.train_item_indices,
                                   dataset.train_ratings):
                    u_items = list(dataset.train_user[u].keys())
                    nu_sqrt = np.sqrt(len(u_items))
                    nu = np.sum(self.yj[u_items], axis=0) / nu_sqrt
                    dot = np.dot(self.qi[i], self.pu[u] + nu)
                    err = r - (self.global_mean + self.bu[u] + self.bi[i] + dot)
                    self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                    self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                    self.pu[u] += self.lr * (err * self.qi[i] - self.reg * self.pu[u])
                    self.qi[i] += self.lr * (err * (self.pu[u] + nu) - self.reg * self.qi[i])
                    self.yj[u_items] += self.lr * (err * self.qi[u_items] / nu_sqrt -
                                                   self.reg * self.yj[u_items])

                if epoch % 1 == 0:
                    print("Epoch {} time: {:.4f}".format(epoch, time.time() - t0))
                    print("training rmse: ", self.rmse(dataset, "train"))
                    print("test rmse: ", self.rmse(dataset, "test"))
        else:
            for epoch in range(1, self.n_epochs + 1):
                t0 = time.time()
                random_users = np.random.permutation(list(dataset.train_user.keys()))
                for u in random_users:
                    u_items = list(dataset.train_user[u].keys())
                    u_ratings = np.array(list(dataset.train_user[u].values()))
                    nu_sqrt = np.sqrt(len(u_items))
                    nu = np.sum(self.yj[u_items], axis=0) / nu_sqrt
                    dot = np.dot(self.qi[u_items], self.pu[u] + nu)
                    err = u_ratings - (self.global_mean + self.bu[u] + self.bi[u_items] + dot)
                    err = err.reshape(len(u_items), 1)
                    self.bu[u] += self.lr * (np.sum(err) - self.reg * self.bu[u])
                    self.bi[u_items] += self.lr * (err.flatten() - self.reg * self.bi[u_items])
                    self.qi[u_items] += self.lr * (err * (self.pu[u] + nu) - self.reg * self.qi[u_items])
                    self.pu[u] += self.lr * (np.sum(err * self.qi[u_items], axis=0) - self.reg * self.pu[u])
                    self.yj[u_items] += self.lr * (err * self.qi[u_items] / nu_sqrt - self.reg * self.yj[u_items])

                if epoch % 1 == 0:
                    print("Epoch {} time: {:.4f}".format(epoch, time.time() - t0))
                    print("training rmse: ", self.rmse(dataset, "train"))
                    print("test rmse: ", self.rmse(dataset, "test"))


    def predict(self, u, i):
        try:
            u_items = list(self.dataset.train_user[u].keys())
            nu = np.sum(self.yj[u_items], axis=0) / np.sqrt(len(u_items))
            pred = self.global_mean + self.bu[u] + self.bi[i] + np.dot(self.pu[u] + nu, self.qi[i])
            pred = np.clip(pred, 1, 5)
        except IndexError:
            pred = self.global_mean
        return pred

    def rmse(self, dataset, mode="train"):
        if mode == "train":
            user_indices = dataset.train_user_indices
            item_indices = dataset.train_item_indices
            ratings = dataset.train_ratings
        elif mode == "test":
            user_indices = dataset.test_user_indices
            item_indices = dataset.test_item_indices
            ratings = dataset.test_ratings

        pred = []
        for u, i in zip(user_indices, item_indices):
            p = self.predict(u, i)
            pred.append(p)
        score = np.sqrt(np.mean(np.power(pred - ratings, 2)))
        return score

'''
class SVDpp_tf:
    def __init__(self, n_factors=100, n_epochs=20, lr=0.01, reg=1e-3,
                 batch_size=256, batch_training=True, seed=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.batch_training = batch_training
        self.seed = seed

    def fit(self, dataset):
        start_time = time.time()
        tf.set_random_seed(self.seed)
        train_user_indices = dataset.train_user_indices
        train_item_indices = dataset.train_item_indices
        test_user_indices = dataset.test_user_indices
        test_item_indices = dataset.test_item_indices
        train_ratings = dataset.train_ratings
        test_ratings = dataset.test_ratings
        global_mean = dataset.global_mean

        bu = tf.Variable(tf.zeros([dataset.n_users]))
        bi = tf.Variable(tf.zeros([dataset.n_items]))
        pu = tf.Variable(tf.random_normal([dataset.n_users, self.n_factors], 0.0, 0.01))
        qi = tf.Variable(tf.random_normal([dataset.n_items, self.n_factors], 0.0, 0.01))
        yj = tf.Variable(tf.random_normal([dataset.n_items, self.n_factors], 0.0, 0.01))

        ratings = tf.placeholder(tf.int32, shape=[None])
        user_indices = tf.placeholder(tf.int32, shape=[None])
        item_indices = tf.placeholder(tf.int32, shape=[None])

        bias_user = tf.nn.embedding_lookup(bu, user_indices)
        bias_item = tf.nn.embedding_lookup(bi, item_indices)

        users = list(dataset.train_user.keys())
        u_items = [list(dataset.train_user[u].keys()) for u in users]
        nu = [tf.reduce_sum(tf.gather(yj, u_i), axis=0) / np.sqrt(len(u_i)) for u, u_i in zip(users, u_items)]
        pn = pu + nu

        embed_user = tf.nn.embedding_lookup(pn, user_indices)
        embed_item = tf.nn.embedding_lookup(qi, item_indices)

        pred = global_mean + bias_user + bias_item + \
               tf.reduce_sum(tf.multiply(embed_user, embed_item), axis=1)

        loss = tf.reduce_sum(
                    tf.square(
                        tf.subtract(
                            tf.cast(ratings, tf.float32), pred)))

        reg_pu = tf.contrib.layers.l2_regularizer(self.reg)(pu)
        reg_qi = tf.contrib.layers.l2_regularizer(self.reg)(qi)
        reg_bu = tf.contrib.layers.l2_regularizer(self.reg)(bu)
        reg_bi = tf.contrib.layers.l2_regularizer(self.reg)(bi)
        total_loss = tf.add_n([loss, reg_pu, reg_qi, reg_bu, reg_bi])

        optimizer = tf.train.AdamOptimizer(self.lr)
    #    optimizer = tf.train.GradientDescentOptimizer(self.lr)
        training_op = optimizer.minimize(total_loss)
        init = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init)
        print("tf initialize time: {:.4f}".format(time.time() - start_time))

        with self.sess.as_default():
            for epoch in range(self.n_epochs):
                t0 = time.time()
                if not self.batch_training:
                    self.sess.run(training_op, feed_dict={ratings: train_ratings,
                                                          user_indices: train_user_indices,
                                                          item_indices: train_item_indices})
                else:
                    n_batches = len(train_ratings) // self.batch_size
                    for n in range(n_batches):  # batch training
                        end = min(len(train_ratings), (n+1) * self.batch_size)
                        r = train_ratings[n * self.batch_size: end]
                        u = train_user_indices[n * self.batch_size: end]
                        i = train_item_indices[n * self.batch_size: end]
                        self.sess.run([training_op], feed_dict={ratings: r,
                                                                user_indices: u,
                                                                item_indices: i})

                train_loss = self.sess.run(total_loss,
                                           feed_dict={ratings: train_ratings,
                                                      user_indices: train_user_indices,
                                                      item_indices: train_item_indices})
                print("Epoch: ", epoch + 1, "\ttrain loss: {}".format(train_loss))
                print("Epoch {}, training time: {:.4f}".format(epoch + 1, time.time() - t0))

            self.pu = pu.eval()
            self.qi = qi.eval()
            self.yj = yj.eval()
            self.bu = bu.eval()
            self.bi = bi.eval()

        self.pred = pred
        self.ratings = ratings
        self.user_indices = user_indices
        self.item_indices = item_indices
        self.global_mean = global_mean
        self.dataset = dataset

    def predict(self, u, i):
        try:
            u_items = list(self.dataset.train_user[u].keys())
            nu = np.sum(self.yj[u_items], axis=0) / np.sqrt(len(u_items))
            pred = self.global_mean + self.bu[u] + self.bi[i] + np.dot(self.pu[u] + nu, self.qi[i])
            pred = np.clip(pred, 1, 5)
        except IndexError:
            pred = self.global_mean
        return pred
'''


class SVDpp_tf:
    def __init__(self, n_factors=100, n_epochs=20, lr=0.01, reg=1e-3,
                 batch_size=256, batch_training=True, seed=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.batch_training = batch_training
        self.seed = seed

    def fit(self, dataset):
        start_time = time.time()
        tf.set_random_seed(self.seed)
        train_user_indices = dataset.train_user_indices
        train_item_indices = dataset.train_item_indices
        test_user_indices = dataset.test_user_indices
        test_item_indices = dataset.test_item_indices
        train_ratings = dataset.train_ratings
        test_ratings = dataset.test_ratings
        global_mean = dataset.global_mean

        bu = tf.Variable(tf.zeros([dataset.n_users]))
        bi = tf.Variable(tf.zeros([dataset.n_items]))
        pu = tf.Variable(tf.random_normal([dataset.n_users, self.n_factors], 0.0, 0.01))
        qi = tf.Variable(tf.random_normal([dataset.n_items, self.n_factors], 0.0, 0.01))
        yj = tf.Variable(tf.random_normal([dataset.n_items, self.n_factors], 0.0, 0.01))

        ratings = tf.placeholder(tf.int32, shape=[None])
        user_indices = tf.placeholder(tf.int32, shape=[None])
        item_indices = tf.placeholder(tf.int32, shape=[None])
        bias_user = tf.nn.embedding_lookup(bu, user_indices)
        bias_item = tf.nn.embedding_lookup(bi, item_indices)

        user_split_items = [[] for u in range(dataset.n_users)]
        for u, i in zip(train_user_indices, train_item_indices):
            user_split_items[u].append(i)

        sparse_dict = {'indices': [], 'values': []}
        for i, user in enumerate(user_split_items):
            for j, item in enumerate(user):
                sparse_dict['indices'].append((i, j))
                sparse_dict['values'].append(item)
        sparse_dict['dense_shape'] = (dataset.n_users, dataset.n_items)
        implicit_feedback = tf.SparseTensor(**sparse_dict)
        yjs = tf.nn.embedding_lookup_sparse(yj, implicit_feedback, sp_weights=None, combiner="sqrtn")
        nu = tf.gather(yjs, np.arange(dataset.n_users))

        pn = pu + nu
        embed_user = tf.nn.embedding_lookup(pn, user_indices)
        embed_item = tf.nn.embedding_lookup(qi, item_indices)

        pred = global_mean + bias_user + bias_item + \
               tf.reduce_sum(tf.multiply(embed_user, embed_item), axis=1)

        loss = tf.reduce_sum(
                    tf.square(
                        tf.subtract(
                            tf.cast(ratings, tf.float32), pred)))

        reg_pu = tf.contrib.layers.l2_regularizer(self.reg)(pu)
        reg_qi = tf.contrib.layers.l2_regularizer(self.reg)(qi)
        reg_bu = tf.contrib.layers.l2_regularizer(self.reg)(bu)
        reg_bi = tf.contrib.layers.l2_regularizer(self.reg)(bi)
        total_loss = tf.add_n([loss, reg_pu, reg_qi, reg_bu, reg_bi])

        optimizer = tf.train.AdamOptimizer(self.lr)
    #    optimizer = tf.train.GradientDescentOptimizer(self.lr)
        training_op = optimizer.minimize(total_loss)
        init = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init)
        print("tf initialize time: {:.4f}".format(time.time() - start_time))

        with self.sess.as_default():
            for epoch in range(self.n_epochs):
                t0 = time.time()
                if not self.batch_training:
                    self.sess.run(training_op, feed_dict={ratings: train_ratings,
                                                          user_indices: train_user_indices,
                                                          item_indices: train_item_indices})
                else:
                    n_batches = len(train_ratings) // self.batch_size
                    for n in range(n_batches):  # batch training
                        end = min(len(train_ratings), (n+1) * self.batch_size)
                        r = train_ratings[n * self.batch_size: end]
                        u = train_user_indices[n * self.batch_size: end]
                        i = train_item_indices[n * self.batch_size: end]
                        self.sess.run([training_op], feed_dict={ratings: r,
                                                                user_indices: u,
                                                                item_indices: i})

                train_loss = self.sess.run(total_loss,
                                           feed_dict={ratings: train_ratings,
                                                      user_indices: train_user_indices,
                                                      item_indices: train_item_indices})
                print("Epoch: ", epoch + 1, "\ttrain loss: {}".format(train_loss))
                print("Epoch {}, training time: {:.4f}".format(epoch + 1, time.time() - t0))

            self.pu = pu.eval()
            self.qi = qi.eval()
            self.yj = yj.eval()
            self.bu = bu.eval()
            self.bi = bi.eval()

        self.pred = pred
        self.ratings = ratings
        self.user_indices = user_indices
        self.item_indices = item_indices
        self.global_mean = global_mean
        self.dataset = dataset

    def predict(self, u, i):
        try:
            u_items = list(self.dataset.train_user[u].keys())
            nu = np.sum(self.yj[u_items], axis=0) / np.sqrt(len(u_items))
            pred = self.global_mean + self.bu[u] + self.bi[i] + np.dot(self.pu[u] + nu, self.qi[i])
            pred = np.clip(pred, 1, 5)
        except IndexError:
            pred = self.global_mean
        return pred


