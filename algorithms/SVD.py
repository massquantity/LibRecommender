import time
from operator import itemgetter
import numpy as np
from ..evaluate import rmse_svd
try:
    import tensorflow as tf
except ModuleNotFoundError:
    print("you need tensorflow for tf-version of this model")


class SVD:
    def __init__(self, n_factors=100, n_epochs=20, reg=5.0, seed=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.reg = reg
        self.seed = seed

    def fit(self, dataset):
        np.random.seed(self.seed)
        self.dataset = dataset
        self.default_prediction = dataset.global_mean
        self.pu = np.random.normal(loc=0.0, scale=0.1,
                                   size=(dataset.n_users, self.n_factors))
        self.qi = np.random.normal(loc=0.0, scale=0.1,
                                   size=(dataset.n_items, self.n_factors))

        for epoch in range(1, self.n_epochs + 1):
            t0 = time.time()
            for u in dataset.train_user:
                u_items = np.array(list(dataset.train_user[u].keys()))
                u_ratings = np.array(list(dataset.train_user[u].values()))
                u_ratings_expand = np.expand_dims(u_ratings, axis=1)
                yy_reg = self.qi[u_items].T.dot(self.qi[u_items]) + \
                         self.reg * np.eye(self.n_factors)
                r_y = np.sum(np.multiply(u_ratings_expand, self.qi[u_items]), axis=0)
                self.pu[u] = np.linalg.inv(yy_reg).dot(r_y)

            for i in dataset.train_item:
                i_users = np.array(list(dataset.train_item[i].keys()))
            #    if len(i_users) == 0: continue
                i_ratings = np.array(list(dataset.train_item[i].values()))
                i_ratings_expand = np.expand_dims(i_ratings, axis=1)
                xx_reg = self.pu[i_users].T.dot(self.pu[i_users]) + \
                         self.reg * np.eye(self.n_factors)
                r_x = np.sum(np.multiply(i_ratings_expand, self.pu[i_users]), axis=0)
                self.qi[i] = np.linalg.inv(xx_reg).dot(r_x)


            if epoch % 5 == 0:
                print("Epoch {} time: {:.4f}".format(epoch, time.time() - t0))
                print("training rmse: ", self.rmse(dataset, "train"))
                print("test rmse: ", self.rmse(dataset, "test"))
        return self

    def predict(self, u, i):
        try:
            pred = np.dot(self.pu[u], self.qi[i])
            pred = np.clip(pred, 1, 5)
        except IndexError:
            pred = self.default_prediction
        return pred

    def topN(self, u, n_rec, random_rec=False):
        rank = [(j, self.predict(u, j)) for j in range(len(self.qi))
                if j not in self.dataset.train_user[u]]
        if random_rec:
            item_pred_dict = {j: r for j, r in rank if r >= 4}
            item_list = list(item_pred_dict.keys())
            pred_list = list(item_pred_dict.values())
            p = [p / np.sum(pred_list) for p in pred_list]
            item_candidates = np.random.choice(item_list, n_rec, replace=False, p=p)
            reco = [(item, item_pred_dict[item]) for item in item_candidates]
            return reco
        else:
            rank.sort(key=itemgetter(1), reverse=True)
            return rank[:n_rec]

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

        if not self.batch_training:
            for epoch in range(1, self.n_epochs + 1):
                t0 = time.time()
                for u, i, r in zip(dataset.train_user_indices,
                                   dataset.train_item_indices,
                                   dataset.train_ratings):
                    dot = np.dot(self.qi[i], self.pu[u])
                    err = r - (self.global_mean + self.bu[u] + self.bi[i] + dot)
                    self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                    self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                    self.pu[u] += self.lr * (err * self.qi[i] - self.reg * self.pu[u])
                    self.qi[i] += self.lr * (err * self.pu[u] - self.reg * self.qi[i])

                if epoch % 1 == 0:
                    print("Epoch {} time: {:.4f}".format(epoch, time.time() - t0))
                    print("training rmse: ", self.rmse(dataset, "train"))
                    print("test rmse: ", self.rmse(dataset, "test"))
        else:
            for epoch in range(1, self.n_epochs + 1):
                t0 = time.time()
                n_batches = len(dataset.train_ratings) // self.batch_size
                for n in range(n_batches):
                    end = min(len(dataset.train_ratings), (n + 1) * self.batch_size)
                    u = dataset.train_user_indices[n * self.batch_size: end]
                    i = dataset.train_item_indices[n * self.batch_size: end]
                    r = dataset.train_ratings[n * self.batch_size: end]

                    dot = np.sum(np.multiply(self.pu[u], self.qi[i]), axis=1)
                    err = r - (self.global_mean + self.bu[u] + self.bi[i] + dot)
                    self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                    self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                    self.pu[u] += self.lr * (err.reshape(-1, 1) * self.qi[i] - self.reg * self.pu[u])
                    self.qi[i] += self.lr * (err.reshape(-1, 1) * self.pu[u] - self.reg * self.qi[i])

                if epoch % 10 == 0:
                    print("Epoch {} time: {:.4f}".format(epoch, time.time() - t0))
                    print("training rmse: ", self.rmse(dataset, "train"))
                    print("test rmse: ", self.rmse(dataset, "test"))


    def predict(self, u, i):
        try:
            pred = self.global_mean + self.bu[u] + self.bi[i] + np.dot(self.pu[u], self.qi[i])
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


class SVD_tf:
    def __init__(self, n_factors=100, n_epochs=20, lr=0.01, reg=1e-3,
                 batch_size=256, batch_training=True, seed=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.batch_training = batch_training
        self.seed = seed

    def build_model(self, dataset):
        self.bu = tf.Variable(tf.zeros([dataset.n_users]))
        self.bi = tf.Variable(tf.zeros([dataset.n_items]))
        self.pu = tf.Variable(tf.random_normal([dataset.n_users, self.n_factors], 0.0, 0.01))
        self.qi = tf.Variable(tf.random_normal([dataset.n_items, self.n_factors], 0.0, 0.01))

        self.bias_user = tf.nn.embedding_lookup(self.bu, self.user_indices)
        self.bias_item = tf.nn.embedding_lookup(self.bi, self.item_indices)
        self.embed_user = tf.nn.embedding_lookup(self.pu, self.user_indices)
        self.embed_item = tf.nn.embedding_lookup(self.qi, self.item_indices)

        self.pred = self.global_mean + self.bias_user + self.bias_item + \
                    tf.reduce_sum(tf.multiply(self.embed_user, self.embed_item), axis=1)

        self.loss = tf.reduce_sum(
                        tf.square(
                            tf.subtract(
                                tf.cast(self.ratings, tf.float32), self.pred)))

        self.reg_pu = tf.contrib.layers.l2_regularizer(self.reg)(self.pu)
        self.reg_qi = tf.contrib.layers.l2_regularizer(self.reg)(self.qi)
        self.reg_bu = tf.contrib.layers.l2_regularizer(self.reg)(self.bu)
        self.reg_bi = tf.contrib.layers.l2_regularizer(self.reg)(self.bi)
        self.total_loss = tf.add_n([self.loss, self.reg_pu, self.reg_qi, self.reg_bu, self.reg_bi])

        self.optimizer = tf.train.AdamOptimizer(self.lr)
    #    self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self.training_op = self.optimizer.minimize(self.total_loss)

    def fit(self, dataset, data_mode="placeholder"):
        """
        :param dataset:
        :param mode: either "placeholder", "structure", "repeat" or "make"
        :return:
        """
        tf.set_random_seed(self.seed)
        self.global_mean = dataset.global_mean
        if data_mode == "placeholder":
            self.user_indices = tf.placeholder(tf.int32, shape=[None])
            self.item_indices = tf.placeholder(tf.int32, shape=[None])
            self.ratings = tf.placeholder(tf.int32, shape=[None])
        elif data_mode == "structure":
            iterator = tf.data.Iterator.from_structure(dataset.trainset_tf.output_types,
                                                       dataset.trainset_tf.output_shapes)
            sample = iterator.get_next()
            self.user_indices = sample['user']
            self.item_indices = sample['item']
            self.ratings = sample['rating']
            iterator_init = iterator.make_initializer(dataset.trainset_tf)
        elif data_mode == "repeat":
            iterator = dataset.trainset_tf.repeat(self.n_epochs).make_one_shot_iterator()
            sample = iterator.get_next()
            self.user_indices = sample['user']
            self.item_indices = sample['item']
            self.ratings = sample['rating']
        elif data_mode == "make":
            iterator = dataset.trainset_tf.make_initializable_iterator()
            sample = iterator.get_next()
            self.user_indices = sample['user']
            self.item_indices = sample['item']
            self.ratings = sample['rating']
            iterator_init = iterator.initializer
        else:
            raise ValueError("data_mode must be one of these: {}".format(
                "placeholder", "structure", "repeat", "make"))

        self.build_model(dataset)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        with self.sess.as_default():
            if data_mode == "placeholder":
                for epoch in range(1, self.n_epochs + 1):
                    t0 = time.time()
                    if not self.batch_training:
                        self.sess.run(self.training_op, feed_dict={self.ratings: dataset.train_ratings,
                                                                   self.user_indices: dataset.train_user_indices,
                                                                   self.item_indices: dataset.train_item_indices})
                    else:
                        n_batches = len(dataset.train_ratings) // self.batch_size
                        for n in range(n_batches):  # batch training
                            end = min(len(dataset.train_ratings), (n + 1) * self.batch_size)
                            r = dataset.train_ratings[n * self.batch_size: end]
                            u = dataset.train_user_indices[n * self.batch_size: end]
                            i = dataset.train_item_indices[n * self.batch_size: end]
                            self.sess.run([self.training_op], feed_dict={self.ratings: r,
                                                                         self.user_indices: u,
                                                                         self.item_indices: i})

                        train_loss = self.sess.run(self.total_loss,
                                                   feed_dict={self.ratings: dataset.train_ratings,
                                                              self.user_indices: dataset.train_user_indices,
                                                              self.item_indices: dataset.train_item_indices})
                    print("Epoch: ", epoch, "\ttrain loss: {}".format(train_loss))
                    print("Epoch {}, training time: {:.4f}".format(epoch + 1, time.time() - t0))

            elif data_mode == "structure":
                for epoch in range(1, self.n_epochs + 1):
                    t0 = time.time()
                    self.sess.run(iterator_init)
                    sample = iterator.get_next()
                    self.user_indices = sample['user']
                    self.item_indices = sample['item']
                    self.ratings = sample['rating']
                    try:
                        while True:
                            train_loss, _ = self.sess.run([self.total_loss, self.training_op])
                    except tf.errors.OutOfRangeError:
                        print("epoch end")
                    print("Epoch: ", epoch, "\ttrain loss: {}".format(train_loss))
                    print("Epoch {}, training time: {:.4f}".format(epoch + 1, time.time() - t0))

            elif data_mode == "repeat":
                try:
                    while True:
                        train_loss, _ = self.sess.run([self.total_loss, self.training_op])
                except tf.errors.OutOfRangeError:
                    print("training end, loss = %.4f" % train_loss)

            elif data_mode == "make":
                for epoch in range(1, self.n_epochs + 1):
                    t0 = time.time()
                    self.sess.run(iterator_init)
                    while True:
                        try:
                            train_loss, _ = self.sess.run([self.total_loss, self.training_op])
                        except tf.errors.OutOfRangeError:
                            break
                    #    train_loss = sess.run(self.total_loss)
                    print("Epoch: ", epoch, "\ttrain loss: {}".format(train_loss))
                    print("Epoch {}, training time: {:.4f}".format(epoch + 1, time.time() - t0))

            self.pu = self.pu.eval()
            self.qi = self.qi.eval()
            self.bu = self.bu.eval()
            self.bi = self.bi.eval()


    def predict(self, u, i):
        try:
            pred = np.dot(self.pu[u], self.qi[i]) + \
                   self.global_mean + \
                   self.bu[u] + \
                   self.bi[i]
            pred = np.clip(pred, 1, 5)
        except IndexError:
            pred = self.global_mean
        return pred

#    def predict(self, u, i):
#        r = np.zeros(len(u))
#        pred = self.sess.run(self.pred, feed_dict={self.ratings: r,
#                                                   self.user_indices: u,
#                                                   self.item_indices: i})
#        pred = np.clip(pred, 1, 5)
#        return pred