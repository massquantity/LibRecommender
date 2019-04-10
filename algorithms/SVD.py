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



class SVD_tf_11111111111111111111555555555555555555:
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
        ratings = tf.placeholder(tf.int32, shape=[None])
        user_indices = tf.placeholder(tf.int32, shape=[None])
        item_indices = tf.placeholder(tf.int32, shape=[None])

        bias_user = tf.nn.embedding_lookup(bu, user_indices)
        bias_item = tf.nn.embedding_lookup(bi, item_indices)
        embed_user = tf.nn.embedding_lookup(pu, user_indices)
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

    #    pred = tf.clip_by_value(pred, 1, 5)
    #    rmse = tf.sqrt(
    #            tf.reduce_mean(
    #                tf.square(
    #                    tf.subtract(
    #                        tf.cast(ratings, tf.float32), pred))))

        optimizer = tf.train.AdamOptimizer(self.lr)
    #    optimizer = tf.train.GradientDescentOptimizer(self.lr)
        training_op = optimizer.minimize(total_loss)
        init = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(init)
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
            self.bu = bu.eval()
            self.bi = bi.eval()

        self.pred = pred
        self.ratings = ratings
        self.user_indices = user_indices
        self.item_indices = item_indices
        self.global_mean = global_mean


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




class SVD_tf_789999999089799999999999766:
    def __init__(self, n_factors=100, n_epochs=20, lr=0.01, reg=1e-3,
                 batch_size=256, batch_training=True, seed=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.batch_training = batch_training
        self.seed = seed

    def loss_define(self):
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

    def fit(self, dataset):
        tf.set_random_seed(self.seed)
        train_user_indices = dataset.train_user_indices
        train_item_indices = dataset.train_item_indices
        test_user_indices = dataset.test_user_indices
        test_item_indices = dataset.test_item_indices
        train_ratings = dataset.train_ratings
        test_ratings = dataset.test_ratings
        self.global_mean = dataset.global_mean

        self.bu = tf.Variable(tf.zeros([dataset.n_users]))
        self.bi = tf.Variable(tf.zeros([dataset.n_items]))
        self.pu = tf.Variable(tf.random_normal([dataset.n_users, self.n_factors], 0.0, 0.01))
        self.qi = tf.Variable(tf.random_normal([dataset.n_items, self.n_factors], 0.0, 0.01))

        iterator = tf.data.Iterator.from_structure(dataset.dataset_tf.output_types,
                                                   dataset.dataset_tf.output_shapes)
        sample = iterator.get_next()
        self.user_indices = sample['user']
        self.item_indices = sample['item']
        self.ratings = sample['rating']

        self.loss_define()
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(self.n_epochs):
                t0 = time.time()
                sess.run(iterator.make_initializer(dataset.dataset_tf))
            #    iterator = dataset.dataset_tf.make_one_shot_iterator()  #####
                sample = iterator.get_next()
                self.user_indices = sample['user']
                self.item_indices = sample['item']
                self.ratings = sample['rating']
                try:
                    while True:
                        train_loss, _ = sess.run([self.total_loss, self.training_op])
                except tf.errors.OutOfRangeError:
                    print("epoch end")
                print("Epoch: ", epoch + 1, "\ttrain loss: {}".format(train_loss))
                print("Epoch {}, training time: {:.4f}".format(epoch + 1, time.time() - t0))

            self.pu = self.pu.eval()
            self.qi = self.qi.eval()
            self.bu = self.bu.eval()
            self.bi = self.bi.eval()

    #    self.pred = pred
    #    self.ratings = ratings
    #    self.user_indices = user_indices
    #    self.item_indices = item_indices
    #    self.global_mean = global_mean


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

    def loss_define(self):
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

    def fit(self, dataset):
        tf.set_random_seed(self.seed)
        train_user_indices = dataset.train_user_indices
        train_item_indices = dataset.train_item_indices
        test_user_indices = dataset.test_user_indices
        test_item_indices = dataset.test_item_indices
        train_ratings = dataset.train_ratings
        test_ratings = dataset.test_ratings
        self.global_mean = dataset.global_mean

        self.bu = tf.Variable(tf.zeros([dataset.n_users]))
        self.bi = tf.Variable(tf.zeros([dataset.n_items]))
        self.pu = tf.Variable(tf.random_normal([dataset.n_users, self.n_factors], 0.0, 0.01))
        self.qi = tf.Variable(tf.random_normal([dataset.n_items, self.n_factors], 0.0, 0.01))
        '''
        iterator = dataset.dataset_tf.make_one_shot_iterator()  # repeat(5)
        sample = iterator.get_next()
        self.user_indices = sample['user']
        self.item_indices = sample['item']
        self.ratings = sample['rating']

        self.loss_define()
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            sess.run(init)
            try:
                while True:
                    train_loss, _ = sess.run([self.total_loss, self.training_op])
            except tf.errors.OutOfRangeError:
                print("epoch end")


            self.pu = self.pu.eval()
            self.qi = self.qi.eval()
            self.bu = self.bu.eval()
            self.bi = self.bi.eval()
        '''

        def get(sample):
            return sample['user'], sample['item'], sample['rating']

        iterator = dataset.dataset_tf.make_initializable_iterator()
        sample = iterator.get_next()
    #    self.user_indices = sample['user']
    #    self.item_indices = sample['item']
    #    self.ratings = sample['rating']
        self.user_indices, self.item_indices, self.ratings = get(sample)

        self.loss_define()
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(1, self.n_epochs + 1):
                t0 = time.time()
                sess.run(iterator.initializer)
                while True:
                    try:
                        train_loss, _ = sess.run([self.total_loss, self.training_op])
                    except tf.errors.OutOfRangeError:
                        break

            #    train_loss = sess.run(self.total_loss)
                print("Epoch: ", epoch, "\ttrain loss: {}".format(train_loss))
                print("Epoch {}, training time: {:.4f}".format(epoch + 1, time.time() - t0))

            self.pu = self.pu.eval()
            self.qi = self.qi.eval()
            self.bu = self.bu.eval()
            self.bi = self.bi.eval()

    #    self.pred = pred
    #    self.ratings = ratings
    #    self.user_indices = user_indices
    #    self.item_indices = item_indices
    #    self.global_mean = global_mean


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