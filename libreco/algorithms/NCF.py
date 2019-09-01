"""

Reference: Xiangnan He et al. "Neural Collaborative Filtering" (https://arxiv.org/pdf/1708.05031.pdf)

author: massquantity

"""
import time
import itertools
import numpy as np
from .Base import BasePure
import tensorflow as tf
from ..utils.sampling import NegativeSampling
from ..evaluate.evaluate import precision_tf


class Ncf(BasePure):
    def __init__(self, embed_size, lr, n_epochs=20, reg=0.0, batch_size=64,
                 dropout_rate=0.0, seed=42, task="rating", neg_sampling=False, network_size=None):
        self.embed_size = embed_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.reg = reg
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.task = task
        self.neg_sampling = neg_sampling
        if network_size is not None:
            assert len(network_size) == 3, "network size does not match true model size"
            self.network_size = network_size
        else:
            self.network_size = [embed_size * 2, embed_size * 2, embed_size]
        super(Ncf, self).__init__()

    def build_model(self, dataset):
        tf.set_random_seed(self.seed)
        self.dataset = dataset
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.global_mean = dataset.global_mean
        if dataset.lower_upper_bound is not None:
            self.lower_bound = dataset.lower_upper_bound[0]
            self.upper_bound = dataset.lower_upper_bound[1]
        else:
            self.lower_bound = None
            self.upper_bound = None

        regularizer = tf.contrib.layers.l2_regularizer(self.reg)
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
        self.dropout_switch = tf.placeholder_with_default(False, shape=[], name="training")

        self.pu_GMF_embedding = tf.nn.embedding_lookup(self.pu_GMF, self.user_indices)
        self.qi_GMF_embedding = tf.nn.embedding_lookup(self.qi_GMF, self.item_indices)
        self.pu_MLP_embedding = tf.nn.embedding_lookup(self.pu_MLP, self.user_indices)
        self.qi_MLP_embedding = tf.nn.embedding_lookup(self.qi_MLP, self.item_indices)

        self.GMF_layer = tf.multiply(self.pu_GMF_embedding, self.qi_GMF_embedding)

        self.MLP_input = tf.concat([self.pu_MLP_embedding, self.qi_MLP_embedding], axis=1, name="MLP_input")
        self.MLP_layer_one = tf.layers.dense(inputs=self.MLP_input,
                                             units=self.network_size[0],
                                             activation=tf.nn.relu,
                                             kernel_initializer=tf.variance_scaling_initializer,
                                             name="MLP_layer_one")
        self.MLP_layer_one = tf.layers.dropout(self.MLP_layer_one, rate=self.dropout_rate, training=self.dropout_switch)
        self.MLP_layer_two = tf.layers.dense(inputs=self.MLP_layer_one,
                                             units=self.network_size[1],
                                             activation=tf.nn.relu,
                                             kernel_initializer=tf.variance_scaling_initializer,
                                             name="MLP_layer_two")
        self.MLP_layer_two = tf.layers.dropout(self.MLP_layer_two, rate=self.dropout_rate, training=self.dropout_switch)
        self.MLP_layer_three = tf.layers.dense(inputs=self.MLP_layer_two,
                                               units=self.network_size[2],
                                               activation=tf.nn.relu,
                                               kernel_initializer=tf.variance_scaling_initializer,
                                               name="MLP_layer_three")

        self.Neu_layer = tf.concat([self.GMF_layer, self.MLP_layer_three], axis=1)

        if self.task == "rating":
            self.pred = tf.layers.dense(inputs=self.Neu_layer, units=1, name="pred")
        #    self.loss = tf.reduce_sum(tf.square(tf.cast(self.labels, tf.float32) - self.pred)) / \
        #                tf.cast(tf.size(self.labels), tf.float32)
            self.loss = tf.losses.mean_squared_error(labels=tf.reshape(self.labels, [-1, 1]),
                                                     predictions=self.pred)

            if self.lower_bound is not None and self.upper_bound is not None:
                self.rmse = tf.sqrt(tf.losses.mean_squared_error(labels=self.labels,
                                predictions=tf.clip_by_value(self.pred, self.lower_bound, self.upper_bound)))
            else:
                self.rmse = self.loss

        elif self.task == "ranking":
            self.logits = tf.layers.dense(inputs=self.Neu_layer, units=1, name="logits")
            self.logits = tf.reshape(self.logits, [-1])
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
            self.y_prob = tf.sigmoid(self.logits)
            self.pred = tf.where(self.y_prob >= 0.5,
                                 tf.fill(tf.shape(self.logits), 1.0),
                                 tf.fill(tf.shape(self.logits), 0.0))
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.labels), tf.float32))
            self.precision = precision_tf(self.pred, self.labels)

    def fit(self, dataset, verbose=1, **kwargs):
        self.build_model(dataset)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
    #    self.optimizer = tf.train.FtrlOptimizer(learning_rate=0.1, l1_regularization_strength=1e-3)
        self.training_op = self.optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        with self.sess.as_default():
            if self.task == "rating" or (self.task == "ranking" and not self.neg_sampling):
                for epoch in range(1, self.n_epochs + 1):
                    t0 = time.time()
                    n_batches = int(np.ceil(len(dataset.train_labels) / self.batch_size))
                    for n in range(n_batches):
                        end = min(len(dataset.train_labels), (n + 1) * self.batch_size)
                        u = dataset.train_user_indices[n * self.batch_size: end]
                        i = dataset.train_item_indices[n * self.batch_size: end]
                        r = dataset.train_labels[n * self.batch_size: end]
                        self.sess.run([self.training_op], feed_dict={self.user_indices: u,
                                                                     self.item_indices: i,
                                                                     self.labels: r,
                                                                     self.dropout_switch: True})

                    if verbose > 0:
                        print("Epoch {}, training_time: {:.2f}".format(epoch, time.time() - t0))
                        metrics = kwargs.get("metrics", self.metrics)
                        if hasattr(self, "sess"):
                            self.print_metrics_tf(dataset, epoch, **metrics)
                        else:
                            self.print_metrics(dataset, epoch, **metrics)
                        print()

            elif self.task == "ranking" and self.neg_sampling:
                for epoch in range(1, self.n_epochs + 1):
                    t0 = time.time()
                    neg = NegativeSampling(dataset, dataset.num_neg, self.batch_size)
                    n_batches = int(np.ceil(len(dataset.train_label_implicit) / self.batch_size))
                    for n in range(n_batches):
                        user_batch, item_batch, label_batch = neg.next_batch()
                        self.sess.run([self.training_op], feed_dict={self.labels: label_batch,
                                                                     self.user_indices: user_batch,
                                                                     self.item_indices: item_batch,
                                                                     self.dropout_switch: True})

                    if verbose > 0:
                        print("Epoch {}: training time: {:.4f}".format(epoch, time.time() - t0))
                        metrics = kwargs.get("metrics", self.metrics)
                        if hasattr(self, "sess"):
                            self.print_metrics_tf(dataset, epoch, **metrics)
                        else:
                            self.print_metrics(dataset, epoch, **metrics)
                        print()

    def predict(self, u, i):
        try:
            target = self.pred if self.task == "rating" else self.y_prob
            pred = self.sess.run(target, feed_dict={self.user_indices: [u],
                                                    self.item_indices: [i]})
            if self.lower_bound is not None and self.upper_bound is not None:
                pred = np.clip(pred, self.lower_bound, self.upper_bound) if self.task == "rating" else pred[0]
        except tf.errors.InvalidArgumentError:
            pred = self.dataset.global_mean if self.task == "rating" else 0.0
        return pred

    def recommend_user(self, u, n_rec):
        user_indices = np.full(self.n_items, u)
        item_indices = np.arange(self.n_items)
        target = self.pred if self.task == "rating" else self.y_prob
        preds = self.sess.run(target, feed_dict={self.user_indices: user_indices,
                                                 self.item_indices: item_indices})
        preds = preds.ravel()
        consumed = self.dataset.train_user[u]
        count = n_rec + len(consumed)
        ids = np.argpartition(preds, -count)[-count:]
        rank = sorted(zip(ids, preds[ids]), key=lambda x: -x[1])
        return list(itertools.islice((rec for rec in rank if rec[0] not in consumed), n_rec))






