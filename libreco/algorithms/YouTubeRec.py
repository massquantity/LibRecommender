"""

Reference: Paul Covington et al.  "Deep Neural Networks for YouTube Recommendations"
           (https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf)

author: massquantity

"""
import os
import time
import itertools
from collections import OrderedDict
import numpy as np
import tensorflow as tf
from .Base import BasePure, BaseFeat
from ..evaluate.evaluate import precision_tf, MAP_at_k, MAR_at_k, recall_at_k, NDCG_at_k
from ..utils.sampling import NegativeSampling, NegativeSamplingFeat


class YouTubeRec(BaseFeat):
    def __init__(self, lr, n_epochs=20, embed_size=100, reg=0.0, batch_size=256, seed=42,
                 use_bn=True, dropout_rate=0.0, task="rating", neg_sampling=False):
        self.lr = lr
        self.n_epochs = n_epochs
        self.embed_size = embed_size
        self.reg = reg
        self.batch_size = batch_size
        self.seed = seed
        self.dropout_rate = dropout_rate
        self.bn = use_bn
        self.task = task
        self.neg_sampling = neg_sampling
        super(YouTubeRec, self).__init__()

    def build_model(self, dataset):
        tf.set_random_seed(self.seed)
        self.dataset = dataset
        self.field_size = dataset.train_feat_indices.shape[1]
        self.feature_size = dataset.feature_size
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.global_mean = dataset.global_mean
        self.total_items_unique = self.item_info
        implicit_feedback = self.get_implicit_feedback(dataset)
        if dataset.lower_upper_bound is not None:
            self.lower_bound = dataset.lower_upper_bound[0]
            self.upper_bound = dataset.lower_upper_bound[1]
        else:
            self.lower_bound = None
            self.upper_bound = None

        self.feature_indices = tf.placeholder(tf.int32, shape=[None, self.field_size])
        self.feature_values = tf.placeholder(tf.float32, shape=[None, self.field_size])
        self.labels = tf.placeholder(tf.float32, shape=[None])
        self.implicit_indices = tf.placeholder(tf.int64, shape=[None, 2])
        self.implicit_values = tf.placeholder(tf.int64, shape=[None])
        self.dropout_switch = tf.placeholder_with_default(False, shape=[])
        self.bn_switch = tf.placeholder_with_default(False, shape=[])

        if self.reg > 0.0:
            self.total_features = tf.get_variable(name="total_features",
                                                  shape=[self.feature_size + 1, self.embed_size],
                                                  initializer=tf.initializers.truncated_normal(0.0, 0.01),
                                                  regularizer=tf.keras.regularizers.l2(self.reg))
        else:
            self.total_features = tf.get_variable(name="total_features",
                                                  shape=[self.feature_size + 1, self.embed_size],
                                                  initializer=tf.initializers.truncated_normal(0.0, 0.01),
                                                  regularizer=None)

        if self.reg > 0.0:
            self.implicit_features = tf.get_variable(name="implicit_features",
                                                     shape=[dataset.n_items, self.embed_size],
                                                     initializer=tf.initializers.truncated_normal(0.0, 0.01),
                                                     regularizer=tf.keras.regularizers.l2(self.reg))
        else:
            self.implicit_features = tf.get_variable(name="implicit_features",
                                                     shape=[dataset.n_items, self.embed_size],
                                                     initializer=tf.initializers.truncated_normal(0.0, 0.01),
                                                     regularizer=None)

        self.feature_values_reshape = tf.reshape(self.feature_values, shape=[-1, self.field_size, 1])

        self.feature_embedding = tf.nn.embedding_lookup(self.total_features, self.feature_indices)  # N * F * K
        self.feature_embedding = tf.multiply(self.feature_embedding, self.feature_values_reshape)
    #    self.feature_embedding = tf.reduce_sum(self.feature_embedding, axis=1)
        self.feature_embedding = tf.reshape(self.feature_embedding, [-1, self.field_size * self.embed_size])

        self.user_embedding = tf.nn.embedding_lookup_sparse(self.implicit_features, implicit_feedback,
                                                            sp_weights=None, combiner="sqrtn")
        user_indices = tf.subtract(self.feature_indices[:, -2], dataset.user_offset)
        self.user_embedding = tf.nn.embedding_lookup(self.user_embedding, user_indices)
        self.concat_embedding = tf.concat([self.user_embedding, self.feature_embedding], axis=1)

        MLP_layer_one = tf.layers.dense(inputs=self.concat_embedding,
                                        units=self.embed_size * 3,
                                        activation=None,
                                        kernel_initializer=tf.variance_scaling_initializer)
                                        # kernel_regularizer=tf.keras.regularizers.l2(0.0001)

        if self.bn:
            MLP_layer_one = tf.layers.batch_normalization(MLP_layer_one, training=self.bn_switch, momentum=0.9)
        MLP_layer_one = tf.nn.relu(MLP_layer_one)
        if self.dropout_rate > 0.0:
            MLP_layer_one = tf.layers.dropout(MLP_layer_one, rate=self.dropout_rate, training=self.dropout_switch)

        MLP_layer_two = tf.layers.dense(inputs=MLP_layer_one,
                                        units=self.embed_size * 2,
                                        activation=None,
                                        kernel_initializer=tf.variance_scaling_initializer)

        if self.bn:
            MLP_layer_two = tf.layers.batch_normalization(MLP_layer_two, training=self.bn_switch, momentum=0.9)
        MLP_layer_two = tf.nn.relu(MLP_layer_two)
        if self.dropout_rate > 0.0:
            MLP_layer_two = tf.layers.dropout(MLP_layer_two, rate=self.dropout_rate, training=self.dropout_switch)

        MLP_layer_three = tf.layers.dense(inputs=MLP_layer_two,
                                          units=self.embed_size,
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.variance_scaling_initializer)
    #    MLP_layer_three = tf.layers.dropout(MLP_layer_three, rate=dropout_rate, training=dropout_switch)
    #    MLP_layer_three = tf.layers.batch_normalization(MLP_layer_three, training=self.bn_switch)

        if self.task == "rating":
            self.pred = tf.layers.dense(inputs=MLP_layer_three, units=1)
            self.loss = tf.losses.mean_squared_error(labels=tf.reshape(self.labels, [-1, 1]),
                                                     predictions=self.pred)

            if self.lower_bound is not None and self.upper_bound is not None:
                self.rmse = tf.sqrt(tf.losses.mean_squared_error(abels=tf.reshape(self.labels, [-1, 1]),
                                    predictions=tf.clip_by_value(self.pred, self.lower_bound, self.upper_bound)))
            else:
                self.rmse = self.loss

        elif self.task == "ranking":
            self.logits = tf.layers.dense(inputs=MLP_layer_three, units=1, name="logits")
            self.logits = tf.reshape(self.logits, [-1])
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits))

            self.y_prob = tf.sigmoid(self.logits)
            self.pred = tf.where(self.y_prob >= 0.5,
                                 tf.fill(tf.shape(self.logits), 1.0),
                                 tf.fill(tf.shape(self.logits), 0.0))
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.labels), tf.float32))
            self.precision = precision_tf(self.pred, self.labels)

        if self.reg > 0.0:
            keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.total_loss = self.loss + tf.add_n(keys)
        else:
            self.total_loss = self.loss

    def fit(self, dataset, verbose=1, pre_sampling=True, **kwargs):
        print("start time: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        self.build_model(dataset)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
    #    self.optimizer = tf.train.FtrlOptimizer(learning_rate=0.1, l1_regularization_strength=1e-3)
        self.training_op = self.optimizer.minimize(self.total_loss)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.training_op = tf.group([self.training_op, update_ops])
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess = tf.Session()
        self.sess.run(init)
        with self.sess.as_default():
            if self.task == "rating" or (self.task == "ranking" and not self.neg_sampling):
                for epoch in range(1, self.n_epochs + 1):
                    t0 = time.time()
                    n_batches = int(np.ceil(len(dataset.train_labels) / self.batch_size))
                    for n in range(n_batches):
                        end = min(len(dataset.train_labels), (n + 1) * self.batch_size)
                        indices_batch = dataset.train_feat_indices[n * self.batch_size: end]
                        values_batch = dataset.train_feat_values[n * self.batch_size: end]
                        labels_batch = dataset.train_labels[n * self.batch_size: end]

                        self.sess.run(self.training_op, feed_dict={self.feature_indices: indices_batch,
                                                                   self.feature_values: values_batch,
                                                                   self.labels: labels_batch,
                                                                   self.dropout_switch: True,
                                                                   self.bn_switch: True})
                    if verbose == 1:
                        print("Epoch {}, training_time: {:.2f}".format(epoch, time.time() - t0), end="\n\n")
                    elif verbose > 1:
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
                    neg = NegativeSamplingFeat(dataset, dataset.num_neg, self.batch_size, pre_sampling=pre_sampling)
                    n_batches = int(np.ceil(len(dataset.train_labels_implicit) / self.batch_size))
                    for n in range(n_batches):
                        indices_batch, values_batch, labels_batch = neg.next_batch()
                        self.sess.run(self.training_op, feed_dict={self.feature_indices: indices_batch,
                                                                   self.feature_values: values_batch,
                                                                   self.labels: labels_batch,
                                                                   self.dropout_switch: True,
                                                                   self.bn_switch: True})

                    if verbose == 1:
                        print("Epoch {}, training_time: {:.2f}".format(epoch, time.time() - t0))
                    elif verbose > 1:
                        print("Epoch {}: training time: {:.4f}".format(epoch, time.time() - t0))
                        metrics = kwargs.get("metrics", self.metrics)
                        if hasattr(self, "sess"):
                            self.print_metrics_tf(dataset, epoch, **metrics)
                        else:
                            self.print_metrics(dataset, epoch, **metrics)
                        print()

    def predict(self, user, item):
        feat_indices, feat_value = self.get_predict_indices_and_values(self.dataset, user, item)
        try:
            target = self.pred if self.task == "rating" else self.y_prob
            pred = self.sess.run(target, feed_dict={self.feature_indices: feat_indices,
                                                    self.feature_values: feat_value,
                                                    self.dropout_switch: False,
                                                    self.bn_switch: False})

            if self.lower_bound is not None and self.upper_bound is not None:
                pred = np.clip(pred, self.lower_bound, self.upper_bound) if self.task == "rating" else pred[0]
        except tf.errors.InvalidArgumentError:
            pred = self.dataset.global_mean if self.task == "rating" else 0.0
        return pred

    def recommend_user(self, u, n_rec):
        consumed = self.dataset.train_user[u]
        count = n_rec + len(consumed)
        target = self.pred if self.task == "rating" else self.y_prob

        feat_indices, feat_values = self.get_recommend_indices_and_values(self.dataset, u, self.total_items_unique)
    #    user_batch = feat_indices[:, -2] - self.dataset.user_offset
        preds = self.sess.run(target, feed_dict={self.feature_indices: feat_indices,
                                                 self.feature_values: feat_values,
                                                 self.dropout_switch: False,
                                                 self.bn_switch: False})

        ids = np.argpartition(preds, -count)[-count:]
        rank = sorted(zip(ids, preds[ids]), key=lambda x: -x[1])
        return list(itertools.islice((rec for rec in rank if rec[0] not in consumed), n_rec))

    @property
    def item_info(self):
        item_col = self.dataset.train_feat_indices.shape[1] - 1
        item_cols = [item_col] + self.dataset.item_feature_cols
        return np.unique(self.dataset.train_feat_indices[:, item_cols], axis=0)

    def get_implicit_feedback(self, data):
        user_split_items = [[] for u in range(data.n_users)]
        for u, i in zip(data.train_user_indices, data.train_item_indices):
            user_split_items[u].append(i)

        sparse_dict = {'indices': [], 'values': []}
        for i, user in enumerate(user_split_items):
            for j, item in enumerate(user):
                sparse_dict['indices'].append((i, j))
                sparse_dict['values'].append(item)
        sparse_dict['dense_shape'] = (data.n_users, data.n_items)
        implicit_feedback = tf.SparseTensor(**sparse_dict)
        return implicit_feedback

#    def reg_term(self):
#        return tf.cond(tf.constant(self.reg > 0.0), lambda: tf.keras.regularizers.l2(self.reg), lambda: None)





