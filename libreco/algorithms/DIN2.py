"""

Reference: Guorui Zhou et al.  "Deep Interest Network for Click-Through Rate Prediction"
           (https://arxiv.org/pdf/1706.06978.pdf)

author: massquantity

"""
import os
import time
import random
import itertools
import numpy as np
import tensorflow as tf
from .Base import BasePure, BaseFeat
from ..evaluate.evaluate import precision_tf, MAP_at_k, MAR_at_k, recall_at_k, NDCG_at_k
from ..utils.sampling import NegativeSampling, NegativeSamplingFeat


class Din2(BaseFeat):
    def __init__(self, lr, n_epochs=20, embed_size=100, reg=0.0, batch_size=256, seed=42, num_att_items=100,
                 use_bn=True, hidden_units="128,64,32", dropout_rate=0.0, task="rating", neg_sampling=False,
                 include_item_feat=True):
        self.lr = lr
        self.n_epochs = n_epochs
        self.embed_size = embed_size
        self.reg = reg
        self.batch_size = batch_size
        self.seed = seed
        self.dropout_rate = dropout_rate
        self.bn = use_bn
        self.hidden_units = list(map(int, hidden_units.split(",")))
        self.task = task
        self.neg_sampling = neg_sampling
        self.num_att_items = num_att_items
        self.include_item_feat = include_item_feat
        super(Din2, self).__init__()

    def build_model(self, dataset):
        tf.set_random_seed(self.seed)
        self.dataset = dataset
        self.field_size = dataset.train_feat_indices.shape[1]
        self.feature_size = dataset.feature_size
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.global_mean = dataset.global_mean
        self.total_items_unique = self.item_info
        if dataset.lower_upper_bound is not None:
            self.lower_bound = dataset.lower_upper_bound[0]
            self.upper_bound = dataset.lower_upper_bound[1]
        else:
            self.lower_bound = None
            self.upper_bound = None
        if self.include_item_feat and dataset.item_feature_cols is not None:
            self.item_cols_num = len(dataset.item_feature_cols) + 1
        else:
            self.item_cols_num = 1
        self.item_feat_matrix = self.get_item_feat()


        self.feature_indices = tf.placeholder(tf.int32, shape=[None, self.field_size])
        self.feature_values = tf.placeholder(tf.float32, shape=[None, self.field_size])
        self.labels = tf.placeholder(tf.float32, shape=[None])
        # seq_matrix shape:  batch_size * max_seq_len * (item_feature_cols + item_indices)
        self.seq_matrix = tf.placeholder(tf.int32, shape=[None, None, self.item_cols_num])
        self.seq_len = tf.placeholder(tf.int32, shape=[None])
        self.is_training = tf.placeholder_with_default(False, shape=[])

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

        feature_values_reshape = tf.reshape(self.feature_values, shape=[-1, self.field_size, 1])

        feature_embedding = tf.nn.embedding_lookup(self.total_features, self.feature_indices)  # N * F * K
        feature_embedding = tf.multiply(feature_embedding, feature_values_reshape)
    #    self.feature_embedding = tf.reduce_sum(self.feature_embedding, axis=1)
        feature_embedding = tf.reshape(feature_embedding, [-1, self.field_size * self.embed_size])

    #    item_indices = tf.subtract(self.feature_indices[:, -1], dataset.user_offset)
    #    item_indices = tf.subtract(item_indices, dataset.n_users)
    #    item_indices = len(dataset.user_feature_cols) + len(dataset.item_feature_cols) + 1
        item_indices = self.field_size - 1
        if self.include_item_feat and dataset.item_feature_cols is not None:
            total_item_cols = dataset.item_feature_cols + [item_indices]
        else:
            total_item_cols = [item_indices]
        total_item_indices = tf.gather(self.feature_indices, total_item_cols, axis=1)
        item_embedding = tf.nn.embedding_lookup(self.total_features, total_item_indices)  # N * F_item * K
        item_embedding = tf.reshape(item_embedding, [-1, self.item_cols_num * self.embed_size])

        seq_embedding = tf.nn.embedding_lookup(self.total_features, self.seq_matrix)   # N * seq_len * F_item * K
        seq_max_len = tf.shape(self.seq_matrix)[1]
        seq_embedding = tf.reshape(seq_embedding, [-1, seq_max_len, self.item_cols_num * self.embed_size])

        attention_layer = self.attention(item_embedding, seq_embedding, self.seq_len)
        attention_layer = tf.layers.batch_normalization(attention_layer, training=self.is_training, momentum=0.99)
        attention_layer = tf.reshape(attention_layer, [-1, self.item_cols_num * self.embed_size])
    #    attention_layer = tf.layers.dense(attention_layer, self.embed_size, activation=None)

        concat_embedding = tf.concat([attention_layer, feature_embedding], axis=1)
        if self.bn:
            concat_embedding = tf.layers.batch_normalization(concat_embedding,
                                                             training=self.is_training,
                                                             momentum=0.99)

        for units in self.hidden_units:
            if self.bn:
                MLP_layer = tf.layers.dense(inputs=concat_embedding,
                                            units=units,
                                            activation=None,
                                            use_bias=False,
                                            kernel_initializer=tf.variance_scaling_initializer)
                                            # kernel_regularizer=tf.keras.regularizers.l2(0.0001)
                MLP_layer = tf.nn.relu(
                    tf.layers.batch_normalization(MLP_layer, training=self.is_training, momentum=0.99))
            else:
                MLP_layer = tf.layers.dense(inputs=concat_embedding,
                                            units=units,
                                            activation=tf.nn.relu,
                                            kernel_initializer=tf.variance_scaling_initializer)
            if self.dropout_rate > 0.0:
                MLP_layer = tf.layers.dropout(MLP_layer, rate=self.dropout_rate, training=self.is_training)

        if self.task == "rating":
            self.pred = tf.layers.dense(inputs=MLP_layer, units=1, name="pred")
            self.pred = tf.reshape(self.pred, [-1])
            self.loss = tf.losses.mean_squared_error(labels=tf.reshape(self.labels, [-1, 1]),
                                                     predictions=self.pred)

            if self.lower_bound is not None and self.upper_bound is not None:
                self.rmse = tf.sqrt(tf.losses.mean_squared_error(abels=tf.reshape(self.labels, [-1, 1]),
                                    predictions=tf.clip_by_value(self.pred, self.lower_bound, self.upper_bound)))
            else:
                self.rmse = self.loss

        elif self.task == "ranking":
            self.logits = tf.layers.dense(inputs=MLP_layer, units=1, name="logits")
            self.logits = tf.reshape(self.logits, [-1])
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits))

            self.y_prob = tf.sigmoid(self.logits)
            self.pred = tf.where(self.y_prob >= 0.5,
                                 tf.fill(tf.shape(self.logits), 1.0),
                                 tf.fill(tf.shape(self.logits), 0.0))
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.labels), tf.float32))
        #    self.precision = precision_tf(self.pred, self.labels)

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
                        seq_len, u_items_seq = self.preprocess_data(indices_batch, num_items=self.num_att_items)
                        self.sess.run(self.training_op, feed_dict={self.feature_indices: indices_batch,
                                                                   self.feature_values: values_batch,
                                                                   self.labels: labels_batch,
                                                                   self.seq_matrix: u_items_seq,
                                                                   self.seq_len: seq_len,
                                                                   self.is_training: True})
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
                    #    t7 = time.time()
                        seq_len, u_items_seq = self.preprocess_data(indices_batch, num_items=self.num_att_items)
                    #    print("prerpocess time: ", time.time() - t7)
                        self.sess.run(self.training_op, feed_dict={self.feature_indices: indices_batch,
                                                                   self.feature_values: values_batch,
                                                                   self.labels: labels_batch,
                                                                   self.seq_matrix: u_items_seq,
                                                                   self.seq_len: seq_len,
                                                                   self.is_training: True})

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
        seq_len, u_items_seq = self.preprocess_data(feat_indices)
        try:
            target = self.pred if self.task == "rating" else self.y_prob
            pred = self.sess.run(target, feed_dict={self.feature_indices: feat_indices,
                                                    self.feature_values: feat_value,
                                                    self.seq_matrix: u_items_seq,
                                                    self.seq_len: seq_len,
                                                    self.is_training: False})

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
        seq_len, u_items_seq = self.preprocess_data(feat_indices)
        preds = self.sess.run(target, feed_dict={self.feature_indices: feat_indices,
                                                 self.feature_values: feat_values,
                                                 self.seq_matrix: u_items_seq,
                                                 self.seq_len: seq_len,
                                                 self.is_training: False})

        if count < self.dataset.n_items:
            ids = np.argpartition(preds, -count)[-count:]
            rank = sorted(zip(ids, preds[ids]), key=lambda x: -x[1])
        else:
            rank = sorted(enumerate(preds), key=lambda x: -x[1])
        return list(itertools.islice((rec for rec in rank if rec[0] not in consumed), n_rec))

    @property
    def item_info(self):
        item_cols = [self.dataset.train_feat_indices.shape[1] - 1]
        if self.dataset.item_feature_cols is not None:
            item_cols += self.dataset.item_feature_cols
        return np.unique(self.dataset.train_feat_indices[:, item_cols], axis=0)

    def preprocess_data_orig(self, batch_data):
        max_seq_len = 0
        user_indices = batch_data[:, -2] - self.dataset.user_offset
        for u in user_indices:
            item_length = len(self.dataset.train_user[u])
            if item_length > max_seq_len:
                max_seq_len = item_length

        seq_len = list()
        u_items_seq = np.zeros((len(user_indices), max_seq_len), dtype=np.int64)
        for i, user in enumerate(user_indices):
            seq_len.append(len(self.dataset.train_user[user]))
            for j, item in enumerate(self.dataset.train_user[user]):
                u_items_seq[i, j] = item

        return seq_len, u_items_seq

    def preprocess_data(self, batch_data, num_items=100):
        max_seq_len = 0
        user_indices = batch_data[:, -2] - self.dataset.user_offset
        for u in user_indices:
            item_length = len(self.dataset.train_user[u])
            if item_length > max_seq_len:
                max_seq_len = item_length
            if max_seq_len > num_items:
                max_seq_len = num_items
                break

        seq_len = list()
        u_items_seq = np.zeros((len(user_indices), max_seq_len, self.item_cols_num), dtype=np.int64)
        for i, user in enumerate(user_indices):
            if len(self.dataset.train_user[user]) > num_items:
                u_items_len = num_items
            else:
                u_items_len = len(self.dataset.train_user[user])
            seq_len.append(u_items_len)
            items = list(self.dataset.train_user[user])
        #    items = [i for i in self.dataset.train_user[u] if self.dataset.train_user[u][i] >= 4]  choose liked items
        #    for j, item in enumerate(items[:num_items]):
        #        u_items_seq[i, j, :] = self.item_feat_dict[item]
            u_items_seq[i, :u_items_len, :] = self.item_feat_matrix[items[:num_items]]
        return seq_len, u_items_seq

    def attention(self, query, keys, keys_length):
        max_seq_len = tf.shape(keys)[1]
    #    max_seq_len = keys.get_shape().as_list()[1]
        queries = tf.tile(query, [1, max_seq_len])
        queries = tf.reshape(queries, [-1, max_seq_len, self.item_cols_num * self.embed_size])
        din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)
        layer = tf.layers.dense(din_all, 10, activation=tf.nn.elu)  # Prelu?
        layer = tf.layers.dense(layer, 1, activation=None)
        outputs = tf.reshape(layer, [-1, 1, max_seq_len])

        key_masks = tf.sequence_mask(keys_length, max_seq_len)
        key_masks = tf.expand_dims(key_masks, 1)
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(key_masks, outputs, paddings)
        outputs = outputs / np.sqrt(self.embed_size)  # divide sqrt(n)
        outputs = tf.nn.softmax(outputs)
        outputs = tf.matmul(outputs, keys)
        return outputs

    def neg_feat_dict_orig(self):
        neg_indices_dict = dict()
        total_items_col = self.dataset.item_feature_cols + [-1]
        total_items_unique = np.unique(self.dataset.train_feat_indices[:, total_items_col], axis=0)
        total_items = total_items_unique[:, -1]
    #    total_items_feat_col = np.delete(total_items_unique, 0, axis=1)

        for item, item_feat_col in zip(total_items, total_items_unique):
            item = item - self.dataset.user_offset - self.dataset.n_users
            neg_indices_dict[item] = item_feat_col.tolist()

        return neg_indices_dict

    def get_item_feat(self):
        item_feat_matrix = np.zeros((self.dataset.n_items, self.item_cols_num), dtype=np.int64)
        if self.include_item_feat and self.dataset.item_feature_cols is not None:
            total_items_col = self.dataset.item_feature_cols + [-1]
        else:
            total_items_col = [-1]
        total_items_unique = np.unique(self.dataset.train_feat_indices[:, total_items_col], axis=0)
        total_items = total_items_unique[:, -1]
    #    total_items_feat_col = np.delete(total_items_unique, 0, axis=1)

        for item, item_feat_col in zip(total_items, total_items_unique):
            item = item - self.dataset.user_offset - self.dataset.n_users
            item_feat_matrix[item] = item_feat_col

        return item_feat_matrix




