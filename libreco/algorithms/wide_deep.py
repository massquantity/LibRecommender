"""

Reference: Heng-Tze Cheng et al. "Wide & Deep Learning for Recommender Systems"  (https://arxiv.org/pdf/1606.07792.pdf)

author: massquantity

"""
import time, os
import itertools
from operator import itemgetter
from collections import OrderedDict
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column as feat_col
from tensorflow.python.estimator import estimator
from ..evaluate.evaluate import precision_tf, MAP_at_k, MAR_at_k, HitRatio_at_k, NDCG_at_k
from ..utils.sampling import NegativeSampling, NegativeSamplingFeat


# TODO batch_norm, dropout
class WideDeepEstimator(estimator.Estimator):  # tf.estimator.Estimator,  NOOOO inheritance
    def __init__(self, lr, embed_size, n_epochs=20, reg=0.0, cross_features=False,
                 batch_size=64, dropout=0.0, seed=42, task="rating"):
        self.lr = lr
        self.embed_size = embed_size
        self.n_epochs = n_epochs
        self.reg = reg
        self.batch_size = batch_size
        self.dropout = dropout
        self.seed = seed
        self.task = task
        self.cross_features = cross_features
        super(WideDeepEstimator, self).__init__(model_fn=WideDeepEstimator.model_func)

    def build_model(self, dataset):
        wide_cols = []
        deep_cols = []
        if self.cross_features:
            for i in range(len(dataset.feature_cols)):
                for j in range(i + 1, len(dataset.feature_cols)):
                    col1 = dataset.feature_cols[i]
                    col2 = dataset.feature_cols[j]
                    col1_feat = feat_col.categorical_column_with_vocabulary_list(
                        col1, dataset.col_unique_values[col1])
                    col2_feat = feat_col.categorical_column_with_vocabulary_list(
                        col2, dataset.col_unique_values[col2])
                    if (col1 not in ["user", "item", "title"]) and (col2 not in ["user", "item", "title"]):
                        wide_cols.append(feat_col.crossed_column([col1_feat, col2_feat], hash_bucket_size=1000))

            for col in dataset.feature_cols:
                col_feat = feat_col.categorical_column_with_vocabulary_list(col, dataset.col_unique_values[col])
                if col in ["user", "item", "title"]:
                    wide_cols.append(col_feat)
                    deep_cols.append(feat_col.embedding_column(col_feat, dimension=self.embed_size))
                elif len(dataset.col_unique_values[col]) < 10:
                    deep_cols.append(feat_col.indicator_column(col_feat))
                else:
                    deep_cols.append(feat_col.embedding_column(col_feat, dimension=self.embed_size))

        else:
            for col in dataset.feature_cols:
                col_feat = feat_col.categorical_column_with_vocabulary_list(col, dataset.col_unique_values[col])
                wide_cols.append(col_feat)
                if len(dataset.col_unique_values[col]) < 10:
                    deep_cols.append(feat_col.indicator_column(col_feat))
                else:
                    deep_cols.append(feat_col.embedding_column(col_feat, dimension=self.embed_size))

        config = tf.estimator.RunConfig(log_step_count_steps=100000, save_checkpoints_steps=100000)
        model = tf.estimator.Estimator(model_fn=WideDeepEstimator.model_func,
                                       model_dir="wide_deep_dir",
                                       config=config,
                                       params={'deep_columns': deep_cols,
                                               'wide_columns': wide_cols,
                                               'hidden_units': [128, 64],
                                               'task': self.task,
                                               'lr': self.lr})

        return model

    @staticmethod
    def model_func(features, labels, mode, params):
        dnn_input = feat_col.input_layer(features, params['deep_columns'])
        for units in params['hidden_units']:
            dnn_input = tf.layers.dense(dnn_input, units=units, activation=tf.nn.relu)
        dnn_logits = tf.layers.dense(dnn_input, units=10, activation=None)

        linear_logits = feat_col.linear_model(units=10, features=features,
                                              feature_columns=params['wide_columns'])

    #    logits = tf.add_n([dnn_logits, linear_logits])
        concat = tf.concat([dnn_logits, linear_logits], axis=-1)
        logits = tf.layers.dense(concat, units=1)
        if params['task'] == "rating":
            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {'predictions': logits}
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

            labels = tf.cast(tf.reshape(labels, [-1, 1]), tf.float32)
            loss = tf.losses.mean_squared_error(labels=labels, predictions=logits)
            rmse = tf.sqrt(tf.losses.mean_squared_error(labels=labels, predictions=tf.clip_by_value(logits, 1, 5)))
            metrics = {'rmse': rmse}

        elif params['task'] == "ranking":
            y_prob = tf.sigmoid(logits)
            pred = tf.where(y_prob >= 0.5,
                            tf.fill(tf.shape(y_prob), 1.0),
                            tf.fill(tf.shape(y_prob), 0.0))

            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {'class_ids': pred,
                               'probabilities': y_prob,
                               'logits': logits}
            #    predictions = y_prob[0]
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

            labels = tf.cast(tf.reshape(labels, [-1, 1]), tf.float32)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
            #    labels = tf.cast(labels, tf.int64)
            accuracy = tf.metrics.accuracy(labels=labels, predictions=pred)
            #    precision_at_k = tf.metrics.precision_at_k(labels=labels, predictions=logits, k=10)
            #    precision_at_k_2 = tf.metrics.precision_at_top_k(labels=labels, predictions_idx=pred, k=10)
            precision = tf.metrics.precision(labels=labels, predictions=pred)
            recall = tf.metrics.recall(labels=labels, predictions=pred)
            f1 = tf.contrib.metrics.f1_score(labels=labels, predictions=pred)
            auc_roc = tf.metrics.auc(labels=labels, predictions=pred, curve="ROC",
                                     summation_method='careful_interpolation')
            auc_pr = tf.metrics.auc(labels=labels, predictions=pred, curve="PR",
                                    summation_method='careful_interpolation')
            metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1,
                       'auc_roc': auc_roc, 'auc_pr': auc_pr}

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

        assert mode == tf.estimator.ModeKeys.TRAIN

    #    optimizer = tf.train.FtrlOptimizer(learning_rate=0.1, l1_regularization_strength=1e-3)
        optimizer = tf.train.AdamOptimizer(learning_rate=params['lr'])
    #    optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
    #    optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=0.01)
        training_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=training_op)

    @staticmethod
    def input_fn(data, repeat=1, batch=256, mode="train", user=None, item=None):  # , original_data=None
        if mode == "train":
            features = {col: data.train_data[col].to_numpy() for col in data.feature_cols}
        #    features = {"age": data.train_data["age"].to_numpy()}
            labels = data.train_data[data.label_cols].to_numpy()

            train_data = tf.data.Dataset.from_tensor_slices((features, labels))
            return train_data.shuffle(len(labels)).repeat(repeat).batch(batch)

        elif mode == "evaluate":
            features = {col: data.test_data[col].to_numpy().reshape(-1, 1) for col in data.feature_cols}
            labels = data.test_data[data.label_cols].to_numpy().reshape(-1, 1)

            evaluate_data = tf.data.Dataset.from_tensor_slices((features, labels))
            return evaluate_data

        elif mode == "test":
            user_part = pd.DataFrame([data.user_dict[user]], columns=data.user_feature_cols)
            item_part = pd.DataFrame([data.item_dict[item]], columns=data.item_feature_cols)
            features = {col: user_part[col].to_numpy().reshape(-1, 1) for col in user_part.columns}
            features.update({col: item_part[col].to_numpy().reshape(-1, 1) for col in item_part.columns})
            for col, col_type in data.column_types:
                if col_type == np.float32 or col_type == np.float64:
                    features[col] = features[col].astype(int)
                elif col != "label":
                    features[col] = features[col].astype(col_type)

            test_data = tf.data.Dataset.from_tensor_slices(features)
            return test_data

        elif mode == "rank":
            n_items = len(data.item_dict)
            user_part = pd.DataFrame([data.user_dict[user]], columns=data.user_feature_cols)
            user_part = user_part.reindex(user_part.index.repeat(n_items))
            item_part = pd.DataFrame(list(data.item_dict.values()), columns=data.item_feature_cols)
            features = {col: user_part[col].to_numpy().reshape(-1, 1) for col in user_part.columns}
            features.update({col: item_part[col].to_numpy().reshape(-1, 1) for col in item_part.columns})
            for col, col_type in data.column_types:
                if col_type == np.float32 or col_type == np.float64:
                    features[col] = features[col].astype(int)
                elif col != "label":
                    features[col] = features[col].astype(col_type)
            rank_data = tf.data.Dataset.from_tensor_slices(features).batch(batch)
            return rank_data

    def fit(self, dataset, verbose=1):
        self.dataset = dataset
        self.model = self.build_model(dataset)
        for epoch in range(1, self.n_epochs):  # epoch_per_eval
            t0 = time.time()
            self.model.train(input_fn=lambda: WideDeepEstimator.input_fn(
                data=dataset, repeat=1, batch=self.batch_size, mode="train"))
            train_result = self.model.evaluate(input_fn=lambda: WideDeepEstimator.input_fn(
                data=dataset, repeat=1, batch=self.batch_size, mode="train"))
            eval_result = self.model.evaluate(input_fn=lambda: WideDeepEstimator.input_fn(
                data=dataset, mode="evaluate"))

            if verbose > 0:
                print("Epoch {} training time: {:.4f}".format(epoch, time.time() - t0))
                if self.task == "rating":
                    print("train loss: {loss:.4f}, train rmse: {rmse:.4f}".format(**train_result))
                    print("test loss: {loss:.4f}, test rmse: {rmse:.4f}".format(**eval_result))
                elif self.task == "ranking":
                    print("train loss: {loss:.4f}, accuracy: {accuracy:.4f}, precision: {precision:.4f}, "
                          "recall: {recall:.4f}, f1: {f1:.4f}, auc_roc: {auc_roc:.4f}, "
                          "auc_pr: {auc_pr:.4f}".format(**train_result))
                    print("test loss: {loss:.4f}, accuracy: {accuracy:.4f}, precision: {precision:.4f}, "
                          "recall: {recall:.4f}, f1: {f1:.4f}, auc_roc: {auc_roc:.4f}, "
                          "auc_pr: {auc_pr:.4f}".format(**eval_result))

                t0 = time.time()
                NDCG = NDCG_at_k(self, self.dataset, 10, mode="wide_deep")
                print("\t NDCG @ {}: {:.4f}".format(10, NDCG))
                print("\t NDCG time: {:.4f}".format(time.time() - t0))

    def predict_ui(self, u, i):  # cannot override Estimator's predict method
        pred_result = self.model.predict(input_fn=lambda: WideDeepEstimator.input_fn(
            data=self.dataset, mode="test", user=u, item=i))  #  original_data=self.dataset
        pred_result = list(pred_result)[0]
        if self.task == "rating":
            return list(pred_result)[0]['predictions']
        elif self.task == "ranking":
            return pred_result['probabilities'][0], pred_result['class_ids'][0]

    def recommend_user(self, u, n_rec=10):
        rank_list = self.model.predict(input_fn=lambda: WideDeepEstimator.input_fn(
            data=self.dataset, mode="rank", user=u, batch=256))

        items = np.array(list(self.dataset.item_dict.keys()))
        if self.task == "rating":
            rank = np.array([res['predictions'][0] for res in rank_list])
            indices = np.argpartition(rank, -n_rec)[-n_rec:]
            return sorted(zip(items[indices], rank[indices]), key=lambda x: -x[1])

        elif self.task == "ranking":
            rank = np.array([res['probabilities'][0] for res in rank_list])
            indices = np.argpartition(rank, -n_rec)[-n_rec:]
            return sorted(zip(items[indices], rank[indices]), key=itemgetter(1), reverse=True)


class WideDeep:
    def __init__(self, lr, n_epochs=20, embed_size=100, reg=0.0, batch_size=256, seed=42, 
                 dropout_rate=0.0, task="rating"):
        self.lr = lr
        self.n_epochs = n_epochs
        self.embed_size = embed_size
        self.reg = reg
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.task = task

    def build_model(self, dataset):
        tf.set_random_seed(self.seed)
        self.dataset = dataset
        self.field_size = dataset.train_feat_indices.shape[1]
        self.feature_size = dataset.feature_size
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.total_items_unique = self.item_info

        self.feature_indices = tf.placeholder(tf.int32, shape=[None, self.field_size])
        self.feature_values = tf.placeholder(tf.float32, shape=[None, self.field_size])
        self.labels = tf.placeholder(tf.float32, shape=[None])

        self.w = tf.Variable(tf.truncated_normal([self.feature_size + 1, 1], 0.0, 0.01))  # feature_size + 1
        self.v = tf.Variable(tf.truncated_normal([self.feature_size + 1, self.embed_size], 0.0, 0.01))
        self.feature_values_reshape = tf.reshape(self.feature_values, shape=[-1, self.field_size, 1])

        self.linear_embedding = tf.nn.embedding_lookup(self.w, self.feature_indices)   # N * F * 1
        self.linear_term = tf.reduce_sum(tf.multiply(self.linear_embedding, self.feature_values_reshape), 2)  # axis=1?

        self.MLP_embedding = tf.nn.embedding_lookup(self.v, self.feature_indices)  # N * F * K
        self.MLP_embedding = tf.multiply(self.MLP_embedding, self.feature_values_reshape)
        self.MLP_embedding = tf.reshape(self.MLP_embedding, [-1, self.field_size * self.embed_size])

        self.MLP_layer1 = tf.layers.dense(inputs=self.MLP_embedding,
                                          units=self.embed_size * 2,
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.variance_scaling_initializer)
        self.MLP_layer1 = tf.layers.dropout(self.MLP_layer1, rate=self.dropout_rate)
        self.MLP_layer2 = tf.layers.dense(inputs=self.MLP_layer1,
                                          units=self.embed_size,
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.variance_scaling_initializer)
        self.MLP_layer2 = tf.layers.dropout(self.MLP_layer2, rate=self.dropout_rate)
        self.MLP_layer3 = tf.layers.dense(inputs=self.MLP_layer2,
                                          units=self.embed_size,
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.variance_scaling_initializer)
        self.MLP_layer3 = tf.layers.dropout(self.MLP_layer3, rate=self.dropout_rate)

        self.concat_layer = tf.concat([self.linear_term, self.MLP_layer3], axis=1)

        if self.task == "rating":
            self.pred = tf.layers.dense(inputs=self.concat_layer, units=1)
            self.loss = tf.losses.mean_squared_error(labels=tf.reshape(self.labels, [-1, 1]),
                                                     predictions=self.pred)
            self.rmse = tf.sqrt(tf.losses.mean_squared_error(labels=tf.reshape(self.labels, [-1, 1]),
                                                             predictions=tf.clip_by_value(self.pred, 1, 5)))

        elif self.task == "ranking":
            self.logits = tf.layers.dense(inputs=self.concat_layer, units=1, name="logits")
            self.logits = tf.reshape(self.logits, [-1])
            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits))

            self.y_prob = tf.sigmoid(self.logits)
            self.pred = tf.where(self.y_prob >= 0.5,
                                 tf.fill(tf.shape(self.logits), 1.0),
                                 tf.fill(tf.shape(self.logits), 0.0))
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.labels), tf.float32))
            self.precision = precision_tf(self.pred, self.labels)

    def fit(self, dataset, verbose=1, pre_sampling=True):
        self.build_model(dataset)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
    #    self.optimizer = tf.train.FtrlOptimizer(learning_rate=0.1, l1_regularization_strength=1e-3)
        self.training_op = self.optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.sess.run(tf.local_variables_initializer())
        with self.sess.as_default():
            if self.task == "rating":
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

                    if verbose > 0:
               #         train_rmse = self.rmse.eval(feed_dict={self.feature_indices: dataset.train_feat_indices,
                #                                               self.feature_values: dataset.train_feat_values,
                #                                               self.labels: dataset.train_labels})

                        test_loss, test_rmse = self.sess.run([self.loss, self.rmse],
                                                              feed_dict={
                                                                  self.feature_indices: dataset.test_feat_indices,
                                                                  self.feature_values: dataset.test_feat_values,
                                                                  self.labels: dataset.test_labels})

                #        print("Epoch {}, train_rmse: {:.4f}, training_time: {:.2f}".format(
                #                epoch, train_rmse, time.time() - t0))
                        print("Epoch {}, training_time: {:.2f}".format(epoch, time.time() - t0))
                        print("Epoch {}, test_loss: {:.4f}, test_rmse: {:.4f}".format(
                            epoch, test_loss, test_rmse))
                        print()

            elif self.task == "ranking":
                for epoch in range(1, self.n_epochs + 1):
                    t0 = time.time()
                    neg = NegativeSamplingFeat(dataset, dataset.num_neg, self.batch_size, pre_sampling=pre_sampling)
                    n_batches = int(np.ceil(len(dataset.train_labels_implicit) / self.batch_size))
                    for n in range(n_batches):
                        indices_batch, values_batch, labels_batch = neg.next_batch()
                        self.sess.run(self.training_op, feed_dict={self.feature_indices: indices_batch,
                                                                   self.feature_values: values_batch,
                                                                   self.labels: labels_batch})

                    if verbose > 0:
                        print("Epoch {}: training time: {:.4f}".format(epoch, time.time() - t0))
                        t3 = time.time()
                #        train_loss, train_accuracy, train_precision = \
                #            self.sess.run([self.loss, self.accuracy, self.precision],
                #                          feed_dict={self.feature_indices: dataset.train_indices_implicit,
                #                                     self.feature_values: dataset.train_values_implicit,
                #                                     self.labels: dataset.train_labels_implicit})

                        test_pred, test_loss, test_accuracy, test_precision = \
                            self.sess.run([self.pred, self.loss, self.accuracy, self.precision],
                                          feed_dict={self.feature_indices: dataset.test_indices_implicit,
                                                     self.feature_values: dataset.test_values_implicit,
                                                     self.labels: dataset.test_labels_implicit})

                        print("\ttest loss: {:.4f}".format(test_loss))
                        print("\ttest accuracy: {:.4f}".format(test_accuracy))
                        print("\ttest precision: {:.4f}".format(test_precision))
                        print("\tloss time: {:.4f}".format(time.time() - t3))

                        t4 = time.time()
                        mean_average_precision_10 = MAP_at_k(self, self.dataset, 10, sample_user=1000)
                        print("\t MAP@{}: {:.4f}".format(10, mean_average_precision_10))
                        print("\t MAP@10 time: {:.4f}".format(time.time() - t4))

                        t5 = time.time()
                        mean_average_recall_50 = MAR_at_k(self, self.dataset, 50, sample_user=1000)
                        print("\t MAR@{}: {:.4f}".format(50, mean_average_recall_50))
                        print("\t MAR@50 time: {:.4f}".format(time.time() - t5))

                        t6 = time.time()
                        NDCG = NDCG_at_k(self, self.dataset, 10, sample_user=1000)
                        print("\t NDCG@{}: {:.4f}".format(10, NDCG))
                        print("\t NDCG@10 time: {:.4f}".format(time.time() - t6))
                        print()

    def predict(self, user, item):
        feat_indices, feat_value = self.get_predict_indices_and_values(self.dataset, user, item)
        try:
            target = self.pred if self.task == "rating" else self.y_prob
            pred = self.sess.run(target, feed_dict={self.feature_indices: feat_indices,
                                                       self.feature_values: feat_value})
            pred = np.clip(pred, 1, 5) if self.task == "rating" else pred[0]
        except tf.errors.InvalidArgumentError:
            pred = self.dataset.global_mean
        return pred

    def recommend_user(self, u, n_rec):
        consumed = self.dataset.train_user[u]
        count = n_rec + len(consumed)
        target = self.pred if self.task == "rating" else self.y_prob

        feat_indices, feat_values = self.get_recommend_indices_and_values(self.dataset, u)
        preds = self.sess.run(target, feed_dict={self.feature_indices: feat_indices,
                                                 self.feature_values: feat_values})

        ids = np.argpartition(preds, -count)[-count:]
        rank = sorted(zip(ids, preds[ids]), key=lambda x: -x[1])
        return list(itertools.islice((rec for rec in rank if rec[0] not in consumed), n_rec))

    def get_predict_indices_and_values(self, data, user, item):
        user_col = data.train_feat_indices.shape[1] - 2
        item_col = data.train_feat_indices.shape[1] - 1

        user_repr = user + data.user_offset
        user_cols = data.user_feature_cols + [user_col]
        user_features = data.train_feat_indices[:, user_cols]
        user = user_features[user_features[:, -1] == user_repr][0]

        item_repr = item + data.user_offset + data.n_users
        item_cols = [item_col] + data.item_feature_cols
        item_features = data.train_feat_indices[:, item_cols]
        item = item_features[item_features[:, 0] == item_repr][0]

        orig_cols = user_cols + item_cols
        col_reindex = np.array(range(len(orig_cols)))[np.argsort(orig_cols)]
        concat_indices = np.concatenate([user, item])[col_reindex]

        feat_values = np.ones(len(concat_indices))
        if data.numerical_col is not None:
            for col in range(len(data.numerical_col)):
                if col in data.user_feature_cols:
                    user_indices = np.where(data.train_feat_indices[:, user_col] == user_repr)[0]
                    numerical_values = data.train_feat_values[user_indices, col][0]
                    feat_values[col] = numerical_values
                elif col in data.item_feature_cols:
                    item_indices = np.where(data.train_feat_indices[:, item_col] == item_repr)[0]
                    numerical_values = data.train_feat_values[item_indices, col][0]
                    feat_values[col] = numerical_values

        return concat_indices.reshape(1, -1), feat_values.reshape(1, -1)

    def get_recommend_indices_and_values(self, data, user):
        user_col = data.train_feat_indices.shape[1] - 2
        item_col = data.train_feat_indices.shape[1] - 1

        user_repr = user + data.user_offset
        user_cols = data.user_feature_cols + [user_col]
        user_features = data.train_feat_indices[:, user_cols]
        user_unique = user_features[user_features[:, -1] == user_repr][0]
        users = np.tile(user_unique, (data.n_items, 1))

        #   np.unique is sorted from starting with the first element, so put item col first
        item_cols = [item_col] + data.item_feature_cols
        orig_cols = user_cols + item_cols
        col_reindex = np.array(range(len(orig_cols)))[np.argsort(orig_cols)]

        assert users.shape[0] == self.total_items_unique.shape[0], "user shape must equal to num of candidate items"
        concat_indices = np.concatenate([users, self.total_items_unique], axis=-1)[:, col_reindex]

        #   construct feature values, mainly fill numerical columns
        feat_values = np.ones(shape=(data.n_items, concat_indices.shape[1]))
        if data.numerical_col is not None:
            numerical_dict = OrderedDict()
            for col in range(len(data.numerical_col)):
                if col in data.user_feature_cols:
                    user_indices = np.where(data.train_feat_indices[:, user_col] == user_repr)[0]
                    numerical_values = data.train_feat_values[user_indices, col][0]
                    numerical_dict[col] = numerical_values
                elif col in data.item_feature_cols:
                    # order according to item indices
                    numerical_map = OrderedDict(
                                        sorted(
                                            zip(data.train_feat_indices[:, -1],
                                                data.train_feat_values[:, col]), key=lambda x: x[0]))
                    numerical_dict[col] = [v for v in numerical_map.values()]

            for k, v in numerical_dict.items():
                feat_values[:, k] = np.array(v)

        return concat_indices, feat_values

    @property
    def item_info(self):
        item_col = self.dataset.train_feat_indices.shape[1] - 1
        item_cols = [item_col] + self.dataset.item_feature_cols
        return np.unique(self.dataset.train_feat_indices[:, item_cols], axis=0)








