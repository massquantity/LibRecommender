"""

Reference: Heng-Tze Cheng et al. "Wide & Deep Learning for Recommender Systems"  (https://arxiv.org/pdf/1606.07792.pdf)

author: massquantity

"""
import time, os
import itertools
from operator import itemgetter
from collections import OrderedDict
import numpy as np
import tensorflow as tf
from tensorflow import feature_column as feat_col
# from tensorflow.python.estimator import estimator
from .Base import BasePure, BaseFeat
from ..evaluate.evaluate import precision_tf, MAP_at_k, MAR_at_k, HitRatio_at_k, NDCG_at_k
from ..utils.sampling import NegativeSampling, NegativeSamplingFeat


class WideDeepEstimator(tf.estimator.Estimator):
    def __init__(self, lr, embed_size, n_epochs=20, reg=0.0, batch_size=64,
                 use_bn=False, hidden_units="256,128,64", dropout=0.0, seed=42, eval_top_n="10,50",
                 task="rating", pred_feat_func=None, rank_feat_func=None, item_indices=None):
        self.lr = lr
        self.embed_size = embed_size
        self.n_epochs = n_epochs
        self.reg = reg
        self.batch_size = batch_size
        self.dropout = dropout
        self.use_bn = use_bn
        self.hidden_units = hidden_units
        self.eval_top_n = eval_top_n
        self.seed = seed
        self.task = task
        self.pred_feat_func = pred_feat_func
        self.rank_feat_func = rank_feat_func
        self.item_indices = item_indices
        super(WideDeepEstimator, self).__init__(model_fn=WideDeepEstimator.model_func)

    def build_model(self, wide_cols, deep_cols):
        config = tf.estimator.RunConfig(log_step_count_steps=100000, save_checkpoints_steps=100000)
        model = tf.estimator.Estimator(model_fn=WideDeepEstimator.model_func,
                                       model_dir="wide_deep_dir",
                                       config=config,
                                       params={'deep_columns': deep_cols,
                                               'wide_columns': wide_cols,
                                               'hidden_units': self.hidden_units.split(","),
                                               'use_bn': self.use_bn,
                                               'task': self.task,
                                               'lr': self.lr,
                                               'n_items': len(self.item_indices),
                                               'eval_top_n': self.eval_top_n})

        return model

    @staticmethod
    def model_func(features, labels, mode, params):
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        dnn_input = feat_col.input_layer(features, params['deep_columns'])
        if params["use_bn"]:
            dnn_input = tf.layers.batch_normalization(dnn_input, training=is_training, momentum=0.99)
        for units in params['hidden_units']:
            dnn_input = tf.layers.dense(dnn_input, units=units, activation=None,
                                        kernel_initializer=tf.variance_scaling_initializer)
            if params["use_bn"]:
                dnn_input = tf.layers.dense(dnn_input, units=units, activation=None, use_bias=False)
                dnn_input = tf.nn.relu(tf.layers.batch_normalization(dnn_input, training=is_training, momentum=0.99))
            else:
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

        #    item_labels = tf.one_hot(features["item"], params["n_items"])
            item_labels = features["item"]
            item_logits = tf.layers.dense(concat, units=params["n_items"])

        else:
            raise ValueError("task should be `rating` or `ranking`")

        if mode == tf.estimator.ModeKeys.EVAL:
            if params["task"] == "rating":
                rmse = tf.sqrt(tf.losses.mean_squared_error(labels=labels, predictions=tf.clip_by_value(logits, 1, 5)))
                metrics = {'rmse': rmse}
            elif params["task"] == "ranking":
                metrics = dict()
                metrics["auc_roc"] = tf.metrics.auc(labels=labels, predictions=y_prob, curve="ROC",
                                                    summation_method='careful_interpolation')
                metrics["auc_pr"] = tf.metrics.auc(labels=labels, predictions=y_prob, curve="PR",
                                                   summation_method='careful_interpolation')
                metrics["recall/recall@10"] = tf.metrics.recall_at_k(item_labels, item_logits, 10)
                metrics["precision/precision@10"] = tf.metrics.precision_at_k(item_labels, item_logits, 10)
            else:
                raise ValueError("task should be `rating` or `ranking`")

            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

        assert mode == tf.estimator.ModeKeys.TRAIN
    #    optimizer = tf.train.FtrlOptimizer(learning_rate=0.1, l1_regularization_strength=1e-3)
        optimizer = tf.train.AdamOptimizer(learning_rate=params['lr'])
    #    optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
    #    optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=0.01)
        training_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        training_op = tf.group([training_op, update_ops])
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=training_op)

    @staticmethod
    def input_fn(file_path=None, batch_size=256, mode="train", user=None, item=None, feature_func=None):
        def _parse_record(record):
            example = tf.io.parse_single_example(record, features=feature_func)
            labels = example.pop("label")
            return example, labels

        if mode == "train":
            dataset = tf.data.TFRecordDataset([file_path])
            dataset = dataset.map(_parse_record, num_parallel_calls=4)
            dataset = dataset.shuffle(buffer_size=100000).batch(batch_size=batch_size).prefetch(buffer_size=1)
        #    print(data.output_types)
        #    print(data.output_shapes)
            return dataset

        elif mode == "eval":
            dataset = tf.data.TFRecordDataset([file_path])
            dataset = dataset.map(_parse_record, num_parallel_calls=4)
            dataset = dataset.batch(batch_size=batch_size)
            return dataset

        elif mode == "pred":
            pred_features = feature_func(u=user, i=item)
            pred_data = tf.data.Dataset.from_tensor_slices(pred_features)
            return pred_data

        elif mode == "rank":
            rank_features = feature_func(u=user)
            rank_data = tf.data.Dataset.from_tensor_slices(rank_features).batch(batch_size)
            return rank_data

    def fit(self, wide_cols, deep_cols, train_data, eval_data, feature_func, eval_info, verbose=1):
        self.model = self.build_model(wide_cols, deep_cols)
        for epoch in range(1, self.n_epochs + 1):
            t0 = time.time()
            self.model.train(input_fn=lambda: WideDeepEstimator.input_fn(
                file_path=train_data, batch_size=self.batch_size, mode="train", feature_func=feature_func))

            if verbose == 1:
                print("Epoch {} training time: {:.4f}".format(epoch, time.time() - t0))
            elif verbose > 1:
                print("Epoch {} training time: {:.4f}".format(epoch, time.time() - t0))
                eval_result = self.model.evaluate(input_fn=lambda: WideDeepEstimator.input_fn(
                                                  file_path=eval_data, mode="eval", feature_func=feature_func))
                if self.task == "rating":
                    print("eval loss: {loss:.4f}, test rmse: {rmse:.4f}".format(**eval_result))
                elif self.task == "ranking":
                    for key in sorted(eval_result):
                        print("%s: %.4f" % (key, eval_result[key]))

            #    t0 = time.time()
            #    NDCG = NDCG_at_k(self, dataset=eval_info, k=10, sample_user=10, mode="estimator")
            #    print("\t NDCG@{}: {:.4f}".format(10, NDCG))
            #    print("\t NDCG time: {:.4f}".format(time.time() - t0))
            print()

    def predict_ui(self, u, i):  # cannot override Estimator's predict method
        pred_result = self.model.predict(input_fn=lambda: WideDeepEstimator.input_fn(   # use predict_fn ??
            mode="pred", user=u, item=i, feature_func=self.pred_feat_func))
        pred_result = list(pred_result)[0]
        if self.task == "rating":
            return list(pred_result)[0]['predictions']
        elif self.task == "ranking":
            return pred_result['probabilities'][0], pred_result['class_ids'][0]

    def recommend_user(self, u, n_rec=10):
        rank_list = self.model.predict(input_fn=lambda: WideDeepEstimator.input_fn(
            mode="rank", user=u, batch_size=256, feature_func=self.rank_feat_func))

        if self.task == "rating":
            rank = np.array([res['predictions'][0] for res in rank_list])
            indices = np.argpartition(rank, -n_rec)[-n_rec:]
            return sorted(zip(self.item_indices[indices], rank[indices]), key=lambda x: -x[1])

        elif self.task == "ranking":
            rank = np.array([res['probabilities'][0] for res in rank_list])
            indices = np.argpartition(rank, -n_rec)[-n_rec:]
            return sorted(zip(self.item_indices[indices], rank[indices]), key=itemgetter(1), reverse=True)


class WideDeep(BaseFeat):
    def __init__(self, lr, n_epochs=20, embed_size=100, reg=0.0, batch_size=256, seed=42,
                 hidden_units="128,64,32", use_bn=False, dropout_rate=0.0, task="rating", neg_sampling=False):
        self.lr = lr
        self.n_epochs = n_epochs
        self.embed_size = embed_size
        self.reg = reg
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.bn = use_bn
        self.hidden_units = list(map(int, hidden_units.split(",")))
        self.seed = seed
        self.task = task
        self.neg_sampling = neg_sampling
        super(WideDeep, self).__init__()

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

        self.feature_indices = tf.placeholder(tf.int32, shape=[None, self.field_size])
        self.feature_values = tf.placeholder(tf.float32, shape=[None, self.field_size])
        self.labels = tf.placeholder(tf.float32, shape=[None])
        self.is_training = tf.placeholder_with_default(False, shape=[])

        if self.reg > 0:
            self.linear_features = tf.get_variable(name="linear_features",
                                                   shape=[self.feature_size + 1, 1],
                                                   initializer=tf.initializers.truncated_normal(0.0, 0.01),
                                                   regularizer=tf.keras.regularizers.lr(self.reg))
            self.embedding_features = tf.get_variable(name="embedding_features",
                                                      shape=[self.feature_size + 1, self.embed_size],
                                                      initializer=tf.initializers.truncated_normal(0.0, 0.01),
                                                      regularizer=tf.keras.regularizers.lr(self.reg))
        else:
            self.linear_features = tf.get_variable(name="linear_features",
                                                   shape=[self.feature_size + 1, 1],
                                                   initializer=tf.initializers.truncated_normal(0.0, 0.01),
                                                   regularizer=None)
            self.embedding_features = tf.get_variable(name="embedding_features",
                                                      shape=[self.feature_size + 1, self.embed_size],
                                                      initializer=tf.initializers.truncated_normal(0.0, 0.01),
                                                      regularizer=None)
    #    self.w = tf.Variable(tf.truncated_normal([self.feature_size + 1, 1], 0.0, 0.01))  # feature_size + 1
    #    self.v = tf.Variable(tf.truncated_normal([self.feature_size + 1, self.embed_size], 0.0, 0.01))
        feature_values_reshape = tf.reshape(self.feature_values, shape=[-1, self.field_size, 1])

        linear_embedding = tf.nn.embedding_lookup(self.linear_features, self.feature_indices)   # N * F * 1
        linear_term = tf.reduce_sum(tf.multiply(linear_embedding, feature_values_reshape), 2)  # axis=1?

        MLP_embedding = tf.nn.embedding_lookup(self.embedding_features, self.feature_indices)  # N * F * K
        MLP_embedding = tf.multiply(MLP_embedding, feature_values_reshape)
        MLP_embedding = tf.reshape(MLP_embedding, [-1, self.field_size * self.embed_size])
        if self.bn:
            MLP_layer = tf.layers.batch_normalization(MLP_embedding,
                                                      training=self.is_training,
                                                      momentum=0.99)

        for units in self.hidden_units:
            if self.bn:
                MLP_layer = tf.layers.dense(inputs=MLP_layer,
                                            units=units,
                                            activation=None,
                                            use_bias=False,
                                            kernel_initializer=tf.variance_scaling_initializer)
                MLP_layer = tf.layers.batch_normalization(MLP_layer, training=self.is_training, momentum=0.99)
            else:
                MLP_layer = tf.layers.dense(inputs=MLP_layer,
                                            units=self.hidden_units[0],
                                            activation=None,
                                            use_bias=True,
                                            kernel_initializer=tf.variance_scaling_initializer)
            MLP_layer = tf.nn.relu(MLP_layer)
            if self.dropout_rate > 0.0:
                MLP_layer = tf.layers.dropout(MLP_layer, rate=self.dropout_rate, training=self.is_training)

        concat_layer = tf.concat([linear_term, MLP_layer], axis=1)

        if self.task == "rating":
            self.pred = tf.layers.dense(inputs=concat_layer, units=1)
            self.loss = tf.losses.mean_squared_error(labels=tf.reshape(self.labels, [-1, 1]),
                                                     predictions=self.pred)
            if self.lower_bound is not None and self.upper_bound is not None:
                self.rmse = tf.sqrt(tf.losses.mean_squared_error(abels=tf.reshape(self.labels, [-1, 1]),
                                    predictions=tf.clip_by_value(self.pred, self.lower_bound, self.upper_bound)))
            else:
                self.rmse = self.loss

        elif self.task == "ranking":
            self.logits = tf.layers.dense(inputs=concat_layer, units=1, name="logits")
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
                        self.sess.run(self.training_op, feed_dict={self.feature_indices: indices_batch,
                                                                   self.feature_values: values_batch,
                                                                   self.labels: labels_batch,
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
        try:
            target = self.pred if self.task == "rating" else self.y_prob
            pred = self.sess.run(target, feed_dict={self.feature_indices: feat_indices,
                                                    self.feature_values: feat_value,
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
        preds = self.sess.run(target, feed_dict={self.feature_indices: feat_indices,
                                                 self.feature_values: feat_values,
                                                 self.is_training: False})

        ids = np.argpartition(preds, -count)[-count:]
        rank = sorted(zip(ids, preds[ids]), key=lambda x: -x[1])
        return list(itertools.islice((rec for rec in rank if rec[0] not in consumed), n_rec))

    @property
    def item_info(self):
        item_col = self.dataset.train_feat_indices.shape[1] - 1
        item_cols = [item_col] + self.dataset.item_feature_cols
        return np.unique(self.dataset.train_feat_indices[:, item_cols], axis=0)








