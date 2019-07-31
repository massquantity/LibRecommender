import time, os
from operator import itemgetter
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column as feat_col
from tensorflow.python.estimator import estimator


class WideDeep:
    def __init__(self, embed_size, n_epochs=20, reg=0.0,
                 batch_size=64, dropout=0.0, seed=42, task="rating"):
        self.embed_size = embed_size
        self.n_epochs = n_epochs
        self.reg = reg
        self.batch_size = batch_size
        self.dropout = dropout
        self.seed = seed
        self.task = task

    def build_model(self, dataset):
        '''
        users = tf.feature_column.categorical_column_with_vocabulary_list("user", np.arange(dataset.n_users))
        items = tf.feature_column.categorical_column_with_vocabulary_list("item", np.arange(dataset.n_items))
        timestamps = tf.feature_column.numeric_column("timestamp")
        wide_cols = [tf.feature_column.crossed_column([users, items], hash_bucket_size=1000),
                     tf.feature_column.bucketized_column(timestamps,
                                                         boundaries=[9.653026e+08, 9.722437e+08, 9.752209e+08])]
        deep_cols = [tf.feature_column.embedding_column(users, dimension=self.embed_size),
                     tf.feature_column.embedding_column(items, dimension=self.embed_size),
                     tf.feature_column.bucketized_column(timestamps,
                                                         boundaries=[9.653026e+08, 9.722437e+08, 9.752209e+08])]
        '''

        users = tf.feature_column.categorical_column_with_vocabulary_list("user", np.arange(dataset.n_users))
        items = tf.feature_column.categorical_column_with_vocabulary_list("item", np.arange(dataset.n_items))
        timestamps_onehot = tf.feature_column.categorical_column_with_vocabulary_list(
                            "timestamp", np.arange(dataset.kb.n_bins_))
        wide_cols = [tf.feature_column.crossed_column([users, items], hash_bucket_size=1000),
                     timestamps_onehot]
        deep_cols = [tf.feature_column.embedding_column(users, dimension=self.embed_size),
                     tf.feature_column.embedding_column(items, dimension=self.embed_size),
                     tf.feature_column.indicator_column(timestamps_onehot)]

        config = tf.estimator.RunConfig(log_step_count_steps=1000, save_checkpoints_steps=10000)
        if self.task == "rating":
            self.model = tf.estimator.DNNLinearCombinedRegressor(
                model_dir="wide_deep_dir",
                config=config,
                linear_feature_columns=wide_cols,
                linear_optimizer=tf.train.FtrlOptimizer(learning_rate=0.1, l1_regularization_strength=0.001),
                dnn_feature_columns=deep_cols,
                dnn_hidden_units=[128, 64],
                dnn_optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
                dnn_dropout=0.0,
                batch_norm=False,
                loss_reduction=tf.losses.Reduction.MEAN)

        elif self.task == "ranking":
            self.model = tf.estimator.DNNLinearCombinedClassifier(
                model_dir="wide_deep_dir",
                config=config,
                linear_feature_columns=wide_cols,
                linear_optimizer=tf.train.FtrlOptimizer(learning_rate=0.1),  # l1_regularization_strength=0.001
                dnn_feature_columns=deep_cols,
                dnn_hidden_units=[128, 64],
                dnn_optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
                dnn_dropout=0.0,
                batch_norm=False,
                loss_reduction=tf.losses.Reduction.MEAN)

        else:
            raise ValueError("task must be rating or ranking")

    @staticmethod
    def input_fn(data, original_data=None, repeat=10, batch=256, mode="train", task="rating", user=1):
        if mode == "train":
            if task == "rating":
                features = {'user': data.train_user_indices,
                            'item': data.train_item_indices,
                            'timestamp': data.train_timestamp}
                labels = data.train_ratings
            elif task == "ranking":
                features = {'user': data.train_user_implicit,
                            'item': data.train_item_implicit,
                            'timestamp': data.train_timestamp}
                labels = data.train_label_implicit

            train_data = tf.data.Dataset.from_tensor_slices((features, labels))
            return train_data.shuffle(len(data.train_ratings)).repeat(repeat).batch(batch)

        elif mode == "evaluate":
            if task == "rating":
                features = {'user': data.test_user_indices.reshape(-1, 1),
                            'item': data.test_item_indices.reshape(-1, 1),
                            'timestamp': data.test_timestamp.reshape(-1, 1)}
                labels = data.test_ratings.reshape(-1, 1)
            elif task == "ranking":
                features = {'user': data.test_user_implicit.reshape(-1, 1),
                            'item': data.test_item_implicit.reshape(-1, 1),
                            'timestamp': data.test_timestamp.reshape(-1, 1)}
                labels = data.test_label_implicit.reshape(-1, 1)

            evaluate_data = tf.data.Dataset.from_tensor_slices((features, labels))
            return evaluate_data

        elif mode == "test":
            features = {'user': np.array(data[0]).reshape(-1,1),
                        'item': np.array(data[1]).reshape(-1,1)}
            try:
                ts = time.mktime(time.strptime(data[2][0], "%Y-%m-%d"))
                ts = np.array(ts).reshape(-1, 1)
                features['timestamp'] = original_data.kb.transform(ts).astype(int)
            except IndexError:
                print("use one day after max timestamp")
                max_timestamp = float(max(max(data.train_timestamp), max(data.test_timestamp))) + 3600 * 24
                max_timestamp = np.array(max_timestamp).reshape(-1, 1)
                features['timestamp'] = original_data.kb.transform(max_timestamp).astype(int)
            test_data = tf.data.Dataset.from_tensor_slices(features)
            return test_data

        elif mode == "rank":
            max_timestamp = float(max(max(data.train_timestamp), max(data.test_timestamp))) + 3600 * 24
            max_timestamp = np.array(max_timestamp).reshape(-1, 1)
            max_timestamp = original_data.kb.transform(max_timestamp).astype(int)
            features = {'user': np.full(data.n_items, user).reshape(-1, 1),
                        'item': np.arange(data.n_items).reshape(-1, 1),
                        'timestamp': np.full(data.n_items, max_timestamp).reshape(-1, 1)}
            test_data = tf.data.Dataset.from_tensor_slices(features)
            return test_data

    def fit(self, dataset):
        self.dataset = dataset
        self.build_model(dataset)
        for epoch in range(1, 3):  # epoch_per_eval
            t0 = time.time()
            self.model.train(input_fn=lambda: WideDeep.input_fn(
                data=dataset, repeat=self.n_epochs, batch=self.batch_size, mode="train", task=self.task))
            train_loss = self.model.evaluate(input_fn=lambda: WideDeep.input_fn(
                data=dataset, repeat=self.n_epochs, batch=self.batch_size, mode="train", task=self.task))
            evaluate_loss = self.model.evaluate(input_fn=lambda: WideDeep.input_fn(
                data=dataset, repeat=self.n_epochs, batch=self.batch_size, mode="evaluate", task=self.task))
            print("Epoch {} training time: {:.4f}".format(epoch, time.time() - t0))
            print("train loss: {loss:.4f}, step: {global_step}".format(**train_loss))
            print("evaluate loss: {loss:.4f}".format(**evaluate_loss))

    def predict(self, u, i, *args):
        pred_result = self.model.predict(input_fn=lambda: WideDeep.input_fn(
            data=[u, i, args], original_data=self.dataset, mode="test"))
        if self.task == "rating":
            return list(pred_result)[0]['predictions']
        elif self.task == "ranking":
            return list(pred_result)[0]['class_ids']

    def predict_user(self, u):
        rank_list = self.model.predict(input_fn=lambda: WideDeep.input_fn(
            data=self.dataset, original_data=self.dataset, mode="rank", user=u))
        if self.task == "rating":
            return sorted([(item, rating['predictions'][0]) for item, rating in enumerate(list(rank_list))],
                          key=itemgetter(1), reverse=True)[:10]
        elif self.task == "ranking":
            return sorted([(item, rating['probabilities'][0]) for item, rating in enumerate(list(rank_list))],
                          key=itemgetter(1), reverse=True)[:10]



# TODO batch_norm, dropout
class WideDeepCustom(estimator.Estimator):  # tf.estimator.Estimator,  NOOOO inheritance
    def __init__(self, embed_size, n_epochs=20, reg=0.0, cross_features=False,
                 batch_size=64, dropout=0.0, seed=42, task="rating"):
        self.embed_size = embed_size
        self.n_epochs = n_epochs
        self.reg = reg
        self.batch_size = batch_size
        self.dropout = dropout
        self.seed = seed
        self.task = task
        self.cross_features = cross_features
        super(WideDeepCustom, self).__init__(model_fn=WideDeepCustom.model_func)

    def build_model(self, dataset):
        wide_cols = []
        deep_cols = []
        if self.cross_features:
            for col1 in dataset.feature_cols:
                for col2 in dataset.feature_cols:
                    if col1 != col2 and col1 != "user" and col1 != "item" and col2 != "user" and col2 != "item":
                        col1_feat = feat_col.categorical_column_with_vocabulary_list(
                            col1, dataset.col_unique_values[col1])
                        col2_feat = feat_col.categorical_column_with_vocabulary_list(
                            col2, dataset.col_unique_values[col2])
                        wide_cols.append(feat_col.crossed_column([col1_feat, col2_feat], hash_bucket_size=1000))

            for col in dataset.feature_cols:
                col_feat = feat_col.categorical_column_with_vocabulary_list(col, dataset.col_unique_values[col])
                deep_cols.append(feat_col.embedding_column(col_feat, dimension=self.embed_size))

        else:
            for col in dataset.feature_cols:
                col_feat = feat_col.categorical_column_with_vocabulary_list(col, dataset.col_unique_values[col])
                wide_cols.append(col_feat)
                deep_cols.append(feat_col.embedding_column(col_feat, dimension=self.embed_size))

        config = tf.estimator.RunConfig(log_step_count_steps=1000, save_checkpoints_steps=10000)
        model = tf.estimator.Estimator(model_fn=WideDeepCustom.model_func,
                                       model_dir="wide_deep_dir",
                                       config=config,
                                       params={'deep_columns': deep_cols,
                                               'wide_columns': wide_cols,
                                               'hidden_units': [128, 64],
                                               'task': self.task})

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
            rmse = tf.metrics.root_mean_squared_error(labels=labels, predictions=logits)
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

        '''
        train_op = []
        optimizer2 = tf.train.AdamOptimizer(learning_rate=0.01)
        training_op2 = optimizer2.minimize(loss, global_step=tf.train.get_global_step())
        train_op.append(training_op2)
        optimizer = tf.train.FtrlOptimizer(learning_rate=0.1, l1_regularization_strength=1e-3) ### Adagrad
        training_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        train_op.append(training_op)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=tf.group(*train_op))
        '''
    #    optimizer = tf.train.FtrlOptimizer(learning_rate=0.1, l1_regularization_strength=1e-3)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    #    optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
    #    optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=0.01)
        training_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=training_op)

    @staticmethod
    def input_fn(data, original_data=None, repeat=10, batch=256, mode="train", task="rating", user=None):
        if mode == "train":
            features = {col: data.train_data[col].values for col in data.feature_cols}
            labels = data.train_data[data.label_cols].values

            train_data = tf.data.Dataset.from_tensor_slices((features, labels))
            return train_data.shuffle(len(labels)).repeat(repeat).batch(batch)

        elif mode == "evaluate":
            features = {col: data.test_data[col].values for col in data.feature_cols}
            labels = data.test_data[data.label_cols].values

            evaluate_data = tf.data.Dataset.from_tensor_slices((features, labels))
            return evaluate_data

        elif mode == "test":
            features = {col: data.test_data[col].values for col in data.feature_cols}
            test_data = tf.data.Dataset.from_tensor_slices(features)
            return test_data

        elif mode == "rank":
            n_items = len(data.item_dict)
            user_part = pd.DataFrame([data.user_dict[user]], columns=data.user_feature_cols)
            user_part = user_part.reindex(user_part.index.repeat(n_items))
            item_part = pd.DataFrame(list(data.item_dict.values()), columns=data.item_feature_cols)
            features = {col: user_part[col].values for col in user_part.columns}
            features.update({col: item_part[col].values for col in item_part.columns})

            rank_data = tf.data.Dataset.from_tensor_slices(features)
            return rank_data

    def fit(self, dataset, verbose=1):
        self.dataset = dataset
        self.model = self.build_model(dataset)
        for epoch in range(1, 3):  # epoch_per_eval
            t0 = time.time()
            self.model.train(input_fn=lambda: WideDeepCustom.input_fn(
                data=dataset, repeat=self.n_epochs, batch=self.batch_size, mode="train", task=self.task))
            train_result = self.model.evaluate(input_fn=lambda: WideDeepCustom.input_fn(
                data=dataset, repeat=self.n_epochs, batch=self.batch_size, mode="train", task=self.task))
            eval_result = self.model.evaluate(input_fn=lambda: WideDeepCustom.input_fn(
                data=dataset, mode="evaluate", task=self.task))

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

    def predict_ui(self, u, i, *args):  # cannot override Estimator's predict method
        pred_result = self.model.predict(input_fn=lambda: WideDeep.input_fn(
            data=[u, i, args], original_data=self.dataset, mode="test"))
        if self.task == "rating":
            return list(pred_result)[0]['predictions']
        elif self.task == "ranking":
            return list(pred_result)[0]['class_ids']

    def predict_user(self, u, k):
        rank_list = self.model.predict(input_fn=lambda: WideDeep.input_fn(
            data=self.dataset, original_data=self.dataset, mode="rank", user=u))
        if self.task == "rating":
            return sorted([(item, rating['predictions'][0]) for item, rating in enumerate(list(rank_list))],
                          key=itemgetter(1), reverse=True)[:k]
        elif self.task == "ranking":
            return sorted([(item, rating['probabilities'][0]) for item, rating in enumerate(list(rank_list))],
                          key=itemgetter(1), reverse=True)[:k]



