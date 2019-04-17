import time
import numpy as np
import tensorflow as tf
from tensorflow import feature_column as feat_col
from tensorflow.python.estimator import estimator
from ..evaluate.evaluate import NDCG_at_k, NDCG_at_k_tf, precision_tf


class WideDeep:
    def __init__(self, embed_size, n_epochs=20, reg=0.0,
                 batch_size=64, dropout=0.0, seed=42):
        self.embed_size = embed_size
        self.n_epochs = n_epochs
        self.reg = reg
        self.batch_size = batch_size
        self.dropout = dropout
        self.seed = seed

    def build_model(self, dataset):
        users = tf.feature_column.categorical_column_with_vocabulary_list("user", np.arange(dataset.n_users))
        items = tf.feature_column.categorical_column_with_vocabulary_list("item", np.arange(dataset.n_items))
        timestamps = tf.feature_column.numeric_column("timestamp")
        wide_cols = [tf.feature_column.crossed_column([users, items], hash_bucket_size=1000),
                     tf.feature_column.bucketized_column(timestamps,
                                                         boundaries=[9.653026e+08, 9.722437e+08, 9.752209e+08])]
        deep_cols = [tf.feature_column.embedding_column(users, dimension=self.embed_size),
                     tf.feature_column.embedding_column(items, dimension=self.embed_size)]

        config = tf.estimator.RunConfig(log_step_count_steps=1000, save_checkpoints_steps=10000)
        self.model = tf.estimator.DNNLinearCombinedRegressor(
            model_dir="wide_deep_dir",
            config=config,
            linear_feature_columns=wide_cols,
            linear_optimizer=tf.train.FtrlOptimizer(learning_rate=0.1),  # l1_regularization_strength=0.001
            dnn_feature_columns=deep_cols,
            dnn_hidden_units=[128, 64],
            dnn_optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
            dnn_dropout=0.0,
            batch_norm=True,
            loss_reduction=tf.losses.Reduction.MEAN)

    def fit(self, dataset):
        def input_fn(data=dataset, mode="train"):
            if mode == "train":
                features = {'user': data.train_user_indices,
                            'item': data.train_item_indices,
                            'timestamp': data.train_timestamp}
                labels = data.train_ratings
                train_data = tf.data.Dataset.from_tensor_slices((features, labels))
                return train_data.shuffle(len(data.train_ratings)).repeat(self.n_epochs).batch(self.batch_size)
            #    return train_data.shuffle(len(data.train_ratings)).batch(self.batch_size).repeat(self.n_epochs)
            elif mode == "evaluate":
                features = {'user': data.test_user_indices,
                            'item': data.test_item_indices,
                            'timestamp': data.test_timestamp}
                labels = data.test_ratings
                train_data = tf.data.Dataset.from_tensor_slices((features, labels))
                return train_data.shuffle(len(data.test_ratings)).batch(self.batch_size)

        self.build_model(dataset)
        for epoch in range(1, 101):  # epoch_per_eval
            t0 = time.time()
            self.model.train(lambda: input_fn(mode="train"))
            train_loss = self.model.evaluate(lambda: input_fn(mode="train"))
            test_loss = self.model.evaluate(lambda: input_fn(mode="evaluate"))
            print("Epoch {} training time: {:.4f}".format(epoch, time.time() - t0))
            print("train loss: {loss:.4f}, step: {global_step}".format(**train_loss))
            print("test loss: {loss:.4f}".format(**test_loss))





class WideDeep_898:
    def __init__(self, embed_size, n_epochs=20, reg=0.0,
                 batch_size=64, dropout=0.0, seed=42):
        self.embed_size = embed_size
        self.n_epochs = n_epochs
        self.reg = reg
        self.batch_size = batch_size
        self.dropout = dropout
        self.seed = seed

    def build_model(self, dataset):
        users = tf.feature_column.categorical_column_with_vocabulary_list("user", np.arange(dataset.n_users))
        items = tf.feature_column.categorical_column_with_vocabulary_list("item", np.arange(dataset.n_items))
        wide_cols = tf.feature_column.crossed_column([users, items], hash_bucket_size=1000)
        deep_cols = [tf.feature_column.embedding_column(users, dimension=self.embed_size),
                     tf.feature_column.embedding_column(items, dimension=self.embed_size)]

        config = tf.estimator.RunConfig(log_step_count_steps=1000, save_checkpoints_steps=10000)
        self.model = tf.estimator.DNNLinearCombinedRegressor(
            model_dir="wide_deep_dir",
            config=config,
            linear_feature_columns=wide_cols,
            linear_optimizer=tf.train.FtrlOptimizer(learning_rate=0.1),  # l1_regularization_strength=0.001
            dnn_feature_columns=deep_cols,
            dnn_hidden_units=[128, 64],
            dnn_optimizer=tf.train.FtrlOptimizer(learning_rate=0.1),
            dnn_dropout=0.0,
            batch_norm=True,
            loss_reduction=tf.losses.Reduction.MEAN)

    #    self.model = tf.estimator.DNNRegressor(
    #        model_dir="wide_deep_dir",
    #        config=config,
    #        feature_columns=deep_cols,
    #        hidden_units=[128, 64],
    #        batch_norm=False,
    #        loss_reduction=tf.losses.Reduction.MEAN)

    def fit(self, dataset):
        def input_fn(data=dataset, mode="train"):
            if mode == "train":
                features = {'user': data.train_user_indices,
                            'item': data.train_item_indices}
                labels = data.train_ratings
                train_data = tf.data.Dataset.from_tensor_slices((features, labels))
                return train_data.shuffle(len(data.train_ratings)).repeat(self.n_epochs).batch(self.batch_size)
            #    return train_data.shuffle(len(data.train_ratings)).batch(self.batch_size).repeat(self.n_epochs)
            elif mode == "evaluate":
                features = {'user': data.test_user_indices,
                            'item': data.test_item_indices}
                labels = data.test_ratings
                train_data = tf.data.Dataset.from_tensor_slices((features, labels))
                return train_data.shuffle(len(data.test_ratings)).batch(self.batch_size)

        self.build_model(dataset)
        for epoch in range(1, 101):  # epoch_per_eval
            t0 = time.time()
            self.model.train(lambda: input_fn(mode="train"))
            train_loss = self.model.evaluate(lambda: input_fn(mode="train"))
            test_loss = self.model.evaluate(lambda: input_fn(mode="evaluate"))
            print("Epoch {} training time: {:.4f}".format(epoch, time.time() - t0))
            print("train loss: {loss:.4f}, step: {global_step}".format(**train_loss))
            print("test loss: {loss:.4f}".format(**test_loss))


# TODO batch_norm, dropout
class WideDeep_999345(estimator.Estimator):
    def __init__(self, embed_size, n_epochs=20, reg=0.0,
                 batch_size=64, dropout=0.0, seed=42):
        self.embed_size = embed_size
        self.n_epochs = n_epochs
        self.reg = reg
        self.batch_size = batch_size
        self.dropout = dropout
        self.seed = seed
        super(WideDeep, self).__init__(model_fn=WideDeep.my_model,
                                       model_dir="wide_deep_dir")

    def build_model(self, dataset):
        users = feat_col.categorical_column_with_vocabulary_list("user", np.arange(dataset.n_users))
        items = feat_col.categorical_column_with_vocabulary_list("item", np.arange(dataset.n_items))
        wide_cols = feat_col.crossed_column([users, items], hash_bucket_size=1000)
    #    self.wide_cols = feat_col.indicator_column(self.wide_cols)
        deep_cols = [feat_col.embedding_column(users, dimension=self.embed_size),
                     feat_col.embedding_column(items, dimension=self.embed_size)]

    #    feature_cols = deep_cols.append(feat_col.indicator_column(wide_cols))
        model = tf.estimator.Estimator(model_fn=WideDeep.my_model,
                                       model_dir="wide_deep_dir",
                                       params={'deep_columns': deep_cols,
                                               'wide_columns': wide_cols,
                                               'hidden_units': [128, 64]})

        return model

    @staticmethod
    def my_model(features, labels, mode, params):
        dnn_input = feat_col.input_layer(features, params['deep_columns'])
        for units in params['hidden_units']:
            dnn_input = tf.layers.dense(dnn_input, units=units, activation=tf.nn.relu)
        dnn_logits = tf.layers.dense(dnn_input, units=1, activation=None)

    #    linear_input = feat_col.input_layer(features, params['wide_columns'])
    #    features = tf.parse_example(
        #    serialized=tf.placeholder(tf.string, name='tf_example'),
    #        serialized=["wide_columns"],
    #        features=feat_col.make_parse_example_spec([self.wide_cols]))
        linear_logits = feat_col.linear_model(units=1, features=features,
                                              feature_columns=params['wide_columns'])

        logits = tf.add_n([dnn_logits, linear_logits])
    #    print("logits shape: ", tf.shape(dnn_logits).get_shape())

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {'prediction': logits}
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        labels = tf.cast(tf.reshape(labels, [-1,1]), tf.float32)
        loss = tf.losses.mean_squared_error(labels=labels, predictions=logits)
        rmse = tf.metrics.root_mean_squared_error(labels=labels, predictions=logits)
    #    NDCG = NDCG_at_k_tf(labels=labels, predictions=logits, k=1)
    #    metrics = {'rmse': rmse, 'NDCG': NDCG}
        metrics = {'rmse': rmse}

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

        assert mode == tf.estimator.ModeKeys.TRAIN

        '''
        train_op = []
        optimizer2 = tf.train.AdamOptimizer(learning_rate=0.02)
        training_op2 = optimizer2.minimize(loss, global_step=tf.train.get_global_step())
        train_op.append(training_op2)
        optimizer = tf.train.FtrlOptimizer(learning_rate=0.1, l1_regularization_strength=1e-3) ### Adagrad
        training_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        train_op.append(training_op)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=tf.group(*train_op))
        '''
        optimizer = tf.train.FtrlOptimizer(learning_rate=0.1, l1_regularization_strength=1e-3)
    #    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    #    optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
    #    optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=0.01)
        training_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=training_op)

    def fit(self, dataset):
        def input_fn(data=dataset, mode="train"):
            if mode == "train":
                features = {'user': data.train_user_indices,
                            'item': data.train_item_indices}
                labels = data.train_ratings
                train_data = tf.data.Dataset.from_tensor_slices((features, labels))
                return train_data.shuffle(len(data.train_ratings)).repeat(self.n_epochs).batch(self.batch_size)
            #    return train_data.shuffle(len(data.train_ratings)).batch(self.batch_size).repeat(self.n_epochs)
            #    train_data = train_data.shuffle(len(data.train_ratings)).repeat(self.n_epochs).batch(self.batch_size)
            #    return train_data.make_one_shot_iterator().get_next()
            elif mode == "evaluate":
                features = {'user': data.test_user_indices,
                            'item': data.test_item_indices}
                labels = data.test_ratings
                train_data = tf.data.Dataset.from_tensor_slices((features, labels))
                return train_data.shuffle(len(data.test_ratings)).batch(self.batch_size)

        model = self.build_model(dataset)
        for epoch in range(1, 10001):  # epoch_per_eval
            t0 = time.time()
            model.train(input_fn=lambda: input_fn(mode="train"))
            train_result = model.evaluate(input_fn=lambda: input_fn(mode="train"))
            eval_result = model.evaluate(input_fn=lambda: input_fn(mode="evaluate"))
        #    pred_result = model.predict(input_fn=lambda: input_fn(mode="evaluate"), predict_keys="prediction")
            print("Epoch {} training time: {:.4f}".format(epoch, time.time() - t0))
            print("train loss: {loss:.4f}, train rmse: {rmse:.4f}".format(**train_result))
            print("test loss: {loss:.4f}, test rmse: {rmse:.4f}".format(**eval_result))
        #    print("pred", [p.get("prediction")[0] for p in list(pred_result)[:10]])




class WideDeep_989(estimator.Estimator):
    def __init__(self, embed_size, n_epochs=20, reg=0.0,
                 batch_size=64, dropout=0.0, seed=42):
        self.embed_size = embed_size
        self.n_epochs = n_epochs
        self.reg = reg
        self.batch_size = batch_size
        self.dropout = dropout
        self.seed = seed
        super(WideDeep, self).__init__(model_fn=WideDeep.my_model,
                                       model_dir="wide_deep_dir")

    def build_model(self, dataset):
        users = feat_col.categorical_column_with_vocabulary_list("user", np.arange(dataset.n_users))
        items = feat_col.categorical_column_with_vocabulary_list("item", np.arange(dataset.n_items))
        wide_cols = feat_col.crossed_column([users, items], hash_bucket_size=1000)
    #    self.wide_cols = feat_col.indicator_column(self.wide_cols)
        deep_cols = [feat_col.embedding_column(users, dimension=self.embed_size),
                     feat_col.embedding_column(items, dimension=self.embed_size)]

    #    feature_cols = deep_cols.append(feat_col.indicator_column(wide_cols))
        model = tf.estimator.Estimator(model_fn=WideDeep.my_model,
                                       model_dir="wide_deep_dir",
                                       params={'deep_columns': deep_cols,
                                               'wide_columns': wide_cols,
                                               'hidden_units': [128, 64]})

        return model

    @staticmethod
    def my_model(features, labels, mode, params):
        dnn_input = feat_col.input_layer(features, params['deep_columns'])
        for units in params['hidden_units']:
            dnn_input = tf.layers.dense(dnn_input, units=units, activation=tf.nn.relu)
        dnn_logits = tf.layers.dense(dnn_input, units=1, activation=None)

        linear_logits = feat_col.linear_model(units=1, features=features,
                                              feature_columns=params['wide_columns'])

        logits = tf.add_n([dnn_logits, linear_logits])
        y_prob = tf.sigmoid(logits)
        pred = tf.where(y_prob >= 0.5,
                        tf.fill(tf.shape(logits), 1.0),
                        tf.fill(tf.shape(logits), 0.0))

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'class_ids': pred,
                'probabilities': y_prob,
                'logits': logits,
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    #    logits = tf.reshape(logits, [-1])
        labels = tf.reshape(labels, [-1, 1])
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
    #    labels = tf.cast(labels, tf.int64)
        accuracy = tf.metrics.accuracy(labels=labels, predictions=pred)
    #    precision_at_k = tf.metrics.precision_at_k(labels=labels, predictions=logits, k=10)
    #    precision_at_k_2 = tf.metrics.precision_at_top_k(labels=labels, predictions_idx=pred, k=10)
        precision = tf.metrics.precision(labels=labels, predictions=pred)
        metrics = {'accuracy': accuracy,
                   'precision': precision}

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

        assert mode == tf.estimator.ModeKeys.TRAIN

        '''
        train_op = []
        optimizer2 = tf.train.FtrlOptimizer(learning_rate=0.1, l1_regularization_strength=1e-3)
        training_op2 = optimizer2.minimize(loss, global_step=tf.train.get_global_step())
        train_op.append(training_op2)
        optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9)
        training_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        train_op.append(training_op)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=tf.group(*train_op))
        '''
    #    optimizer = tf.train.FtrlOptimizer(learning_rate=0.1)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    #    optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9)
    #    optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=0.01)  #  l1_regularization_strength=1e-3
        training_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=training_op)


    def fit(self, dataset):
        def input_fn(data=dataset, mode="train"):
            if mode == "train":
                features = {'user': data.train_user_implicit,
                            'item': data.train_item_implicit}
                labels = data.train_label_implicit
                train_data = tf.data.Dataset.from_tensor_slices((features, labels))
                return train_data.shuffle(len(data.train_ratings)).repeat(self.n_epochs).batch(self.batch_size)
            elif mode == "evaluate":
                features = {'user': data.test_user_implicit,
                            'item': data.test_item_implicit}
                labels = data.test_label_implicit
                test_data = tf.data.Dataset.from_tensor_slices((features, labels))
                return test_data.shuffle(len(data.test_ratings)).batch(self.batch_size)

        model = self.build_model(dataset)
        for epoch in range(1, 10001):  # epoch_per_eval
            t0 = time.time()
            model.train(input_fn=lambda: input_fn(mode="train"))
            train_result = model.evaluate(input_fn=lambda: input_fn(mode="train"))
            eval_result = model.evaluate(input_fn=lambda: input_fn(mode="evaluate"))
            pred_result = model.predict(input_fn=lambda: input_fn(mode="evaluate"))
            print("Epoch {} training time: {:.4f}".format(epoch, time.time() - t0))
            print("train loss: {loss:.4f}, accuracy: {accuracy:.4f}, " 
                  "precision: {precision:.4f}".format(**train_result))
            print("test loss: {loss:.4f}, accuracy: {accuracy:.4f}, "
                  "precision: {precision:.4f}".format(**eval_result))
            for i, pred in enumerate(list(pred_result), start=1):
                print("pred {}: class_id, {}, probability, {}".format(i, pred['class_ids'], pred['probabilities']))
                i += 1
                if i > 5:
                    break



class WideDeep_76876:
    def __init__(self, embed_size, n_epochs=20, reg=0.0,
                 batch_size=64, dropout=0.0, seed=42):
        self.embed_size = embed_size
        self.n_epochs = n_epochs
        self.reg = reg
        self.batch_size = batch_size
        self.dropout = dropout
        self.seed = seed

    def build_model(self, dataset):
        users = tf.feature_column.categorical_column_with_vocabulary_list("user", np.arange(dataset.n_users))
        items = tf.feature_column.categorical_column_with_vocabulary_list("item", np.arange(dataset.n_items))
        wide_cols = tf.feature_column.crossed_column([users, items], hash_bucket_size=1000)
        deep_cols = [tf.feature_column.embedding_column(users, dimension=self.embed_size),
                     tf.feature_column.embedding_column(items, dimension=self.embed_size)]

        config = tf.estimator.RunConfig(log_step_count_steps=1000, save_checkpoints_steps=10000)
        self.model = tf.estimator.DNNLinearCombinedClassifier(
            model_dir="wide_deep_dir",
            config=config,
            linear_feature_columns=wide_cols,
            linear_optimizer=tf.train.FtrlOptimizer(learning_rate=1.0),  # l1_regularization_strength=0.001
            dnn_feature_columns=deep_cols,
            dnn_hidden_units=[128, 64],
            dnn_optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
            dnn_dropout=0.0,
            batch_norm=False,
            loss_reduction=tf.losses.Reduction.MEAN)


    def fit(self, dataset):
        def input_fn(data=dataset, mode="train"):
            if mode == "train":
                features = {'user': data.train_user_implicit,
                            'item': data.train_item_implicit}
                labels = data.train_label_implicit
                train_data = tf.data.Dataset.from_tensor_slices((features, labels))
                return train_data.shuffle(len(data.train_ratings)).repeat(self.n_epochs).batch(self.batch_size)
            #    return train_data.shuffle(len(data.train_ratings)).batch(self.batch_size).repeat(self.n_epochs)
            elif mode == "evaluate":
                features = {'user': data.test_user_implicit,
                            'item': data.test_item_implicit}
                labels = data.test_label_implicit
                train_data = tf.data.Dataset.from_tensor_slices((features, labels))
                return train_data.shuffle(len(data.test_ratings)).batch(self.batch_size)

        self.build_model(dataset)
        for epoch in range(1, 101):  # epoch_per_eval
            t0 = time.time()
            self.model.train(lambda: input_fn(mode="train"))
            train_loss = self.model.evaluate(lambda: input_fn(mode="train"))
            test_loss = self.model.evaluate(lambda: input_fn(mode="evaluate"))
            print("Epoch {} training time: {:.4f}".format(epoch, time.time() - t0))
            print("train loss: {loss:.4f}, step: {global_step}".format(**train_loss))
            print("test loss: {loss:.4f}".format(**test_loss))
            pred_result = self.model.predict(input_fn=lambda: input_fn(mode="evaluate"))
            print("pred", list(pred_result)[0])



class WideDeep_99999(estimator.Estimator):
    def __init__(self, embed_size, n_epochs=20, reg=0.0,
                 batch_size=64, dropout=0.0, seed=42):
        self.embed_size = embed_size
        self.n_epochs = n_epochs
        self.reg = reg
        self.batch_size = batch_size
        self.dropout = dropout
        self.seed = seed
        super(WideDeep, self).__init__(model_fn=WideDeep.my_model,
                                       model_dir="wide_deep_dir")

    def build_model(self, dataset):
        users = feat_col.categorical_column_with_vocabulary_list("user", np.arange(dataset.n_users))
        items = feat_col.categorical_column_with_vocabulary_list("item", np.arange(dataset.n_items))
        wide_cols = feat_col.crossed_column([users, items], hash_bucket_size=1000)
    #    self.wide_cols = feat_col.indicator_column(self.wide_cols)
        deep_cols = [feat_col.embedding_column(users, dimension=self.embed_size),
                     feat_col.embedding_column(items, dimension=self.embed_size)]

    #    feature_cols = deep_cols.append(feat_col.indicator_column(wide_cols))
        model = tf.estimator.Estimator(model_fn=WideDeep.my_model,
                                       model_dir="wide_deep_dir",
                                       params={'deep_columns': deep_cols,
                                               'wide_columns': wide_cols,
                                               'hidden_units': [128, 64]})

        return model

    @staticmethod
    def my_model(features, labels, mode, params):
        dnn_input = feat_col.input_layer(features, params['deep_columns'])
        for units in params['hidden_units']:
            dnn_input = tf.layers.dense(dnn_input, units=units, activation=tf.nn.relu)
        dnn_logits = tf.layers.dense(dnn_input, units=10, activation=None)

        linear_logits = feat_col.linear_model(units=10, features=features,
                                              feature_columns=params['wide_columns'])

    #    logits = tf.add_n([dnn_logits, linear_logits])
        concat = tf.concat([dnn_logits, linear_logits], axis=-1)
        logits = tf.layers.dense(concat, units=1)
        y_prob = tf.sigmoid(logits)
        pred = tf.where(y_prob >= 0.5,
                        tf.fill(tf.shape(logits), 1.0),
                        tf.fill(tf.shape(logits), 0.0))

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'class_ids': pred,
                'probabilities': y_prob,
                'logits': logits,
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    #    logits = tf.reshape(logits, [-1])
        labels = tf.reshape(labels, [-1, 1])
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
    #    labels = tf.cast(labels, tf.int64)
        accuracy = tf.metrics.accuracy(labels=labels, predictions=pred)
    #    precision_at_k = tf.metrics.precision_at_k(labels=labels, predictions=logits, k=10)
    #    precision_at_k_2 = tf.metrics.precision_at_top_k(labels=labels, predictions_idx=pred, k=10)
        precision = tf.metrics.precision(labels=labels, predictions=pred)
        metrics = {'accuracy': accuracy,
                   'precision': precision}

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

        assert mode == tf.estimator.ModeKeys.TRAIN

        '''
        train_op = []
        optimizer2 = tf.train.FtrlOptimizer(learning_rate=0.1, l1_regularization_strength=1e-3)
        training_op2 = optimizer2.minimize(loss, global_step=tf.train.get_global_step())
        train_op.append(training_op2)
        optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9)
        training_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        train_op.append(training_op)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=tf.group(*train_op))
        '''
    #    optimizer = tf.train.FtrlOptimizer(learning_rate=0.1)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    #    optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9)
    #    optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=0.01)  #  l1_regularization_strength=1e-3
        training_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=training_op)


    def fit(self, dataset):
        def input_fn(data=dataset, mode="train"):
            if mode == "train":
                features = {'user': data.train_user_implicit,
                            'item': data.train_item_implicit}
                labels = data.train_label_implicit
                train_data = tf.data.Dataset.from_tensor_slices((features, labels))
                return train_data.shuffle(len(data.train_ratings)).repeat(self.n_epochs).batch(self.batch_size)
            elif mode == "evaluate":
                features = {'user': data.test_user_implicit,
                            'item': data.test_item_implicit}
                labels = data.test_label_implicit
                test_data = tf.data.Dataset.from_tensor_slices((features, labels))
                return test_data.shuffle(len(data.test_ratings)).batch(self.batch_size)

        model = self.build_model(dataset)
        for epoch in range(1, 10001):  # epoch_per_eval
            t0 = time.time()
            model.train(input_fn=lambda: input_fn(mode="train"))
            train_result = model.evaluate(input_fn=lambda: input_fn(mode="train"))
            eval_result = model.evaluate(input_fn=lambda: input_fn(mode="evaluate"))
            pred_result = model.predict(input_fn=lambda: input_fn(mode="evaluate"))
            print("Epoch {} training time: {:.4f}".format(epoch, time.time() - t0))
            print("train loss: {loss:.4f}, accuracy: {accuracy:.4f}, " 
                  "precision: {precision:.4f}".format(**train_result))
            print("test loss: {loss:.4f}, accuracy: {accuracy:.4f}, "
                  "precision: {precision:.4f}".format(**eval_result))
            for i, pred in enumerate(list(pred_result), start=1):
                print("pred {}: class_id, {}, probability, {}".format(i, pred['class_ids'], pred['probabilities']))
                i += 1
                if i > 5:
                    break