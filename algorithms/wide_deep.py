import time
import numpy as np
import tensorflow as tf
from tensorflow import feature_column as feat_col


class WideDeep_4565:
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
            linear_optimizer=tf.train.FtrlOptimizer(learning_rate=0.1, l1_regularization_strength=1e-3),
            dnn_feature_columns=deep_cols,
            dnn_hidden_units=[128, 64],
            dnn_optimizer=tf.train.FtrlOptimizer(learning_rate=0.1, l1_regularization_strength=1e-3),
            dnn_dropout=0.0,
            batch_norm=False,
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
            print("train loss: ", train_loss)
            print("test loss: ", test_loss)




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
        users = feat_col.categorical_column_with_vocabulary_list("user", np.arange(dataset.n_users))
        items = feat_col.categorical_column_with_vocabulary_list("item", np.arange(dataset.n_items))
        wide_cols = feat_col.crossed_column([users, items], hash_bucket_size=1000)
        deep_cols = [feat_col.embedding_column(users, dimension=self.embed_size),
                     feat_col.embedding_column(items, dimension=self.embed_size)]

        feature_cols = deep_cols  # .append(feat_col.indicator_column(wide_cols))
        model = tf.estimator.Estimator(model_dir="wide_deep_dir",
                                       model_fn=WideDeep.my_model,
                                       params={'feature_columns': feature_cols,
                                               'hidden_units': [128, 64]})
        return model

    @staticmethod
    def my_model(features, labels, mode, params):
        model_input = feat_col.input_layer(features, params['feature_columns'])
        for units in params['hidden_units']:
            model_input = tf.layers.dense(model_input, units=units, activation=tf.nn.relu)

        logits = tf.layers.dense(model_input, units=1, activation=None)
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {'prediction': logits}
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        labels = tf.cast(tf.reshape(labels, [-1,1]), tf.float32)
        loss = tf.losses.mean_squared_error(labels=labels, predictions=logits)
        rmse = tf.metrics.root_mean_squared_error(labels=labels, predictions=logits)
        metrics = {'rmse': rmse}

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

        assert mode == tf.estimator.ModeKeys.TRAIN
        optimizer = tf.train.FtrlOptimizer(learning_rate=0.1, l1_regularization_strength=1e-3) ### Adagrad
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
            print("Epoch {} training time: {:.4f}".format(epoch, time.time() - t0))
            print("train loss: {loss:.4f}, train rmse: {rmse:.4f}".format(**train_result))
            print("test loss: {loss:.4f}, test rmse: {rmse:.4f}".format(**eval_result))



class WideDeep(tf.estimator.DNNLinearCombinedRegressor):
    def __init__(self, embed_size, n_epochs=20, reg=0.0,
                 batch_size=64, dropout=0.0, seed=42):
        super(WideDeep, self).__init__()
        self.embed_size = embed_size
        self.n_epochs = n_epochs
        self.reg = reg
        self.batch_size = batch_size
        self.dropout = dropout
        self.seed = seed

    def build_model(self, dataset):
        users = feat_col.categorical_column_with_vocabulary_list("user", np.arange(dataset.n_users))
        items = feat_col.categorical_column_with_vocabulary_list("item", np.arange(dataset.n_items))
        self.wide_cols = feat_col.crossed_column([users, items], hash_bucket_size=1000)
    #    self.wide_cols = feat_col.indicator_column(self.wide_cols)
        deep_cols = [feat_col.embedding_column(users, dimension=self.embed_size),
                     feat_col.embedding_column(items, dimension=self.embed_size)]

        feature_cols = deep_cols  # .append(feat_col.indicator_column(wide_cols))
        model = tf.estimator.Estimator(model_dir="wide_deep_dir",
                                       model_fn=self.my_model,
                                       params={'feature_columns': feature_cols,
                                               'wide_columns': self.wide_cols,
                                               'hidden_units': [128, 64]})
        return model


    def my_model(self, features, labels, mode, params):
        model_input = feat_col.input_layer(features, params['feature_columns'])
        for units in params['hidden_units']:
            model_input = tf.layers.dense(model_input, units=units, activation=tf.nn.relu)
        dnn_logits = tf.layers.dense(model_input, units=1, activation=None)

    #    linear_input = feat_col.input_layer(features, params['wide_columns'])
        features = tf.parse_example(
        #    serialized=tf.placeholder(tf.string, name='tf_example'),
            serialized=["wide_columns"],
            features=feat_col.make_parse_example_spec([self.wide_cols]))
        linear_logits = feat_col.linear_model(units=1, features=features,
                                              feature_columns=[self.wide_cols])

    #    logits = []
    #    for logits in [dnn_logits, linear_logits]:  # shape: [batch_size, 1]
    #        if logits is not None:
    #            logits.append(logits)
        logits = tf.add_n([dnn_logits, linear_logits])

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {'prediction': logits}
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        labels = tf.cast(tf.reshape(labels, [-1,1]), tf.float32)
        loss = tf.losses.mean_squared_error(labels=labels, predictions=logits)
        rmse = tf.metrics.root_mean_squared_error(labels=labels, predictions=logits)
        metrics = {'rmse': rmse}

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

        assert mode == tf.estimator.ModeKeys.TRAIN
        optimizer = tf.train.FtrlOptimizer(learning_rate=0.1, l1_regularization_strength=1e-3) ### Adagrad
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
            print("Epoch {} training time: {:.4f}".format(epoch, time.time() - t0))
            print("train loss: {loss:.4f}, train rmse: {rmse:.4f}".format(**train_result))
            print("test loss: {loss:.4f}, test rmse: {rmse:.4f}".format(**eval_result))