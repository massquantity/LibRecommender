import time
import numpy as np
import tensorflow as tf
from tensorflow import feature_column as feat_col


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
        wide_cols = tf.feature_column.crossed_column([users, items], hash_bucket_size=1000)
        deep_cols = [tf.feature_column.embedding_column(users, dimension=self.embed_size),
                     tf.feature_column.embedding_column(items, dimension=self.embed_size)]

        config = tf.estimator.RunConfig(log_step_count_steps=1000, save_checkpoints_steps=10000)
        self.model = tf.estimator.DNNLinearCombinedRegressor(
            model_dir="wide_deep_dir",
            config=config,
            linear_feature_columns=wide_cols,
            linear_optimizer="Ftrl",
            dnn_feature_columns=deep_cols,
            dnn_hidden_units=[128, 64],
            dnn_dropout=0.0,
            batch_norm=False,
            loss_reduction=tf.losses.Reduction.MEAN)

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
                return train_data.shuffle(len(data.test_ratings)).repeat(self.n_epochs).batch(self.batch_size)

        self.build_model(dataset)
        for epoch in range(1, 101):  # epoch_per_eval
            t0 = time.time()
            self.model.train(lambda: input_fn(mode="train"))
            train_loss = self.model.evaluate(lambda: input_fn(mode="train"))
            test_loss = self.model.evaluate(lambda: input_fn(mode="evaluate"))
            print("Epoch {} training time: {:.4f}".format(epoch, time.time() - t0))
            print("train loss: ", train_loss)
            print("test loss: ", test_loss)




