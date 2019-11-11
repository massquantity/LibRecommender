import os
import time
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column as fc
from libreco.dataset import DatasetPure, DatasetFeat
from libreco.algorithms import WideDeepEstimator
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def create_feature_columns(file_path, embed_size=32, hash_size=10000):
    dataset = pd.read_csv(file_path, header=None, sep=",",
                          names=["user", "item", "label", "gender", "age",
                                 "occupation", "genre1", "genre2", "genre3"])
    n_users = dataset.user.nunique()
    n_items = dataset.item.nunique()
    genre_list = dataset.genre1.unique()
    users = fc.categorical_column_with_vocabulary_list("user", np.arange(n_users), default_value=-1, dtype=tf.int64)
    items = fc.categorical_column_with_vocabulary_list("item", np.arange(n_items), default_value=-1, dtype=tf.int64)
    gender = fc.categorical_column_with_vocabulary_list("gender", ["M", "F"])
    age = fc.categorical_column_with_vocabulary_list("age", [1, 18, 25, 35, 45, 50, 56], dtype=tf.int64)
    occupation = fc.categorical_column_with_vocabulary_list("occupation", np.arange(21), dtype=tf.int64)
    genre1 = fc.categorical_column_with_vocabulary_list("genre1", genre_list)
    genre2 = fc.categorical_column_with_vocabulary_list("genre2", genre_list)
    genre3 = fc.categorical_column_with_vocabulary_list("genre3", genre_list)

    wide_cols = [users, items, gender, age, occupation, genre1, genre2, genre3]
    wide_cols.append(fc.crossed_column([gender, age, occupation], hash_bucket_size=hash_size))
    wide_cols.append(fc.crossed_column([age, genre1], hash_bucket_size=hash_size))

    embed_cols = [users, items, age, occupation, genre1, genre2, genre3]
    deep_cols = list()
    for col in embed_cols:
        deep_cols.append(fc.embedding_column(col, embed_size))
    deep_cols.append(fc.indicator_column(gender))

    return wide_cols, deep_cols



if __name__ == "__main__":
    conf_movielens = {
        "data_path": "../ml-1m/merged_data.csv",
        "length": 100000,
        "user_col": 0,
        "item_col": 1,
        "label_col": 2,
        "numerical_col": None,
        "categorical_col": [3, 4, 5],
        "merged_categorical_col": [[6, 7, 8]],
        "user_feature_cols": [3, 4, 5],
        "item_feature_cols": [6, 7, 8],
        "convert_implicit": True,
        "build_negative": True,
        "num_neg": 2,
        "batch_size": 256,
        "sep": ",",
    }

    conf = conf_movielens
    t0 = time.time()
    dataset = DatasetFeat(include_features=True)
    dataset.build_dataset(**conf)
    print("num users: {}, num items: {}".format(dataset.n_users, dataset.n_items))
    if conf.get("convert_implicit"):
        print("data size: ", len(dataset.train_labels_implicit) + len(dataset.test_labels_implicit))
    else:
        print("data size: ", len(dataset.train_user_indices) + len(dataset.test_user_indices))
    print("data processing time: {:.2f}".format(time.time() - t0))
    print()

    wide_cols, deep_cols = create_feature_columns("../ml-1m/merged_data.csv")
    wdc = WideDeepEstimator(lr=0.005, embed_size=32, n_epochs=100, batch_size=256, use_bn=False,
                            task="ranking", cross_features=False)
    wdc.fit(dataset, wide_cols, deep_cols, "../ml-1m/train.tfrecord", "../ml-1m/test.tfrecord")
    print(wdc.predict(1, 2))
    t6 = time.time()
    print(wdc.recommend_user(1, n_rec=10))
    print("rec time: ", time.time() - t6)


