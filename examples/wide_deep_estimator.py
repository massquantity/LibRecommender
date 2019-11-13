import os
import time
import functools
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import predictor
from tensorflow import feature_column as fc
from libreco.dataset import DatasetPure, DatasetFeat
from libreco.algorithms import WideDeepEstimator
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def create_feature_columns(dataset, embed_size=32, hash_size=10000):
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

    label = fc.numeric_column("label", default_value=0.0, dtype=tf.float32)
    feat_columns = [label]
    feat_columns += wide_cols
    feat_columns += deep_cols
    feat_spec = fc.make_parse_example_spec(feat_columns)
    return wide_cols, deep_cols, feat_spec


def get_unique_info(data):
    user_unique = data.drop_duplicates("user")
    user_dict = dict()
    for u in user_unique.user.values:
        user_dict[u] = user_unique.loc[user_unique.user == u, ["gender", "age", "occupation"]].values.ravel()

    item_unique = data.drop_duplicates("item")
    item_dict = dict()
    for i in item_unique.item.values:
        item_dict[i] = item_unique.loc[item_unique.item == i, ["genre1", "genre2", "genre3"]].values.ravel()

    item_info = item_unique.sort_values("item", ascending=True)
    item_indices = item_info.item.values
    return user_dict, item_dict, item_info, item_indices


def predict_info(user_dict, item_dict, u, i):
#    user_info = data[data.user == u].values[0]
#    item_info = data[data.item == i].values[0]
    features = {
        "user": np.array(u).reshape(-1, 1),
        "item": np.array(i).reshape(-1, 1),
        "gender": np.array(user_dict[u][0]).reshape(-1, 1),
        "age": np.array(user_dict[u][1]).reshape(-1, 1),
        "occupation": np.array(user_dict[u][2]).reshape(-1, 1),
        "genre1": np.array(item_dict[i][0]).reshape(-1, 1),
        "genre2": np.array(item_dict[i][1]).reshape(-1, 1),
        "genre3": np.array(item_dict[i][2]).reshape(-1, 1)
    }
    return features


def rank_info(user_dict, u, item_info):
    n_items = len(item_info)
    features = {
        "user": np.tile(u, [n_items, 1]),
        "item": item_info.item.values.reshape(-1, 1),
        "gender": np.tile(user_dict[u][0], [n_items, 1]),
        "age": np.tile(user_dict[u][1], [n_items, 1]),
        "occupation": np.tile(user_dict[u][2], [n_items, 1]),
        "genre1": item_info.genre1.values.reshape(-1, 1),
        "genre2": item_info.genre2.values.reshape(-1, 1),
        "genre3": item_info.genre3.values.reshape(-1, 1),
    }
    return features


def serving_input_receiver_fn():
    features = {"user": tf.placeholder(tf.int64, shape=[None, 1]),
                "item": tf.placeholder(tf.int64, shape=[None, 1]),
                "gender": tf.placeholder(tf.string, shape=[None, 1]),
                "age": tf.placeholder(tf.int64, shape=[None, 1]),
                "occupation": tf.placeholder(tf.int64, shape=[None, 1]),
                "genre1": tf.placeholder(tf.string, shape=[None, 1]),
                "genre2": tf.placeholder(tf.string, shape=[None, 1]),
                "genre3": tf.placeholder(tf.string, shape=[None, 1])}

    return tf.estimator.export.ServingInputReceiver(features, features)


def get_tf_feat():
    feat_func = {"user": tf.io.FixedLenFeature([], tf.int64),
                 "item": tf.io.FixedLenFeature([], tf.int64),
                 "label": tf.io.FixedLenFeature([], tf.float32),
                 "gender": tf.io.FixedLenFeature([], tf.string),
                 "age": tf.io.FixedLenFeature([], tf.int64),
                 "occupation": tf.io.FixedLenFeature([], tf.int64),
                 "genre1": tf.io.FixedLenFeature([], tf.string),
                 "genre2": tf.io.FixedLenFeature([], tf.string),
                 "genre3": tf.io.FixedLenFeature([], tf.string)}
    return feat_func


if __name__ == "__main__":
    dataset = pd.read_csv("../ml-1m/merged_data.csv", header=None, sep=",",
                          names=["user", "item", "label", "gender", "age",
                                 "occupation", "genre1", "genre2", "genre3"])
    wide_cols, deep_cols, feat_func = create_feature_columns(dataset)
    eval_info = np.load("../ml-1m/test_data.npy", allow_pickle=True)  # used for evaluate

    t1 = time.time()
    user_dict, item_dict, item_info, item_indices = get_unique_info(dataset)
    print("get user-item info time: {:.4f}".format(time.time() - t1))

    pred_feat_func = functools.partial(predict_info, user_dict=user_dict, item_dict=item_dict)
    rank_feat_func = functools.partial(rank_info, user_dict=user_dict, item_info=item_info)

    wde = WideDeepEstimator(lr=0.005, embed_size=32, n_epochs=0, batch_size=4096, use_bn=True,
                            task="ranking", pred_feat_func=pred_feat_func,
                            rank_feat_func=rank_feat_func, item_indices=item_indices)
    wde.fit(wide_cols, deep_cols, "../ml-1m/train.tfrecord", "../ml-1m/test.tfrecord",
            feat_func, eval_info, verbose=2)
    print(wde.predict_ui(1, 2))
    t6 = time.time()
    print(wde.recommend_user(1, n_rec=7))
    print("rec time: ", time.time() - t6)

    wde.model.export_saved_model("./export_model_dir", serving_input_receiver_fn)

    from pathlib import Path
    subdirs = [x for x in Path("./export_model_dir").iterdir()
               if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])

    predict_fn = predictor.from_saved_model(latest)
    samples = rank_info(user_dict, 1, item_info)
    t1 = time.time()
    print(predict_fn(samples)["probabilities"][0])
    print("rec time: ", time.time() - t1)

