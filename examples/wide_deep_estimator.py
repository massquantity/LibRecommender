import os
import time
import functools
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import predictor
from tensorflow import feature_column as fc
from libreco.dataset import DatasetPure, DatasetFeat
from libreco.algorithms import WideDeepEstimator
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


flags = tf.app.flags
# flags.DEFINE_string("saved_model_dir", "./saved_model_dir", "Path to the saved model.")
flags.DEFINE_string("export_model_dir", "./export_model_dir", "Path to the export model.")
flags.DEFINE_string("train_path", None, "Path to the training data, tfrecord format.")
flags.DEFINE_string("eval_path", None, "Path to the evaluate data, tfrecord format.")
flags.DEFINE_string("orig_path", None, "Path to the original data, csv format.")
flags.DEFINE_string("eval_info", None, "Path to the evaluate information, numpy npy format.")
flags.DEFINE_integer("embed_size", 32, "Embedding vector size")
flags.DEFINE_integer("epochs", 10, "Total training epochs")
flags.DEFINE_string("hidden_units", "256,128,64", "Comma-separated list of number of hidden units")
flags.DEFINE_string("eval_top_n", "5,20,50", "Comma-separated list of top evaluate numbers")
flags.DEFINE_integer("batch_size", 256, "Training batch size")
flags.DEFINE_float("lr", 0.005, "Learning rate")
flags.DEFINE_string("task", "ranking", "Specific task: rating or raking")
flags.DEFINE_boolean("use_bn", False, "Whether to use batch normalization for hidden layers")
flags.DEFINE_boolean("export_and_load", False, "Whether to export and load model")
flags.DEFINE_integer("n_rec", 7, "Number of recommended items for a user")
FLAGS = flags.FLAGS


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

    wide_cols = [users, items, gender, age, occupation, genre1, genre2, genre3,
                 fc.crossed_column([gender, age, occupation], hash_bucket_size=hash_size),
                 fc.crossed_column([age, genre1], hash_bucket_size=hash_size)]

    embed_cols = [users, items, age, occupation]
    deep_cols = list()
    for col in embed_cols:
        deep_cols.append(fc.embedding_column(col, embed_size))

    shared_embed_cols = [genre1, genre2, genre3]
    deep_cols.extend(fc.shared_embedding_columns(shared_embed_cols, embed_size))
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


def main(unused_argv):
    dataset = pd.read_csv(FLAGS.orig_path, header=None, sep=",",
                          names=["user", "item", "label", "gender", "age",
                                 "occupation", "genre1", "genre2", "genre3"])
    wide_cols, deep_cols, feat_func = create_feature_columns(dataset)
    eval_info = np.load(FLAGS.eval_info, allow_pickle=True)  # used for evaluate
    user_dict, item_dict, item_info, item_indices = get_unique_info(dataset)
    pred_feat_func = functools.partial(predict_info, user_dict=user_dict, item_dict=item_dict)
    rank_feat_func = functools.partial(rank_info, user_dict=user_dict, item_info=item_info)
    if FLAGS.use_bn:
        print("use batch normalization...")

    wde = WideDeepEstimator(lr=FLAGS.lr,
                            embed_size=FLAGS.embed_size,
                            n_epochs=FLAGS.epochs,
                            batch_size=FLAGS.batch_size,
                            use_bn=FLAGS.use_bn,
                            task=FLAGS.task,
                            pred_feat_func=pred_feat_func,
                            rank_feat_func=rank_feat_func,
                            item_indices=item_indices)
    wde.fit(wide_cols, deep_cols, FLAGS.train_path, FLAGS.eval_path, feat_func, eval_info, verbose=2)
    print(wde.predict_ui(1, 2))
    t6 = time.time()
    print(wde.recommend_user(1, n_rec=7))
    print("recommend time: ", time.time() - t6)

    if FLAGS.export_and_load:
        wde.model.export_saved_model(FLAGS.export_model_dir, serving_input_receiver_fn)

        sub_dirs = [x for x in Path(FLAGS.export_model_dir).iterdir() if x.is_dir() and 'temp' not in str(x)]
        latest = str(sorted(sub_dirs)[-1])
        predict_fn = predictor.from_saved_model(latest)
        samples = rank_info(user_dict, 1, item_info)

        t1 = time.time()
        rank = predict_fn(samples)["probabilities"].ravel()
        indices = np.argpartition(rank, -FLAGS.n_rec)[-FLAGS.n_rec:]
        print("recommend for user 1: %s" % sorted(zip(item_indices[indices], rank[indices]), key=lambda x: -x[1]))
        print("predict_fn recommend time: ", time.time() - t1)


if __name__ == "__main__":
    tf.app.run(main=main)

# from libreco.utils import download_data
# download_data.prepare_data("par_path="...", feat=True")


# from wide_deep_tfrecord_estimator import export_TFRecord
# export_TFRecord(par_dir="...", convert_implicit=True, num_neg=2, task="ranking")


# $python wide_deep_estimator.py
#         --train_path "../ml-1m/train.tfrecord" \
#         --eval_path "../ml-1m/test.tfrecord" \
#         --orig_path "../ml-1m/merged_data.csv" \
#         --eval_info "../ml-1m/test_data.npy" \
#         --epochs 10 \
#         --export_and_load True

