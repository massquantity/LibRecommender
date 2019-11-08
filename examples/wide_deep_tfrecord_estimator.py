import os
from collections import defaultdict
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


def preprocess_data(par_path=None, convert_implicit=False, num_neg=0, task="nce", test_frac=0.2, seed=42):
    file_path = os.path.join(par_path, "merged_data.csv")
    if task == "rating" or task == "ranking":
        dataset = pd.read_csv(file_path, header=None, sep=",",
                              names=["user", "item", "label", "gender", "age",
                                     "occupation", "genre1", "genre2", "genre3"])

        user_unique = np.unique(dataset.user.values)
        print("num users: ", len(user_unique))
        user_id_map = dict(zip(user_unique, np.arange(len(user_unique))))
        dataset["user"] = dataset["user"].map(user_id_map)

        item_unique = np.unique(dataset.item.values)
        print("num items: ", len(item_unique))
        item_id_map = dict(zip(item_unique, np.arange(len(item_unique))))
        dataset["item"] = dataset["item"].map(item_id_map)

        if convert_implicit:
            dataset["label"] = 1.0
        else:
            dataset["label"] = dataset["label"].astype(np.float32)

        train_data, test_data = train_test_split(dataset, test_size=test_frac, random_state=seed, shuffle=True)
        train_data = train_data.values
        test_data = test_data.values

        if num_neg > 0:
            print("carry out negative sampling...")
            item_dict = dict()    # item_dict contains unique items and their features
            total_item_cols = ["item", "genre1", "genre2", "genre3"]
            dataset_items = dataset[total_item_cols]
            total_items_unique = dataset_items.drop_duplicates()
            total_items = total_items_unique["item"].values
            total_item_features = total_items_unique[["genre1", "genre2", "genre3"]].values
            for item, item_feat_cols in zip(total_items, total_item_features):
                item_dict[item] = item_feat_cols

            consumed_items = defaultdict(set)   # items that one user has consumed
            for user, item in zip(train_data[:, 0], train_data[:, 1]):
                consumed_items[user].add(item)

            train_negative_samples = list()
            for s in train_data:
                sample = s.tolist()
                u = sample[0]
                for _ in range(num_neg):
                    item_neg = np.random.randint(0, len(item_unique))
                    while item_neg in consumed_items[u]:
                        item_neg = np.random.randint(0, len(item_unique))
                    sample[1] = item_neg
                    sample[2] = 0.0

                    neg_item = item_dict[item_neg]
                    for col, orig_col in enumerate([6, 7, 8]):  # col index for ["genre1", "genre2", "genre3"]
                        sample[orig_col] = neg_item[col]

                    train_negative_samples.append(sample)
            train_data = np.concatenate([train_data, train_negative_samples], axis=0)
            train_data = np.random.permutation(train_data)
            train_data[:, 0] = train_data[:, 0].astype(np.int64)
            train_data[:, 1] = train_data[:, 1].astype(np.int64)
            train_data[:, 2] = train_data[:, 2].astype(np.float32)
            train_data[:, 4] = train_data[:, 4].astype(np.int64)
            train_data[:, 5] = train_data[:, 5].astype(np.int64)

            # sample test data
            for user, item in zip(test_data[:, 0], test_data[:, 1]):
                consumed_items[user].add(item)

            test_negative_samples = list()
            for s in test_data:
                sample = s.tolist()
                u = sample[0]
                for _ in range(num_neg):
                    item_neg = np.random.randint(0, len(item_unique))
                    while item_neg in consumed_items[u]:
                        item_neg = np.random.randint(0, len(item_unique))
                    sample[1] = item_neg
                    sample[2] = 0.0

                    neg_item = item_dict[item_neg]
                    for col, orig_col in enumerate([6, 7, 8]):
                        sample[orig_col] = neg_item[col]

                    test_negative_samples.append(sample)
            test_data = np.concatenate([test_data, test_negative_samples], axis=0)
            test_data = np.random.permutation(test_data)
            test_data[:, 0] = test_data[:, 0].astype(np.int64)
            test_data[:, 1] = test_data[:, 1].astype(np.int64)
            test_data[:, 2] = test_data[:, 2].astype(np.float32)
            test_data[:, 4] = test_data[:, 4].astype(np.int64)
            test_data[:, 5] = test_data[:, 5].astype(np.int64)

            print("train size after negative sampling: {}, "
                  "test size after negative sampling: {}".format(len(train_data), len(test_data)))

    elif task == "nce":
        dataset = pd.read_csv(file_path, header=None, usecols=[0, 1, 3, 4, 5],
                              names=["user", "item", "gender", "age", "occupation"])

        user_unique = np.unique(dataset.user.values)
        print("num users: ", len(user_unique))
        user_id_map = dict(zip(user_unique, np.arange(len(user_unique))))
        dataset["user"] = dataset["user"].map(user_id_map)

        item_unique = np.unique(dataset.item.values)
        print("num items: ", len(item_unique))
        item_id_map = dict(zip(item_unique, np.arange(len(item_unique))))
        dataset["item"] = dataset["item"].map(item_id_map)

        train_data, test_data = train_test_split(dataset, test_size=test_frac, random_state=seed, shuffle=True)
        train_data = train_data.values
        test_data = test_data.values

    return train_data, test_data


def export_TFRecord(par_path=None, convert_implicit=False, num_neg=0, task="nce", test_frac=0.2,
                    seed=42, compress=False, compress_level=5):
    train_tfrecord_path = os.path.join(par_path, "train.tfrecord")
    test_tfrecord_path = os.path.join(par_path, "test.tfrecord")
    train_data, test_data = preprocess_data(par_path, convert_implicit, num_neg, task, test_frac, seed)
    if compress:
        compress_options = tf.io.TFRecordOptions(compression_type=tf.io.TFRecordCompressionType.GZIP,
                                                 compression_level=compress_level)
    else:
        compress_options = None

    if task == "rating" or task == "ranking":
        print("start export tfrecord")
        with tf.io.TFRecordWriter(train_tfrecord_path, options=compress_options) as w:
            for i in range(0, len(train_data)):
                tf_example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "user": tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[train_data[i, 0]])),
                            "item": tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[train_data[i, 1]])),
                            "label": tf.train.Feature(
                                float_list=tf.train.FloatList(value=[train_data[i, 2]])),
                            "gender": tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[train_data[i, 3].encode()])),
                            "age": tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[train_data[i, 4]])),
                            "occupation": tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[train_data[i, 5]])),
                            "genre1": tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[train_data[i, 6].encode()])),
                            "genre2": tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[train_data[i, 7].encode()])),
                            "genre3": tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[train_data[i, 8].encode()])),
                        }))
                w.write(tf_example.SerializeToString())
            print("train tfrecord export done !")

        with tf.io.TFRecordWriter(test_tfrecord_path, options=compress_options) as w:
            for i in range(0, len(test_data)):
                tf_example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "user": tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[test_data[i, 0]])),
                            "item": tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[test_data[i, 1]])),
                            "label": tf.train.Feature(
                                float_list=tf.train.FloatList(value=[test_data[i, 2]])),
                            "gender": tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[test_data[i, 3].encode()])),
                            "age": tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[test_data[i, 4]])),
                            "occupation": tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[test_data[i, 5]])),
                            "genre1": tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[test_data[i, 6].encode()])),
                            "genre2": tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[test_data[i, 7].encode()])),
                            "genre3": tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[test_data[i, 8].encode()])),
                        }))
                w.write(tf_example.SerializeToString())
            print("test tfrecord export done !")

    elif task == "nce":
        with tf.io.TFRecordWriter(train_tfrecord_path, options=compress_options) as w:
            for i in range(0, len(train_data)):
                tf_example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "user": tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[train_data[i, 0]])),
                            "item": tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[train_data[i, 1]])),
                            "gender": tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[train_data[i, 2].encode()])),
                            "age": tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[train_data[i, 3]])),
                            "occupation": tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[train_data[i, 4]])),
                        }))
                w.write(tf_example.SerializeToString())
            print("train tfrecord export done !")

        with tf.io.TFRecordWriter(test_tfrecord_path, options=compress_options) as w:
            for i in range(0, len(train_data)):
                tf_example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "user": tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[test_data[i, 0]])),
                            "item": tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[test_data[i, 1]])),
                            "gender": tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[test_data[i, 2].encode()])),
                            "age": tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[test_data[i, 3]])),
                            "occupation": tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[test_data[i, 4]])),
                        }))
                w.write(tf_example.SerializeToString())
            print("test tfrecord export done !")





