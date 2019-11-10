import os
import time
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column as fc


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

