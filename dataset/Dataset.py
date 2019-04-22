import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict
import time
from multiprocessing import Pool
import numpy as np
import pandas as pd
from math import sqrt
import tensorflow as tf
from sklearn.preprocessing import KBinsDiscretizer
from ..utils.negative_sampling import negative_sampling


class Dataset:
    def __init__(self):
    #    self.data_user = defaultdict(dict)
    #    self.data_item = defaultdict(dict)
        self.train_user = defaultdict(dict)
        self.train_item = defaultdict(dict)
    #    self.test_user = defaultdict(dict)
    #    self.test_item = defaultdict(dict)
        self.user2id = dict()
        self.item2id = dict()
        self.id2user = dict()
        self.id2item = dict()
        self.train_user_indices = list()
        self.train_item_indices = list()
        self.train_ratings = list()
        self.test_user_indices = list()
        self.test_item_indices = list()
        self.test_ratings = list()
        self.train_timestamp = list()
        self.test_timestamp = list()

    @classmethod
    def load_dataset(cls, data_path="../ml-1m/ratings.dat", shuffle=False):
        loaded_data = open(data_path, 'r').readlines()
        if shuffle:
            loaded_data = np.random.permutation(loaded_data)
        return loaded_data

    def build_dataset(self, data_path="../ml-1m/ratings.dat", shuffle=True, length="all",
                      train_frac=0.8, implicit=False, batch_size=256, seed=42, time_bin=0, num_neg=None):
        np.random.seed(seed)
        self.batch_size = batch_size
        index_user = 0
        index_item = 0
        loaded_data = open(data_path, 'r').readlines()
        if shuffle:
            loaded_data = np.random.permutation(loaded_data)
        if length == "all":
            length = len(loaded_data)
        for i, line in enumerate(loaded_data[:length]):
            user = line.split("::")[0]
            item = line.split("::")[1]
            rating = line.split("::")[2]
            timestamp = line.split("::")[3]
            try:
                user_id = self.user2id[user]
            except KeyError:
                user_id = index_user
                self.user2id[user] = index_user
                index_user += 1
            try:
                item_id = self.item2id[item]
            except KeyError:
                item_id = index_item
                self.item2id[item] = index_item
                index_item += 1

            if i <= int(train_frac * length):
                self.train_user_indices.append(user_id)
                self.train_item_indices.append(item_id)
                self.train_ratings.append(int(rating))
                self.train_timestamp.append(int(timestamp))
                self.train_user[user_id].update(dict(zip([item_id], [int(rating)])))
                self.train_item[item_id].update(dict(zip([user_id], [int(rating)])))
            else:
                self.test_user_indices.append(user_id)
                self.test_item_indices.append(item_id)
                self.test_ratings.append(int(rating))
                self.test_timestamp.append(int(timestamp))

        self.train_user_indices = np.array(self.train_user_indices)
        self.train_item_indices = np.array(self.train_item_indices)
        self.train_ratings = np.array(self.train_ratings)
        self.train_timestamp = np.array(self.train_timestamp)

        if time_bin > 0:
            self.kb = KBinsDiscretizer(n_bins=time_bin, encode="ordinal", strategy="uniform")  #### quantile
            self.train_timestamp_bin = self.kb.fit_transform(self.train_timestamp.reshape(-1, 1))
            self.train_timestamp_bin = self.train_timestamp_bin.astype(int)

        if implicit:
            self.train_labels = np.ones(len(self.train_ratings), dtype=np.float32)
            self.build_trainset_implicit(num_neg)

    #    self.test_user_indices = np.array(self.test_user_indices)
    #    self.test_item_indices = np.array(self.test_item_indices)
    #    self.test_ratings = np.array(self.test_ratings)

        print("testset size before: ", len(self.test_ratings))
        test_all = np.concatenate([np.expand_dims(self.test_user_indices, 1),
                                   np.expand_dims(self.test_item_indices, 1),
                                   np.expand_dims(self.test_ratings, 1),
                                   np.expand_dims(self.test_timestamp, 1)], axis=1)
        test_safe = test_all[(test_all[:, 0] < self.n_users) & (test_all[:, 1] < self.n_items)]
        test_danger = test_all[(test_all[:, 0] >= self.n_users) & (test_all[:, 1] >= self.n_items)]
        self.test_user_indices = test_safe[:, 0]
        self.test_item_indices = test_safe[:, 1]
        self.test_ratings = test_safe[:, 2]
        self.test_timestamp = test_safe[:, 3]

        if time_bin > 0:
            self.test_timestamp_bin = self.kb.transform(self.test_timestamp.reshape(-1, 1))
            self.test_timestamp_bin = self.test_timestamp_bin.astype(int)

        if implicit:
            self.test_labels = np.ones(len(self.test_ratings), dtype=np.float32)
            self.build_testset_implicit(num_neg)
    #        self.neg = negative_sampling(self, 4, self.batch_size)
        #    self.build_trainset_implicit()
        #    self.build_testset_implicit()
        print("testset size after: ", len(self.test_ratings))
        return self


    # TODO: split k test sample from each user
    def train_test_split_LOOV(self, k):
        """
        leave-last-k-out-split
        :return: train - test, user - item - ratings
        """
        train_user_indices = []
        test_user_indices = []
        train_item_indices = []
        test_item_indices = []
        train_ratings = []
        test_ratings = []
        train_data = defaultdict(dict)
        test_data = defaultdict(dict)

        _, user_position, user_counts = np.unique(self.user_indices,
                                                  return_inverse=True,
                                                  return_counts=True)
        user_indices = np.split(np.argsort(user_position, kind="mergesort"),
                                np.cumsum(user_counts)[:-1])

        for u in self.data.keys():
            user_length = len(user_indices[u])
            if user_length <= k:
                p = 1
            else:
                p = k
            train_indices = user_indices[u][:-p]
            test_indices = user_indices[u][-p:]

            train_user_indices.extend(self.user_indices[train_indices])
            test_user_indices.extend(self.user_indices[test_indices])
            train_item_indices.extend(self.item_indices[train_indices])
            test_item_indices.extend(self.item_indices[test_indices])
            train_ratings.extend(self.ratings[train_indices])
            test_ratings.extend(self.ratings[test_indices])

        for u, i, r in zip(train_user_indices, train_item_indices, train_ratings):
            train_data[u].update(dict(zip([i], [r])))
        for u, i, r in zip(test_user_indices, test_item_indices, test_ratings):
            test_data[u].update(dict(zip([i], [r])))

        return (train_user_indices, train_item_indices, train_ratings, train_data), \
               (test_user_indices, test_item_indices, test_ratings, test_data)


#   TODO
#   def load_pandas

    def build_trainset_implicit(self, num_neg):
        neg = negative_sampling(self, num_neg)
        self.train_user_implicit, \
        self.train_item_implicit, \
        self.train_label_implicit, \
        self.train_timestamp = neg(mode="train")


    def build_testset_implicit(self, num_neg):
        neg = negative_sampling(self, num_neg)
        self.test_user_implicit, \
        self.test_item_implicit, \
        self.test_label_implicit, \
        self.test_timestamp = neg(mode="test")


    def load_tf_trainset(self, batch_size=1):
        trainset_tf = tf.data.Dataset.from_tensor_slices({'user': self.train_user_indices,
                                                          'item': self.train_item_indices,
                                                          'rating': self.train_ratings})
        self.trainset_tf = trainset_tf.shuffle(len(self.train_ratings)).batch(batch_size)
        return self

    def load_tf_testset(self):
        testset_tf = tf.data.Dataset.from_tensor_slices({'user': self.test_user_indices,
                                                         'item': self.test_item_indices,
                                                         'rating': self.test_ratings})
        self.testset_tf = testset_tf.filter(lambda x: (x['user'] < self.n_users) & (x['item'] < self.n_items))
        return self

    def ratings(dataset):
        for user, r in dataset.items():
            for item, rating in r.items():
                yield user, item, rating

    @property
    def get_id2user(self):
        return {idx: user for user, idx in self.user2id.items()}

    @property
    def global_mean(self):
        return np.mean(self.train_ratings)

    @property
    def n_users(self):
        return len(self.train_user)

    @property
    def n_items(self):
        return len(self.train_item)