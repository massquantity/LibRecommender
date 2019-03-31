import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict
import time
from multiprocessing import Pool
import numpy as np
import pandas as pd
from math import sqrt
import tensorflow as tf


class Dataset:
    def __init__(self):
        self.data = defaultdict(dict)
        self.user2id = dict()
        self.item2id = dict()
        self.id2user = dict()
        self.id2item = dict()
        self.user_indices = list()
        self.item_indices = list()
        self.ratings = list()

    @classmethod
    def load_dataset(cls, data_path="../ml-1m/ratings.dat", shuffle=False):
        train_set = open(data_path, 'r').readlines()
        if shuffle:
            train_set = np.random.permutation(train_set)
        return train_set

    def build_dataset(self, train_set):
        index_user = 0
        index_item = 0
        for line in train_set:
            user = line.split("::")[0]
            item = line.split("::")[1]
            rating = line.split("::")[2]
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
            self.user_indices.append(user_id)
            self.item_indices.append(item_id)
            self.ratings.append(int(rating))
            self.data[user_id].update(dict(zip([item_id], [int(rating)])))
        self.user_indices = np.array(self.user_indices)
        self.item_indices = np.array(self.item_indices)
        self.ratings = np.array(self.ratings)


    def train_test_split(self, seed=42, train_frac=0.8, shuffle=False):
        np.random.seed(int(seed))
        if shuffle:
            user_indices = np.random.permutation(self.user_indices)
            item_indices = np.random.permutation(self.item_indices)
            ratings = np.random.permutation(self.ratings)
        else:
            user_indices = self.user_indices
            item_indices = self.item_indices
            ratings = self.ratings
        split = int(train_frac * len(self.user_indices))
        print("train data length: {}, test data length:{}".format(
            split, len(self.user_indices) - split))
        train_user_indices, test_user_indices = user_indices[:split], user_indices[split:]
        train_item_indices, test_item_indices = item_indices[:split], item_indices[split:]
        train_ratings, test_ratings = ratings[:split], ratings[split:]
        train_data = defaultdict(dict)
        test_data = defaultdict(dict)
        for u, i, r in zip(train_user_indices, train_item_indices, train_ratings):
            train_data[u].update(dict(zip([i], [r])))
        for u, i, r in zip(test_user_indices, test_item_indices, test_ratings):
            test_data[u].update(dict(zip([i], [r])))
        return train_user_indices, train_item_indices, train_ratings, train_data, \
               test_user_indices, test_item_indices, test_ratings, test_data

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

        return train_user_indices, train_item_indices, train_ratings, train_data, \
               test_user_indices, test_item_indices, test_ratings, test_data



#    def load_pandas

#    def load_implicit_data

    def ratings(dataset):
        for user, r in dataset.items():
            for item, rating in r.items():
                yield user, item, rating

    @property
    def get_id2user(self):
        return {idx: user for user, idx in self.user2id.items()}