import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict
import time
from multiprocessing import Pool
import numpy as np
import pandas as pd
from math import sqrt
import tensorflow as tf
from ..utils.sampling import negative_sampling


class Dataset:
    def __init__(self):
        self.train_user = defaultdict(dict)
        self.train_item = defaultdict(dict)
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

#    TODO
#    @classmethod
#    def load_builtin_dataset(cls, data_path="../ml-1m/ratings"):

    @classmethod
    def load_builtin_dataset(cls, data_path="../ml-1m/ratings.dat", shuffle=False):
        loaded_data = open(data_path, 'r').readlines()
        if shuffle:
            loaded_data = np.random.permutation(loaded_data)
        return loaded_data

    def build_dataset(self, data_path="../ml-1m/ratings.dat", shuffle=True, length="all",
                      train_frac=0.8, convert_implicit=False, build_negative=False, batch_size=256,
                      seed=42, num_neg=None):
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
                self.train_user[user_id].update(dict(zip([item_id], [int(rating)])))
                self.train_item[item_id].update(dict(zip([user_id], [int(rating)])))
            else:
                self.test_user_indices.append(user_id)
                self.test_item_indices.append(item_id)
                self.test_ratings.append(int(rating))

        self.train_user_indices = np.array(self.train_user_indices)
        self.train_item_indices = np.array(self.train_item_indices)
        self.train_ratings = np.array(self.train_ratings)

        if convert_implicit:
            self.train_labels = np.ones(len(self.train_ratings), dtype=np.float32)

        if build_negative:
            self.build_trainset_implicit(num_neg)

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

        if convert_implicit:
            self.test_labels = np.ones(len(self.test_ratings), dtype=np.float32)

        if build_negative:
            self.build_testset_implicit(num_neg)
    #        self.neg = negative_sampling(self, 4, self.batch_size)
        #    self.build_trainset_implicit()
        #    self.build_testset_implicit()
        print("testset size after: ", len(self.test_ratings))
        return self

    def leave_kout_split(self, k, data_path, length="all", sep=",", shuffle=True, seed=42):
        """
        leave-last-k-out-split : split k test sample from each user
        :return: train - test, user - item - ratings
        """
        np.random.seed(seed)
        self.user_indices = []
        self.item_indices = []
        self.ratings = []
        user2id = dict()
        item2id = dict()
        train_user_indices = list()
        train_item_indices = list()
        train_ratings = list()
        test_user_indices = list()
        test_item_indices = list()
        test_ratings = list()

        index_user = 0
        index_item = 0
        loaded_data = open(data_path, 'r').readlines()
        if length == "all":
            length = len(loaded_data)
        for i, line in enumerate(loaded_data[:length]):
            user = line.split(sep)[0]
            item = line.split(sep)[1]
            rating = line.split(sep)[2]

            try:
                user_id = user2id[user]
            except KeyError:
                user_id = index_user
                user2id[user] = index_user
                index_user += 1
            try:
                item_id = item2id[item]
            except KeyError:
                item_id = index_item
                item2id[item] = index_item
                index_item += 1

            self.user_indices.append(user_id)
            self.item_indices.append(item_id)
            self.ratings.append(int(rating))

        self.user_indices = np.array(self.user_indices)
        self.item_indices = np.array(self.item_indices)
        self.ratings = np.array(self.ratings)

        users, user_position, user_counts = np.unique(self.user_indices,
                                                      return_inverse=True,
                                                      return_counts=True)
        user_split_indices = np.split(np.argsort(user_position, kind="mergesort"),
                                      np.cumsum(user_counts)[:-1])

        test_user_temp = []
        test_item_temp = []
        test_rating_temp = []

        for u in users:
            user_length = len(user_split_indices[u])
            if user_length <= 1 or k == 0:
                train_indices = user_split_indices[u]
                test_indices = []
            elif user_length <= k:
                p = 1
                train_indices = user_split_indices[u][:-p]
                test_indices = user_split_indices[u][-p:]
            else:
                p = k
                train_indices = user_split_indices[u][:-p]
                test_indices = user_split_indices[u][-p:]

            train_user_indices.extend(self.user_indices[train_indices])
            train_item_indices.extend(self.item_indices[train_indices])
            train_ratings.extend(self.ratings[train_indices])

            test_user_temp.extend(self.user_indices[test_indices])
            test_item_temp.extend(self.item_indices[test_indices])
            test_rating_temp.extend(self.ratings[test_indices])

        print("item before: ", len(test_item_temp))
        train_item_pool = np.unique(train_item_indices)
        for user, item, rating in zip(test_user_temp, test_item_temp, test_rating_temp):
            if item in train_item_pool:
                test_user_indices.append(user)
                test_item_indices.append(item)
                test_ratings.append(rating)
        print("item after: ", len(test_item_indices))

        user_mapping = dict(zip(set(train_user_indices), np.arange(len(set(train_user_indices)))))
        item_mapping = dict(zip(set(train_item_indices), np.arange(len(set(train_item_indices)))))
        for user, item, rating in zip(train_user_indices, train_item_indices, train_ratings):
            self.train_user_indices.append(user_mapping[user])
            self.train_item_indices.append(item_mapping[item])
            self.train_ratings.append(rating)

        for test_u, test_i, test_r in zip(test_user_indices, test_item_indices, test_ratings):
            self.test_user_indices.append(user_mapping[test_u])
            self.test_item_indices.append(item_mapping[test_i])
            self.test_ratings.append(test_r)

        if shuffle:
            random_mask = np.random.choice(len(self.train_user_indices), len(self.train_user_indices), replace=False)
            self.train_user_indices = np.array(self.train_user_indices)[random_mask]
            self.train_item_indices = np.array(self.train_item_indices)[random_mask]
            self.train_ratings = np.array(self.train_ratings)[random_mask]
        else:
            self.train_user_indices = np.array(self.train_user_indices)
            self.train_item_indices = np.array(self.train_item_indices)
            self.train_ratings = np.array(self.train_ratings)

        self.test_user_indices = np.array(self.test_user_indices)
        self.test_item_indices = np.array(self.test_item_indices)
        self.test_ratings = np.array(self.test_ratings)

        for u, i, r in zip(self.train_user_indices, self.train_item_indices, self.train_ratings):
            self.train_user[u].update(dict(zip([i], [r])))
            self.train_item[i].update(dict(zip([u], [r])))
        return self


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

    # TODO
    def build_tf_sparse(self):
        pass


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

    def ratings(self):
        for user, r in self.train_user:
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
    #    return len(np.unique(self.train_user_indices))

    @property
    def n_items(self):
        return len(self.train_item)
    #    return len(np.unique(self.train_item_indices))