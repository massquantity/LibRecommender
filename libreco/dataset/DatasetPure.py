import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict
import time
from multiprocessing import Pool
import numpy as np
import pandas as pd
from math import sqrt
import tensorflow as tf
from ..utils.sampling import NegativeSampling


class DatasetPure:
    def __init__(self):
        self.train_user = defaultdict(dict)
        self.train_item = defaultdict(dict)
        self.user2id = dict()
        self.item2id = dict()
        self.id2user = dict()
        self.id2item = dict()
        self.train_user_indices = list()
        self.train_item_indices = list()
        self.train_labels = list()
        self.test_user_indices = list()
        self.test_item_indices = list()
        self.test_labels = list()

#    TODO
#    @classmethod
#    def load_builtin_dataset(cls, data_path="../ml-1m/ratings"):

    @classmethod
    def load_builtin_dataset(cls, data_path="../ml-1m/ratings.dat", shuffle=False):
        loaded_data = open(data_path, 'r').readlines()
        if shuffle:
            loaded_data = np.random.permutation(loaded_data)
        return loaded_data

    def build_dataset(self, data_path="../ml-1m/ratings.dat", shuffle=True, length="all", sep=",",
                      user_col=None, item_col=None, label_col=None,
                      train_frac=0.8, convert_implicit=False, build_negative=False, build_tf_dataset=False,
                      batch_size=256, seed=42, num_neg=None, lower_upper_bound=None):
        np.random.seed(seed)
        self.batch_size = batch_size
        self.lower_upper_bound = lower_upper_bound
        if isinstance(num_neg, int) and num_neg > 0:
            self.num_neg = num_neg

    #    if not user_col or not item_col or not label_col:
        if not np.all([user_col, item_col, label_col]):
            user_col = 0
            item_col = 1
            label_col = 2

        index_user = 0
        index_item = 0
        with open(data_path, 'r') as f:
            loaded_data = f.readlines()
    #    f = open(data_path, 'r')
    #    loaded_data = f.readlines()
    #    f.close()
        if shuffle:
            loaded_data = np.random.permutation(loaded_data)
        if length == "all":
            length = len(loaded_data)
        for i, line in enumerate(loaded_data[:length]):
            user = line.split(sep)[user_col]
            item = line.split(sep)[item_col]
            label = line.split(sep)[label_col]
            if convert_implicit and label != 0:
                label = 1
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
                self.train_labels.append(float(label))
                self.train_user[user_id].update(dict(zip([item_id], [float(label)])))
                self.train_item[item_id].update(dict(zip([user_id], [float(label)])))
            else:
                self.test_user_indices.append(user_id)
                self.test_item_indices.append(item_id)
                self.test_labels.append(float(label))

        self.train_user_indices = np.array(self.train_user_indices)
        self.train_item_indices = np.array(self.train_item_indices)
        self.train_labels = np.array(self.train_labels)

        print("testset size before: ", len(self.test_labels))
        test_all = np.concatenate([np.expand_dims(self.test_user_indices, 1),
                                   np.expand_dims(self.test_item_indices, 1),
                                   np.expand_dims(self.test_labels, 1)], axis=1)
        test_safe = test_all[(test_all[:, 0] < self.n_users) & (test_all[:, 1] < self.n_items)]
        test_danger = test_all[(test_all[:, 0] >= self.n_users) & (test_all[:, 1] >= self.n_items)]
        self.test_user_indices = test_safe[:, 0].astype(int)
        self.test_item_indices = test_safe[:, 1].astype(int)
        self.test_labels = test_safe[:, 2]

    #    if convert_implicit:
    #        self.train_labels = np.ones(len(self.train_labels), dtype=np.float32)
    #        self.test_labels = np.ones(len(self.test_labels), dtype=np.float32)

        if build_negative:
            self.build_trainset_implicit(num_neg)
            self.build_testset_implicit(num_neg)

        if build_tf_dataset:
            self.load_tf_trainset(batch_size=self.batch_size)
            self.load_tf_testset()

        print("testset size after: ", len(self.test_labels))
        return self

    def leave_k_out_split(self, k, data_path, length="all", sep=",", shuffle=True, seed=42,
                          convert_implicit=False, build_negative=False, batch_size=256, num_neg=None):
        """
        leave-last-k-out-split : split k test sample from each user
        :return: train - test, user - item - ratings
        """
        np.random.seed(seed)
        self.batch_size = batch_size
        if num_neg is not None:
            self.num_neg = num_neg

        self.user_indices = []
        self.item_indices = []
        self.labels = []
        user2id = dict()
        item2id = dict()
        train_user_indices = list()
        train_item_indices = list()
        train_labels = list()
        test_user_indices = list()
        test_item_indices = list()
        test_labels = list()

        index_user = 0
        index_item = 0
        loaded_data = open(data_path, 'r').readlines()
        if length == "all":
            length = len(loaded_data)

        for i, line in enumerate(loaded_data[:length]):
            user = line.split(sep)[0]
            item = line.split(sep)[1]
            label = line.split(sep)[2]

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
            self.labels.append(int(label))

        self.user_indices = np.array(self.user_indices)
        self.item_indices = np.array(self.item_indices)
        self.labels = np.array(self.labels)

        users, user_position, user_counts = np.unique(self.user_indices,
                                                      return_inverse=True,
                                                      return_counts=True)
        user_split_indices = np.split(np.argsort(user_position, kind="mergesort"),
                                      np.cumsum(user_counts)[:-1])

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
            train_labels.extend(self.labels[train_indices])

            test_user_indices.extend(self.user_indices[test_indices])
            test_item_indices.extend(self.item_indices[test_indices])
            test_labels.extend(self.labels[test_indices])

        print("item before: ", len(test_item_indices))
        train_item_pool = np.unique(train_item_indices)   # remove items in test data that are not in train data
    #    for user, item, label in zip(test_user_temp, test_item_temp, test_label_temp):
    #        if item in train_item_pool:
    #            test_user_indices.append(user)
    #            test_item_indices.append(item)
    #            test_labels.append(label)

        mask = np.isin(test_item_indices, train_item_pool)
        test_user_indices = np.array(test_user_indices)[mask]
        test_item_indices = np.array(test_item_indices)[mask]
        test_labels = np.array(test_labels)[mask]
        print("item after: ", len(test_item_indices))

        user_mapping = dict(zip(set(train_user_indices), np.arange(len(set(train_user_indices)))))
        item_mapping = dict(zip(set(train_item_indices), np.arange(len(set(train_item_indices)))))
        for user, item, label in zip(train_user_indices, train_item_indices, train_labels):
            self.train_user_indices.append(user_mapping[user])
            self.train_item_indices.append(item_mapping[item])
            self.train_labels.append(label)

        for test_u, test_i, test_l in zip(test_user_indices, test_item_indices, test_labels):
            self.test_user_indices.append(user_mapping[test_u])
            self.test_item_indices.append(item_mapping[test_i])
            self.test_labels.append(test_l)

        if shuffle:
            random_mask = np.random.choice(len(self.train_user_indices), len(self.train_user_indices), replace=False)
            self.train_user_indices = np.array(self.train_user_indices)[random_mask]
            self.train_item_indices = np.array(self.train_item_indices)[random_mask]
            self.train_labels = np.array(self.train_labels)[random_mask]
        else:
            self.train_user_indices = np.array(self.train_user_indices)
            self.train_item_indices = np.array(self.train_item_indices)
            self.train_labels = np.array(self.train_labels)

        self.test_user_indices = np.array(self.test_user_indices)
        self.test_item_indices = np.array(self.test_item_indices)
        self.test_labels = np.array(self.test_labels)

        for u, i, r in zip(self.train_user_indices, self.train_item_indices, self.train_labels):
            self.train_user[u].update(dict(zip([i], [r])))
            self.train_item[i].update(dict(zip([u], [r])))

        if convert_implicit:
            self.train_labels = np.ones(len(self.train_labels), dtype=np.float32)
            self.test_labels = np.ones(len(self.test_labels), dtype=np.float32)

        if build_negative:
            self.build_trainset_implicit(num_neg)
            self.build_testset_implicit(num_neg)

        return self


#   TODO
#   def load_pandas

    def build_trainset_implicit(self, num_neg):
        neg = NegativeSampling(self, num_neg, self.batch_size, replacement_sampling=True)
        self.train_user_implicit, \
        self.train_item_implicit, \
        self.train_label_implicit = neg(mode="train")

    def build_testset_implicit(self, num_neg):
        neg = NegativeSampling(self, num_neg, self.batch_size, replacement_sampling=True)
        self.test_user_implicit, \
        self.test_item_implicit, \
        self.test_label_implicit = neg(mode="test")

    # TODO
    def build_tf_sparse(self):
        pass


    def load_tf_trainset(self, batch_size=1):
        trainset_tf = tf.data.Dataset.from_tensor_slices({'user': self.train_user_indices,
                                                          'item': self.train_item_indices,
                                                          'label': self.train_labels})
        self.trainset_tf = trainset_tf.shuffle(len(self.train_labels))  #  .batch(batch_size)
        return self

    def load_tf_testset(self):
        testset_tf = tf.data.Dataset.from_tensor_slices({'user': self.test_user_indices,
                                                         'item': self.test_item_indices,
                                                         'label': self.test_labels})
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
        return np.mean(self.train_labels)

    @property
    def n_users(self):
        return len(self.train_user)
    #    return len(np.unique(self.train_user_indices))

    @property
    def n_items(self):
        return len(self.train_item)
    #    return len(np.unique(self.train_item_indices))