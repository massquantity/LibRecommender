import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from .preprocessing import FeatureBuilder
from ..utils.sampling import negative_sampling


class Dataset:
    def __init__(self, include_features=False):
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
        self.include_features = include_features
        if self.include_features:
            self.train_categorical_features = defaultdict(list)
            self.train_numerical_features = defaultdict(list)
            self.test_categorical_features = defaultdict(list)
            self.test_numerical_features = defaultdict(list)
            self.train_mergecat_features = defaultdict(list)
            self.test_mergecat_features = defaultdict(list)

    def _get_pool(self, data_path="../ml-1m/ratings.dat", shuffle=True, length="all",
                    train_frac=0.8, sep=",", user_col=None, item_col=None, seed=42):
        np.random.seed(seed)
        user_pool = set()
        item_pool = set()
        loaded_data = open(data_path, 'r').readlines()
        if shuffle:
            loaded_data = np.random.permutation(loaded_data)
        if length == "all":
            length = len(loaded_data)
        for i, data in enumerate(loaded_data[:length]):
            line = data.split(sep)
            user = line[user_col]
            item = line[item_col]
            if i <= int(train_frac * length):
                user_pool.add(user)
                item_pool.add(item)
        return user_pool, item_pool, loaded_data

    def build_dataset(self, data_path="../ml-1m/ratings.dat", shuffle=True, length="all",
                      train_frac=0.8, implicit_label=False, build_negative=False, seed=42,
                      num_neg=None, sep=",", user_col=None, item_col=None, label_col=None,
                      numerical_col=None, categorical_col=None, merged_categorical_col=None):  # numerical feature 不做 embedding
        user_pool, item_pool, loaded_data = self._get_pool(data_path=data_path,
                                                             shuffle=shuffle,
                                                             length=length,
                                                             train_frac=train_frac,
                                                             sep=sep,
                                                             user_col=user_col,
                                                             item_col=item_col,
                                                             seed=seed)

        index_user = 0
        index_item = 0
        if length == "all":
            length = len(loaded_data)
        for i, data in enumerate(loaded_data[:length]):
            line = data.split(sep)
            user = line[user_col]
            item = line[item_col]
            label = line[label_col]

            if user not in user_pool or item not in item_pool:
                continue

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
                self.train_labels.append(int(label))
                self.train_user[user_id].update(dict(zip([item_id], [int(label)])))
                self.train_item[item_id].update(dict(zip([user_id], [int(label)])))

                if categorical_col is not None and self.include_features:
                    for cat_feat in categorical_col:
                        self.train_categorical_features[cat_feat].append(line[cat_feat].strip())

                if numerical_col is not None and self.include_features:
                    for num_feat in numerical_col:
                        self.train_numerical_features[num_feat].append(line[num_feat].strip())

                if merged_categorical_col is not None and self.include_features:
                    for merge_feat in merged_categorical_col:
                        merge_col_index = merge_feat[0]
                        for mft in merge_feat:
                            self.train_mergecat_features[merge_col_index].extend([line[mft].strip()])

            else:
                self.test_user_indices.append(user_id)
                self.test_item_indices.append(item_id)
                self.test_labels.append(int(label))

                if categorical_col is not None and self.include_features:
                    for cat_feat in categorical_col:
                        self.test_categorical_features[cat_feat].append(line[cat_feat].strip())

                if numerical_col is not None and self.include_features:
                    for num_feat in numerical_col:
                        self.test_numerical_features[num_feat].append(line[num_feat].strip())

                if merged_categorical_col is not None and self.include_features:
                    for merge_feat in merged_categorical_col:
                        merge_col_index = merge_feat[0]
                        for mft in merge_feat:
                            self.test_mergecat_features[merge_col_index].extend([line[mft].strip()])

        self.train_user_indices = np.array(self.train_user_indices)
        self.train_item_indices = np.array(self.train_item_indices)
        self.train_labels = np.array(self.train_labels)
        if self.include_features:
            fb = FeatureBuilder(include_user_item=True, n_users=self.n_users, n_items=self.n_items)
            self.train_feat_indices, self.train_feat_values, self.feature_size = \
                fb.fit(self.train_categorical_features,
                       self.train_numerical_features,
                       self.train_mergecat_features,
                       len(self.train_labels),
                       self.train_user_indices,
                       self.train_item_indices)

        # user_embedding, item_embedding, feature_embedding
        # np.unique(return_inverse=True)
        # numerical min_max_scale
        # min_occurance

        if implicit_label:
            self.train_labels = np.ones(len(self.train_labels), dtype=np.float32)

        if build_negative:
            self.build_trainset_implicit(num_neg)

        print("testset size before: ", len(self.test_labels))
    #    test_all = np.concatenate([np.expand_dims(self.test_user_indices, 1),
    #                               np.expand_dims(self.test_item_indices, 1),
    #                               np.expand_dims(self.test_labels, 1)],
    #                               axis=1)
    #    test_safe = test_all[(test_all[:, 0] < self.n_users) & (test_all[:, 1] < self.n_items)]
    #    test_danger = test_all[(test_all[:, 0] >= self.n_users) & (test_all[:, 1] >= self.n_items)]
    #    self.test_user_indices = test_safe[:, 0]
    #    self.test_item_indices = test_safe[:, 1]
    #    self.test_labels = test_safe[:, 2]

        self.test_user_indices = np.array(self.test_user_indices)
        self.test_item_indices = np.array(self.test_item_indices)
        self.test_labels = np.array(self.test_labels)
        if self.include_features:
            self.test_feat_indices, self.test_feat_values = \
                fb.transform(self.test_categorical_features,
                             self.test_numerical_features,
                             self.test_mergecat_features,
                             len(self.test_labels),
                             self.test_user_indices,
                             self.test_item_indices)

        if implicit_label:
            self.test_labels = np.ones(len(self.test_labels), dtype=np.float32)

        if build_negative:
            self.build_testset_implicit(num_neg)
    #        self.neg = negative_sampling(self, 4, self.batch_size)
        #    self.build_trainset_implicit()
        #    self.build_testset_implicit()
        print("testset size after: ", len(self.test_labels))
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
                                                          'label': self.train_labels})
        self.trainset_tf = trainset_tf.shuffle(len(self.train_labels)).batch(batch_size)
        return self

    def load_tf_testset(self):
        testset_tf = tf.data.Dataset.from_tensor_slices({'user': self.test_user_indices,
                                                         'item': self.test_item_indices,
                                                         'label': self.test_labels})
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
        return np.mean(self.train_labels)

    @property
    def n_users(self):
        return len(self.train_user)

    @property
    def n_items(self):
        return len(self.train_item)