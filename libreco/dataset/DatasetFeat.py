import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from .preprocessing import FeatureBuilder
from ..utils.sampling import NegativeSampling, NegativeSamplingFeat


class DatasetFeat:
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
            self.train_merge_list = defaultdict(list)
            self.test_merge_list = defaultdict(list)

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

    def build_dataset(self, data_path="../ml-1m/ratings.dat", shuffle=True, length="all", batch_size=256,
                      train_frac=0.8, convert_implicit=False, build_negative=False, seed=42,
                      num_neg=None, sep=",", user_col=None, item_col=None, label_col=None,
                      numerical_col=None, categorical_col=None, merged_categorical_col=None,
                      item_sample_col=None):  # numerical feature 不做 embedding

        np.random.seed(seed)
        self.batch_size = batch_size
        if num_neg is not None:
            self.num_neg = num_neg

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
                    #    merge_list = [[] for _ in range(len(merge_feat))]
                        for mft in merge_feat:
                            self.train_merge_list[mft].append(line[mft].strip())
                    #    merge_col_index = merge_feat[0]
                    #    for ml in merge_list:
                    #        self.train_mergecat_features[merge_col_index].extend(ml)

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
                        for mft in merge_feat:
                            self.test_merge_list[mft].append(line[mft].strip())

        if merged_categorical_col is not None and self.include_features:
            for merge_feat in merged_categorical_col:
                merge_col_index = merge_feat[0]
                for mft in merge_feat:
                    self.train_mergecat_features[merge_col_index].extend(self.train_merge_list[mft])
                    self.test_mergecat_features[merge_col_index].extend(self.test_merge_list[mft])

        self.train_user_indices = np.array(self.train_user_indices)
        self.train_item_indices = np.array(self.train_item_indices)
        self.train_labels = np.array(self.train_labels)
        if self.include_features:
            self.fb = FeatureBuilder(include_user_item=True, n_users=self.n_users, n_items=self.n_items)
            self.train_feat_indices, self.train_feat_values, self.feature_size = \
                self.fb.fit(self.train_categorical_features,
                       self.train_numerical_features,
                       self.train_mergecat_features,
                       len(self.train_labels),
                       self.train_user_indices,
                       self.train_item_indices)
            self.user_offset = self.fb.total_count
            print("offset: {}, n_users: {}, feature_size: {}".format(
                self.user_offset, self.n_users, self.feature_size))
        #    print(self.train_feat_indices.shape)
        #    print(min(self.train_feat_indices[:, 6]))

        # user_embedding, item_embedding, feature_embedding
        # np.unique(return_inverse=True)
        # numerical min_max_scale
        # min_occurance

        print("testset size before: ", len(self.test_labels))
        self.test_user_indices = np.array(self.test_user_indices)
        self.test_item_indices = np.array(self.test_item_indices)
        self.test_labels = np.array(self.test_labels)
        if self.include_features:
            self.test_feat_indices, self.test_feat_values = \
                self.fb.transform(self.test_categorical_features,
                             self.test_numerical_features,
                             self.test_mergecat_features,
                             len(self.test_labels),
                             self.test_user_indices,
                             self.test_item_indices)

        if convert_implicit:
            self.train_labels = np.ones(len(self.train_labels), dtype=np.float32)
            self.test_labels = np.ones(len(self.test_labels), dtype=np.float32)
        #    self.item_sample_col = np.array(item_sample_col) - 3
            self.item_sample_col = [(i - 3) for i in item_sample_col]  # remove user item label column

        if build_negative:
            self.build_trainset_implicit(num_neg)
            self.build_testset_implicit(num_neg)

        print("testset size after: ", len(self.test_labels))
        return self

    def leave_k_out_split(self, k, data_path, length="all", sep=",", shuffle=True, user_col=None,
                         item_col=None, label_col=None, numerical_col=None, categorical_col=None,
                         merged_categorical_col=None, seed=42):
        """
        leave-last-k-out-split
        :return: train - test, user - item - ratings
        """
        np.random.seed(seed)
        self.user_indices = []
        self.item_indices = []
        self.labels = []
        self.categorical_features = defaultdict(list)
        self.numerical_features = defaultdict(list)
        self.mergecat_features = defaultdict(list)

        user2id = dict()
        item2id = dict()
        train_user_indices = list()
        train_item_indices = list()
        train_labels = list()
        train_categorical_features = defaultdict(list)
        train_numerical_features = defaultdict(list)
        train_mergecat_features = defaultdict(list)
        test_user_indices = list()
        test_item_indices = list()
        test_labels = list()
        test_categorical_features = defaultdict(list)
        test_numerical_features = defaultdict(list)
        test_mergecat_features = defaultdict(list)

        index_user = 0
        index_item = 0
        loaded_data = open(data_path, 'r').readlines()
        if length == "all":
            length = len(loaded_data)
        for i, data in enumerate(loaded_data[:length]):
            line = data.split(sep)
            user = line[user_col]
            item = line[item_col]
            label = line[label_col]

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

            if categorical_col is not None and self.include_features:
                for cat_feat in categorical_col:
                    self.categorical_features[cat_feat].append(line[cat_feat].strip())

            if numerical_col is not None and self.include_features:
                for num_feat in numerical_col:
                    self.numerical_features[num_feat].append(line[num_feat].strip())

            if merged_categorical_col is not None and self.include_features:
                for merge_feat in merged_categorical_col:
                    for mft in merge_feat:
                        self.mergecat_features[mft].append(line[mft].strip())

        self.user_indices = np.array(self.user_indices)
        self.item_indices = np.array(self.item_indices)
        self.labels = np.array(self.labels)

        users, user_position, user_counts = np.unique(self.user_indices,
                                                      return_inverse=True,
                                                      return_counts=True)
        user_split_indices = np.split(np.argsort(user_position, kind="mergesort"),
                                      np.cumsum(user_counts)[:-1])

        train_indices_all = []
        test_indices_all = []
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

            train_indices_all.extend(train_indices.tolist())
            test_indices_all.extend(test_indices.tolist())

        if categorical_col is not None and self.include_features:
            for cat_feat in categorical_col:
                train_categorical_features[cat_feat].extend(
                    np.array(self.categorical_features[cat_feat])[np.array(train_indices_all)])
        if numerical_col is not None and self.include_features:
            for num_feat in numerical_col:
                train_numerical_features[num_feat].extend(
                    np.array(self.numerical_features[num_feat])[np.array(train_indices_all)])
        if merged_categorical_col is not None and self.include_features:
            for merge_feat in merged_categorical_col:
                for mft in merge_feat:
                    train_mergecat_features[mft].extend(
                        np.array(self.mergecat_features[mft])[np.array(train_indices_all)])


        if categorical_col is not None and self.include_features:
            for cat_feat in categorical_col:
                test_categorical_features[cat_feat].extend(
                    np.array(self.categorical_features[cat_feat])[np.array(test_indices_all)])
        if numerical_col is not None and self.include_features:
            for num_feat in numerical_col:
                test_numerical_features[num_feat].extend(
                    np.array(self.numerical_features[num_feat])[np.array(test_indices_all)])
        if merged_categorical_col is not None and self.include_features:
            for merge_feat in merged_categorical_col:
                for mft in merge_feat:
                    test_mergecat_features[mft].extend(
                        np.array(self.mergecat_features[mft])[np.array(test_indices_all)])

        print("item before: ", len(test_item_indices))
        train_item_pool = np.unique(train_item_indices)   # remove items in test data that are not in train data
        mask = np.isin(test_item_indices, train_item_pool)
        test_user_indices = np.array(test_user_indices)[mask]
        test_item_indices = np.array(test_item_indices)[mask]
        test_labels = np.array(test_labels)[mask]
        if categorical_col is not None and self.include_features:
            for cat_feat in categorical_col:
                test_categorical_features[cat_feat] = np.array(test_categorical_features[cat_feat])[mask]
        if numerical_col is not None and self.include_features:
            for num_feat in numerical_col:
                test_numerical_features[num_feat] = np.array(test_numerical_features[num_feat])[mask]
        if merged_categorical_col is not None and self.include_features:
            for merge_feat in merged_categorical_col:
                for mft in merge_feat:
                    test_mergecat_features[mft] = np.array(test_mergecat_features[mft])[mask]
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
            if categorical_col is not None and self.include_features:
                for cat_feat in categorical_col:
                    self.train_categorical_features[cat_feat] = \
                        np.array(train_categorical_features[cat_feat])[random_mask]
            if numerical_col is not None and self.include_features:
                for num_feat in numerical_col:
                    self.train_numerical_features[num_feat] = \
                        np.array(train_numerical_features[num_feat])[random_mask]
            if merged_categorical_col is not None and self.include_features:
                for merge_feat in merged_categorical_col:
                    merge_col_index = merge_feat[0]
                    for mft in merge_feat:
                        self.train_mergecat_features[merge_col_index] = \
                            np.array(train_mergecat_features[mft])[random_mask]

        else:
            self.train_user_indices = np.array(self.train_user_indices)
            self.train_item_indices = np.array(self.train_item_indices)
            self.train_labels = np.array(self.train_labels)
            if categorical_col is not None and self.include_features:
                for cat_feat in categorical_col:
                    self.train_categorical_features[cat_feat] = np.array(train_categorical_features[cat_feat])
            if numerical_col is not None and self.include_features:
                for num_feat in numerical_col:
                    self.train_numerical_features[num_feat] = np.array(train_numerical_features[num_feat])
            if merged_categorical_col is not None and self.include_features:
                for merge_feat in merged_categorical_col:
                    merge_col_index = merge_feat[0]
                    for mft in merge_feat:
                        self.train_mergecat_features[merge_col_index] = np.array(train_mergecat_features[mft])

        if self.include_features:
            fb = FeatureBuilder(include_user_item=True, n_users=self.n_users, n_items=self.n_items)
            self.train_feat_indices, self.train_feat_values, self.feature_size = \
                fb.fit(self.train_categorical_features,
                       self.train_numerical_features,
                       self.train_mergecat_features,
                       len(self.train_labels),
                       self.train_user_indices,
                       self.train_item_indices)

        self.test_user_indices = np.array(self.test_user_indices)
        self.test_item_indices = np.array(self.test_item_indices)
        self.test_labels = np.array(self.test_labels)
        if categorical_col is not None and self.include_features:
            for cat_feat in categorical_col:
                self.test_categorical_features[cat_feat] = np.array(test_categorical_features[cat_feat])
        if numerical_col is not None and self.include_features:
            for num_feat in numerical_col:
                self.test_numerical_features[num_feat] = np.array(test_numerical_features[num_feat])
        if merged_categorical_col is not None and self.include_features:
            for merge_feat in merged_categorical_col:
                merge_col_index = merge_feat[0]
                for mft in merge_feat:
                    self.test_mergecat_features[merge_col_index] = np.array(test_mergecat_features[mft])

        print("testset size before: ", len(self.test_labels))
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
        print("testset size after: ", len(self.test_labels))

        for u, i, r in zip(self.train_user_indices, self.train_item_indices, self.train_labels):
            self.train_user[u].update(dict(zip([i], [r])))
            self.train_item[i].update(dict(zip([u], [r])))
        return self

#   TODO
#   def load_pandas

    def build_trainset_implicit(self, num_neg):
        neg = NegativeSamplingFeat(self, num_neg, self.batch_size, replacement_sampling=True)
        self.train_indices_implicit, \
        self.train_values_implicit, \
        self.train_labels_implicit = neg(mode="train")

    def build_testset_implicit(self, num_neg):
        neg = NegativeSamplingFeat(self, num_neg, self.batch_size, replacement_sampling=True)
        self.test_indices_implicit, \
        self.test_values_implicit, \
        self.test_labels_implicit = neg(mode="test")

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