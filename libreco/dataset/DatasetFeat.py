import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from .preprocessing import FeatureBuilder
from ..utils.sampling import NegativeSampling, NegativeSamplingFeat


class DatasetFeat:
    def __init__(self, include_features=True):
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
                      user_feature_cols=None, item_feature_cols=None, lower_upper_bound=None):  # numerical feature 不做 embedding

        np.random.seed(seed)
        self.batch_size = batch_size
        self.lower_upper_bound = lower_upper_bound
        if num_neg is not None:
            self.num_neg = num_neg

    #    if not user_col or not item_col or not label_col:
        if not np.all([user_col, item_col, label_col]):
            user_col = 0
            item_col = 1
            label_col = 2

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
            if convert_implicit and label != 0:
                label = 1

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
            self.fb = FeatureBuilder(include_user=True, include_item=True,
                                     n_users=self.n_users, n_items=self.n_items)
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
        #    self.item_feature_cols = [(i - 3) for i in item_feature_cols]  # remove user item label column

        self.numerical_col = numerical_col
        # remove user - item - label column and add numerical columns
        total_num_index = 0
        if user_feature_cols is not None:
            user_cols = []
            self.user_numerical_cols = []
            for col in user_feature_cols:
                if numerical_col is not None and col in numerical_col:
                    user_cols.append(total_num_index)
                    self.user_numerical_cols.append(total_num_index)
                    total_num_index += 1
                elif numerical_col is not None and col in categorical_col:
                    orig_col = col
                    num_place = np.searchsorted(sorted(numerical_col), orig_col)
                    col += (len(numerical_col) - num_place)
                    if orig_col > user_col:
                        col -= 1
                    if orig_col > item_col:
                        col -= 1
                    if orig_col > label_col:
                        col -= 1
                    user_cols.append(col)
                elif numerical_col is None:
                    orig_col = col
                    if orig_col > user_col:
                        col -= 1
                    if orig_col > item_col:
                        col -= 1
                    if orig_col > label_col:
                        col -= 1
                    user_cols.append(col)
            self.user_feature_cols = sorted(user_cols)

        if item_feature_cols is not None:
            item_cols = []
            self.item_numerical_cols = []
            for col in item_feature_cols:
                if numerical_col is not None and col in numerical_col:
                    item_cols.append(total_num_index)
                    self.item_numerical_cols.append(total_num_index)
                    total_num_index += 1
                elif numerical_col is not None and col in categorical_col:
                    orig_col = col
                    num_place = np.searchsorted(sorted(numerical_col), orig_col)
                    col += (len(numerical_col) - num_place)
                    if orig_col > user_col:
                        col -= 1
                    if orig_col > item_col:
                        col -= 1
                    if orig_col > label_col:
                        col -= 1
                    item_cols.append(col)
                elif numerical_col is None:
                    orig_col = col
                    if orig_col > user_col:
                        col -= 1
                    if orig_col > item_col:
                        col -= 1
                    if orig_col > label_col:
                        col -= 1
                    item_cols.append(col)
            self.item_feature_cols = sorted(item_cols)
            print("user feature cols: {}, item feature cols: {}".format(self.user_feature_cols, self.item_feature_cols))

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

    # this data input function is designed for tensorflow estimator - wide_deep model
    def load_pandas(self, data_path="../ml-1m/ratings.dat", header="infer", col_names=None, length="all",
                      train_frac=0.8, convert_implicit=False, build_negative=False, seed=42,
                      num_neg=None, sep=",", user_col=None, item_col=None, label_col=None,
                      user_feature_cols=None, item_feature_cols=None):

        np.random.seed(seed)
        if isinstance(item_feature_cols[0], str) and col_names is None:
            raise TypeError("user_col, item_col, item_feature_cols... must be integer if col_names are not provided")
        elif isinstance(item_feature_cols[0], str) and col_names is not None:
            assert isinstance(col_names[0], str), "col_names must be string"
            # extract sample column indices in col_names
            _, user_col, _ = np.intersect1d(col_names, user_col, assume_unique=True, return_indices=True)
            _, item_col, _ = np.intersect1d(col_names, item_col, assume_unique=True, return_indices=True)
            _, label_col, _ = np.intersect1d(col_names, label_col, assume_unique=True, return_indices=True)
            _, user_feature_cols, _ = np.intersect1d(col_names, user_feature_cols, assume_unique=True,
                                                     return_indices=True)
            _, item_feature_cols, _ = np.intersect1d(col_names, item_feature_cols, assume_unique=True,
                                                     return_indices=True)

            user_col = user_col[0]
            item_col = item_col[0]
            label_col = label_col[0]
            item_feature_cols.sort()
            user_feature_cols.sort()

        if length == "all":
            length = None
        loaded_data = pd.read_csv(data_path, sep=sep, header=header, names=col_names, nrows=length)
        train_data, test_data = train_test_split(loaded_data, train_size=train_frac, random_state=2019)
        train_data = train_data.to_numpy()
        test_data = test_data.to_numpy()

        print("test size before filtering: ", len(test_data))
        unique_values = dict()  # unique values of every column
        out_of_bounds_indices = set()
        for col in range(train_data.shape[1]):
            key = col_names[col]
            unique_values[key] = pd.unique(train_data[:, col])
            unique_values_set = set(unique_values[key])
            for i, t in enumerate(test_data[:, col]):
                if t not in unique_values_set:  # set is much faster for search contains
                    out_of_bounds_indices.add(i)

        mask = np.arange(len(test_data))
    #    test_data = test_data[~np.isin(mask, list(out_of_bounds_indices))]
    #    filter test values that are not in train_data
        test_data = test_data[np.invert(np.isin(mask, list(out_of_bounds_indices), assume_unique=True))]
        print("test size after filtering: ", len(test_data))

        self.user_dict = dict()  # user_dict contains unique users and their features
        total_users_col = [user_col]
        total_users_col.extend(user_feature_cols)
        train_data_items = pd.DataFrame(train_data[:, total_users_col])
        total_users_unique = train_data_items.drop_duplicates().to_numpy()
        total_users = total_users_unique[:, 0]
        for user, user_columns in zip(total_users, total_users_unique):
            self.user_dict[user] = user_columns

        self.item_dict = dict()  # item_dict contains unique items and their features
        total_items_col = [item_col]
        total_items_col.extend(item_feature_cols)
        train_data_items = pd.DataFrame(train_data[:, total_items_col])
        total_items_unique = train_data_items.drop_duplicates().to_numpy()
        total_items = total_items_unique[:, 0]
        for item, item_columns in zip(total_items, total_items_unique):
            self.item_dict[item] = item_columns

        if convert_implicit:
            train_data[:, label_col] = 1.0
            test_data[:, label_col] = 1.0

        if build_negative:   # negative sampling
            consumed_items = defaultdict(set)
            for user, item in zip(train_data[:, user_col], train_data[:, item_col]):
                consumed_items[user].add(item)

            unique_items = unique_values[col_names[item_col]]
            #   reindexing this will make random choice a lot faster
            unique_indices_items = {i: item for i, item in enumerate(unique_items)}
            train_negative_samples = []
            for s in train_data:
                sample = s.tolist()
                u = sample[user_col]
                for _ in range(num_neg):
                    item_neg = np.random.randint(0, len(unique_items))
                    while item_neg in consumed_items[u]:
                        item_neg = np.random.randint(0, len(unique_items))

                    neg_item = self.item_dict[unique_indices_items[item_neg]]
                    for col, orig_col in enumerate(total_items_col):
                        sample[orig_col] = neg_item[col]

                    sample[label_col] = 0.0
                    train_negative_samples.append(sample)

         #   train_data = np.concatenate([train_data, train_negative_samples], axis=0)
            train_data = pd.concat([pd.DataFrame(train_data), pd.DataFrame(train_negative_samples)], ignore_index=True)
            if col_names is not None:
                train_data.columns = col_names


        #    test_consumed_items = consumed_items.copy()
            for user, item in zip(test_data[:, user_col], test_data[:, item_col]):
                consumed_items[user].add(item)  # plus test consumed items

            test_negative_samples = []
            for s in test_data:
                sample = s.tolist()
                u = sample[user_col]
                for _ in range(num_neg):
                    item_neg = np.random.randint(0, len(unique_items))
                    while item_neg in consumed_items[u]:
                        item_neg = np.random.randint(0, len(unique_items))

                    neg_item = self.item_dict[unique_indices_items[item_neg]]
                    for col, orig_col in enumerate(total_items_col):
                        sample[orig_col] = neg_item[col]

                    sample[label_col] = 0.0
                    test_negative_samples.append(sample)

            test_data = pd.concat([pd.DataFrame(test_data), pd.DataFrame(test_negative_samples)], ignore_index=True)
            if col_names is not None:
                test_data.columns = col_names
            print("test size after negative sampling: {}, all data size: {}".format(
                len(test_data), len(train_data) + len(test_data)))

        else:
            train_data = pd.DataFrame(train_data)
            test_data = pd.DataFrame(test_data)
            if col_names is not None:
                train_data.columns = col_names
                test_data.columns = col_names

        self.train_data = train_data
        self.test_data = test_data
        self.feature_cols = sorted(list(set(list(train_data.columns)) - set(list([col_names[label_col]]))))
        self.user_feature_cols = np.array(col_names)[total_users_col]
        self.item_feature_cols = np.array(col_names)[total_items_col]
        self.label_cols = col_names[label_col]
        self.col_unique_values = unique_values
        self.column_types = list(loaded_data.dtypes.items())

        for col, col_type in self.column_types:
            if col_type == np.float32 or col_type == np.float64:
                self.col_unique_values[col] = self.col_unique_values[col].astype(int)
                self.train_data[col] = self.train_data[col].astype(int)
                self.test_data[col] = self.test_data[col].astype(int)
            else:
                self.col_unique_values[col] = self.col_unique_values[col].astype(col_type)
                self.train_data[col] = self.train_data[col].astype(col_type)
                self.test_data[col] = self.test_data[col].astype(col_type)

#  TODO
#   def leave_k_out_chrono_split(self):

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