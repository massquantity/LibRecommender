from collections import defaultdict
import numpy as np
import tensorflow as tf
from .download_data import prepare_data
from ..utils.sampling import NegativeSampling
import warnings
warnings.filterwarnings("ignore")

class DatasetPure:
    def __init__(self, load_builtin_data="ml-1m", par_path=None):
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
        if load_builtin_data == "ml-1m":
            prepare_data(par_path, feat=False)

    def build_dataset(self, data_path="../ml-1m/ratings.dat", shuffle=True, length="all", sep=",",
                      user_col=None, item_col=None, label_col=None, split_mode="train_test", threshold=0,
                      train_frac=0.8, convert_implicit=False, build_negative=False, build_tf_dataset=False,
                      k=1, batch_size=256, seed=42, num_neg=None, lower_upper_bound=None):
        np.random.seed(seed)
        self.batch_size = batch_size
        self.lower_upper_bound = lower_upper_bound
        if isinstance(num_neg, int) and num_neg > 0:
            self.num_neg = num_neg
        if not np.all([user_col, item_col, label_col]):
            self.user_col = 0
            self.item_col = 1
            self.label_col = 2

        with open(data_path, 'r') as f:
            loaded_data = f.readlines()
        if shuffle:
            loaded_data = np.random.permutation(loaded_data)
        if length == "all":
            length = len(loaded_data)
        if split_mode == "train_test":
            self.train_test_split(loaded_data, length, sep, train_frac, convert_implicit,
                                  build_negative, build_tf_dataset, threshold, num_neg)
        elif split_mode == "leave_k_out":
            self.leave_k_out_split(k, loaded_data, length, sep, convert_implicit, shuffle,
                                   build_negative, threshold, num_neg)
        else:
            raise ValueError("split_mode must be either 'train_test' or 'leave_k_out'")

    def train_test_split(self, loaded_data, length, sep=",", train_frac=0.8, convert_implicit=False,
                         build_negative=False, build_tf_dataset=False, threshold=0, num_neg=None):
        index_user = 0
        index_item = 0
        for i, line in enumerate(loaded_data[:length]):
            user = line.split(sep)[self.user_col]
            item = line.split(sep)[self.item_col]
            label = line.split(sep)[self.label_col]
            if convert_implicit and int(label) > threshold:
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

        if build_negative:
            self.build_trainset_implicit(num_neg)
            self.build_testset_implicit(num_neg)

        if build_tf_dataset:
            self.load_tf_trainset()
            self.load_tf_testset()

        print("testset size after: ", len(self.test_labels))
        return self

    def leave_k_out_split(self, k, loaded_data, length, sep=",", convert_implicit=False,
                          shuffle=True, build_negative=False, threshold=0, num_neg=None):
        """
        leave-last-k-out-split : split k test sample from each user
        :return: dataset
        """
        user_indices = list()
        item_indices = list()
        labels = list()
        train_user_indices = list()
        train_item_indices = list()
        train_labels = list()
        test_user_indices = list()
        test_item_indices = list()
        test_labels = list()
        user2id = dict()
        item2id = dict()
        index_user = 0
        index_item = 0
        for i, line in enumerate(loaded_data[:length]):
            user = line.split(sep)[self.user_col]
            item = line.split(sep)[self.item_col]
            label = line.split(sep)[self.label_col]
            if convert_implicit and int(label) > threshold:
                label = 1
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

            user_indices.append(user_id)
            item_indices.append(item_id)
            labels.append(float(label))

        user_indices = np.array(user_indices)
        item_indices = np.array(item_indices)
        labels = np.array(labels)

        users, user_position, user_counts = np.unique(user_indices,
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

            train_user_indices.extend(user_indices[train_indices])
            train_item_indices.extend(item_indices[train_indices])
            train_labels.extend(labels[train_indices])

            test_user_indices.extend(user_indices[test_indices])
            test_item_indices.extend(item_indices[test_indices])
            test_labels.extend(labels[test_indices])

        print("testset size before: ", len(test_item_indices))
        train_item_pool = np.unique(train_item_indices)
        mask = np.isin(test_item_indices, train_item_pool)  # remove items in test data that are not in train data
        test_user_indices = np.array(test_user_indices)[mask]
        test_item_indices = np.array(test_item_indices)[mask]
        test_labels = np.array(test_labels)[mask]
        print("testset size after: ", len(test_item_indices))

        self.user2id = dict(zip(set(train_user_indices), np.arange(len(set(train_user_indices)))))
        self.item2id = dict(zip(set(train_item_indices), np.arange(len(set(train_item_indices)))))
        for user, item, label in zip(train_user_indices, train_item_indices, train_labels):
            self.train_user_indices.append(self.user2id[user])
            self.train_item_indices.append(self.item2id[item])
            self.train_labels.append(label)

        for test_u, test_i, test_l in zip(test_user_indices, test_item_indices, test_labels):
            self.test_user_indices.append(self.user2id[test_u])
            self.test_item_indices.append(self.item2id[test_i])
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

        if build_negative:
            self.build_trainset_implicit(num_neg)
            self.build_testset_implicit(num_neg)

        return self

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

    def load_tf_trainset(self):
        trainset_tf = tf.data.Dataset.from_tensor_slices({'user': self.train_user_indices,
                                                          'item': self.train_item_indices,
                                                          'label': self.train_labels})
        self.trainset_tf = trainset_tf.shuffle(len(self.train_labels))
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

    @property
    def n_items(self):
        return len(self.train_item)

