import time
import abc
from math import floor
from random import random, seed as set_random_seed
import numpy as np
from .timing import time_block, time_func


class SamplingBase(object):
    def __init__(self, dataset, data_info, num_neg=1, batch_size=64):
        self.dataset = dataset
    #    self.data_size = len(dataset)
        self.data_info = data_info
        self.num_neg = num_neg
        self.batch_size = batch_size
        
    @abc.abstractmethod
    def generate_all(self, **kwargs):
        raise NotImplementedError

    def sample_items_random(self, seed=42):
        set_random_seed(seed)
        np.random.seed(seed)
        n_items = self.data_info.n_items
        item_indices_sampled = list()
        item_append = item_indices_sampled.append  # use local variable to speed up
        # set is much faster for search contains
        user_consumed = {
            u: set(items) for u, items in self.dataset.user_consumed.items()
        }
        # sample negative items for every user
        with time_block("neg"):
            for u, i in zip(self.dataset.user_indices, self.dataset.item_indices):
                item_append(i)
                for _ in range(self.num_neg):
                    item_neg = floor(random() * n_items)
                    while item_neg in user_consumed[u]:
                        item_neg = floor(random() * n_items)
                    item_append(item_neg)
        return np.array(item_indices_sampled)

    def sample_items_popular(self, seed=42):
        set_random_seed(seed)
        np.random.seed(seed)
        data = self.data_info.get_indexed_interaction()
        item_counts = data.item.value_counts().sort_index().to_numpy()
        user_consumed = self.dataset.user_consumed
        items = np.arange(self.data_info.n_items)

        item_order = list()
        item_indices_sampled = list()
        with time_block("neg popular"):
            for user, u_data in data.groupby("user", sort=False):
                item_indices = u_data.index.to_list()
                item_indices = item_indices * (self.num_neg + 1)
                item_order.extend(item_indices)

                item_indices_sampled.extend(u_data.item.tolist())  # add positive items
                u_consumed = user_consumed[user]
                u_item_counts = item_counts.copy()
                u_item_counts[u_consumed] = 0
                item_prob = u_item_counts / np.sum(u_item_counts)
                neg_size = len(u_consumed) * self.num_neg
                neg_sampled = np.random.choice(
                    items, size=neg_size, p=item_prob, replace=True)
                item_indices_sampled.extend(neg_sampled)

        item_indices_sampled = np.asarray(item_indices_sampled)
        item_order = np.argsort(item_order, kind="mergesort")  # must be stable sort to keep order
        return item_indices_sampled[item_order]

    def _label_negative_sampling(self, size):
        factor = self.num_neg + 1
        total_length = size * factor
        labels = np.zeros(total_length, dtype=np.float32)
        labels[::factor] = 1.0
        return labels


class NegativeSamplingPure(SamplingBase):
    def __init__(self, dataset, data_info, num_neg=1, batch_size=64, batch_sampling=False):
        super(NegativeSamplingPure, self).__init__(dataset, data_info, num_neg, batch_size)
        if batch_sampling:
            self.user_indices = dataset.user_indices_orig
            self.item_indices = dataset.item_indices_orig
        else:
            self.user_indices = dataset.user_indices
            self.item_indices = dataset.item_indices
        self.data_size = len(self.user_indices)

    def generate_all(self, seed=42, shuffle=False, mode="random"):
        if mode not in ["random", "popular"]:
            raise ValueError("sampling mode must either be 'random' or 'popular'")
        elif mode == "random":
            item_indices_sampled = self.sample_items_random(seed=seed)
        elif mode == "popular":
            item_indices_sampled = self.sample_items_popular(seed=seed)

        user_indices_sampled = np.repeat(self.user_indices, self.num_neg + 1, axis=0)
        label_sampled = self._label_negative_sampling(self.data_size)
        if shuffle:
            mask = np.random.permutation(range(len(user_indices_sampled)))
            return (user_indices_sampled[mask],
                    item_indices_sampled[mask],
                    label_sampled[mask])
        else:
            return user_indices_sampled, item_indices_sampled, label_sampled
        
    def __call__(self, shuffle=True):
        if shuffle:
            mask = np.random.permutation(range(self.data_size))
            self.user_indices = self.user_indices[mask]
            self.item_indices = self.item_indices[mask]
        #    self.labels = self.labels[mask]

        user_consumed = {
            u: set(items) for u, items in self.dataset.user_consumed.items()
        }
        n_items = self.data_info.n_items
        return self.sample_batch(user_consumed, n_items)

    def sample_batch(self, user_consumed, n_items):
        for k in range(0, self.data_size, self.batch_size):
            batch_slice = slice(k, k + self.batch_size)
            batch_user_indices = self.user_indices[batch_slice]
            batch_item_indices = self.item_indices[batch_slice]

            item_indices_sampled = list()
            for u, i in zip(batch_user_indices, batch_item_indices):
                item_indices_sampled.append(i)
                for _ in range(self.num_neg):
                    item_neg = floor(random() * n_items)
                    while item_neg in user_consumed[u]:
                        item_neg = floor(random() * n_items)
                    item_indices_sampled.append(item_neg)

            item_sampled = np.asarray(item_indices_sampled)
            user_sampled = np.repeat(batch_user_indices, self.num_neg + 1, axis=0)
            label_sampled = self._label_negative_sampling(len(batch_user_indices))

            yield user_sampled, item_sampled, label_sampled


class NegativeSamplingFeat(SamplingBase):
    def __init__(self, dataset, data_info, num_neg, batch_size=64):
        super(NegativeSamplingFeat, self).__init__(dataset, data_info, num_neg, batch_size)
        self.sparse_indices = dataset.sparse_indices
        self.dense_values = dataset.dense_values
    #    self.labels = dataset.labels
        self.data_size = len(self.sparse_indices)

    def generate_all(self, seed=42, dense=True, shuffle=False, mode="random"):
        if mode not in ["random", "popular"]:
            raise ValueError("sampling mode must either be 'random' or 'popular'")
        elif mode == "random":
            item_indices_sampled = self.sample_items_random(seed=seed)
        elif mode == "popular":
            item_indices_sampled = self.sample_items_popular(seed=seed)

        sparse_indices_sampled = self._sparse_indices_sampling(
            self.sparse_indices, item_indices_sampled)
        dense_indices_sampled = self._dense_indices_sampling(
            item_indices_sampled) if dense else None
        dense_values_sampled = self._dense_values_sampling(
            self.dense_values, item_indices_sampled) if dense else None
        label_sampled = self._label_negative_sampling(self.data_size)

        if shuffle:
            return self._shuffle_data(
                dense, sparse_indices_sampled, dense_indices_sampled,
                dense_values_sampled, label_sampled, seed)
        return (sparse_indices_sampled, dense_indices_sampled,
                dense_values_sampled, label_sampled)

    def _sparse_indices_sampling(self, sparse_indices, item_indices_sampled):
        user_sparse_col = self.data_info.user_sparse_col.index
        user_sparse_indices = np.take(sparse_indices, user_sparse_col, axis=1)
        user_sparse_sampled = np.repeat(user_sparse_indices, self.num_neg + 1, axis=0)

        item_sparse_col = self.data_info.item_sparse_col.index
        item_sparse_sampled = self.data_info.item_sparse_unique[item_indices_sampled]

        assert len(user_sparse_sampled) == len(item_sparse_sampled), (
            "num of user sampled must equal to num of item sampled")
        # keep column names in original order
        orig_cols = user_sparse_col + item_sparse_col
        col_reindex = np.arange(len(orig_cols))[np.argsort(orig_cols)]
        return np.concatenate(
            [user_sparse_sampled, item_sparse_sampled], axis=-1)[:, col_reindex]

    def _dense_indices_sampling(self, item_indices_sampled):
        n_samples = len(item_indices_sampled)
        user_dense_col = self.data_info.user_dense_col.index
        item_dense_col = self.data_info.item_dense_col.index
        total_dense_cols = len(user_dense_col) + len(item_dense_col)
        return np.tile(np.arange(total_dense_cols), [n_samples, 1])

    def _dense_values_sampling(self, dense_values, item_indices_sampled):
        user_dense_col = self.data_info.user_dense_col.index
        item_dense_col = self.data_info.item_dense_col.index

        if user_dense_col and item_dense_col:
            user_dense_values = np.take(dense_values, user_dense_col, axis=1)
            user_dense_sampled = np.repeat(user_dense_values, self.num_neg + 1, axis=0)
            item_dense_sampled = self.data_info.item_dense_unique[item_indices_sampled]

            assert len(user_dense_sampled) == len(item_dense_sampled), (
                "num of user sampled must equal to num of item sampled")
            # keep column names in original order
            orig_cols = user_dense_col + item_dense_col
            col_reindex = np.arange(len(orig_cols))[np.argsort(orig_cols)]
            return np.concatenate(
                [user_dense_sampled, item_dense_sampled], axis=-1)[:, col_reindex]

        elif user_dense_col:
            user_dense_values = np.take(dense_values, user_dense_col, axis=1)
            user_dense_sampled = np.repeat(user_dense_values, self.num_neg + 1, axis=0)
            return user_dense_sampled

        elif item_dense_col:
            item_dense_sampled = self.data_info.item_dense_unique[item_indices_sampled]
            return item_dense_sampled

    def _shuffle_data(self, dense, sparse_indices_sampled, dense_indices_sampled,
                      dense_values_sampled, label_sampled, seed=42):
        np.random.seed(seed)
        total_length = self.data_size * (self.num_neg + 1)
        random_mask = np.random.permutation(range(total_length))
        if dense:
            return (sparse_indices_sampled[random_mask],
                    dense_indices_sampled[random_mask],
                    dense_values_sampled[random_mask],
                    label_sampled[random_mask])
        else:
            return (sparse_indices_sampled[random_mask],
                    None, None, label_sampled[random_mask])









