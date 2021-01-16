from math import floor
from random import random, seed as set_random_seed
import numpy as np
from tqdm import tqdm
from ..utils.misc import time_block


class SamplingBase(object):
    def __init__(self, dataset, data_info, num_neg=1):
        self.dataset = dataset
        self.data_info = data_info
        self.num_neg = num_neg

    def sample_items_random(self, seed=42):
        set_random_seed(seed)
        n_items = self.data_info.n_items
        item_indices_sampled = list()
        # set is much faster for search contains
        user_consumed = {
            u: set(items) for u, items in self.data_info.user_consumed.items()
        }
        # sample negative items for every user
        with time_block("random neg item sampling"):
            for u, i in zip(self.dataset.user_indices,
                            self.dataset.item_indices):
                item_indices_sampled.append(i)
                for _ in range(self.num_neg):
                    item_neg = floor(n_items * random())
                    if u in user_consumed:
                        while item_neg in user_consumed[u]:
                            item_neg = floor(n_items * random())
                    item_indices_sampled.append(item_neg)
        return np.asarray(item_indices_sampled)

    def sample_items_popular(self, seed=42):
        data = self.data_info.get_indexed_interaction()
        item_counts = data.item.value_counts().sort_index().to_numpy()
        user_consumed = self.data_info.user_consumed
        items = np.arange(self.data_info.n_items)

        item_order = list()
        item_indices_sampled = list()
        with time_block("popularity-based neg item sampling"):
            for user, u_data in data.groupby("user", sort=False):
                item_indices = u_data.index.to_list()
                item_indices = item_indices * (self.num_neg + 1)
                item_order.extend(item_indices)

                # add positive items
                item_indices_sampled.extend(u_data.item.tolist())
                u_consumed = user_consumed[user]
                u_item_counts = item_counts.copy()
                u_item_counts[u_consumed] = 0
                item_prob = u_item_counts / np.sum(u_item_counts)
                neg_size = len(u_consumed) * self.num_neg

                neg_sampled = np.random.choice(
                    items, size=neg_size, p=item_prob, replace=True)
                item_indices_sampled.extend(neg_sampled)

        item_indices_sampled = np.asarray(item_indices_sampled)
        # must be stable sort to keep relative order
        item_order = np.argsort(item_order, kind="mergesort")
        return item_indices_sampled[item_order]

    def _label_negative_sampling(self, size):
        factor = self.num_neg + 1
        total_length = size * factor
        labels = np.zeros(total_length, dtype=np.float32)
        labels[::factor] = 1.0
        return labels


class NegativeSampling(SamplingBase):
    def __init__(self, dataset, data_info, num_neg, sparse=None, dense=None,
                 batch_sampling=False):
        super(NegativeSampling, self).__init__(dataset, data_info, num_neg)

        if batch_sampling and dataset.has_sampled:
            self.user_indices = dataset.user_indices_orig
            self.item_indices = dataset.item_indices_orig
            self.sparse_indices = (
                dataset.sparse_indices_orig if sparse else None)
            self.dense_values = (
                dataset.dense_values_orig if dense else None)
        else:
            self.user_indices = dataset.user_indices
            self.item_indices = dataset.item_indices
            self.sparse_indices = dataset.sparse_indices if sparse else None
            self.dense_values = dataset.dense_values if dense else None
        self.data_size = len(self.user_indices)
        self.sparse = sparse
        self.dense = dense

    def generate_all(self, seed=42, item_gen_mode="random"):
        user_indices_sampled = np.repeat(
            self.user_indices, self.num_neg + 1, axis=0
        )

        if item_gen_mode not in ["random", "popular"]:
            raise ValueError(
                "sampling item_gen_mode must either be 'random' or 'popular'"
            )
        elif item_gen_mode == "random":
            item_indices_sampled = self.sample_items_random(seed=seed)
        elif item_gen_mode == "popular":
            item_indices_sampled = self.sample_items_popular(seed=seed)

        sparse_indices_sampled = self._sparse_indices_sampling(
            self.sparse_indices, item_indices_sampled
        ) if self.sparse else None
        dense_values_sampled = self._dense_values_sampling(
            self.dense_values, item_indices_sampled
        ) if self.dense else None
        label_sampled = self._label_negative_sampling(self.data_size)

        return (
            user_indices_sampled,
            item_indices_sampled,
            label_sampled,
            sparse_indices_sampled,
            dense_values_sampled
        )

    def __call__(self, shuffle=True, batch_size=None):
        if shuffle:
            mask = np.random.permutation(range(self.data_size))
            self.sparse_indices = (
                self.sparse_indices[mask] if self.sparse else None)
            self.dense_values = (
                self.dense_values[mask] if self.dense else None)

        user_consumed = {
            u: set(items) for u, items in self.data_info.user_consumed.items()
        }
        n_items = self.data_info.n_items
        return self.sample_batch(user_consumed, n_items, batch_size)

    def sample_batch(self, user_consumed, n_items, batch_size):
        for k in tqdm(range(0, self.data_size, batch_size),
                      desc="batch_sampling train"):
            batch_slice = slice(k, k + batch_size)
            batch_user_indices = self.user_indices[batch_slice]
            batch_item_indices = self.item_indices[batch_slice]
            batch_sparse_indices = (
                self.sparse_indices[batch_slice] if self.sparse else None)
            batch_dense_values = (
                self.dense_values[batch_slice] if self.dense else None)

            user_indices_sampled = np.repeat(
                batch_user_indices, self.num_neg + 1, axis=0
            )

            item_indices_sampled = list()
            for u, i in zip(batch_user_indices, batch_item_indices):
                item_indices_sampled.append(i)
                for _ in range(self.num_neg):
                    item_neg = floor(random() * n_items)
                    while item_neg in user_consumed[u]:
                        item_neg = floor(random() * n_items)
                    item_indices_sampled.append(item_neg)
            item_indices_sampled = np.array(item_indices_sampled)

            sparse_indices_sampled = self._sparse_indices_sampling(
                batch_sparse_indices, item_indices_sampled
            ) if self.sparse else None
            dense_values_sampled = self._dense_values_sampling(
                batch_dense_values, item_indices_sampled
            ) if self.dense else None
            label_sampled = self._label_negative_sampling(
                len(batch_user_indices)
            )

            yield (
                user_indices_sampled,
                item_indices_sampled,
                label_sampled,
                sparse_indices_sampled,
                dense_values_sampled
            )

    def _sparse_indices_sampling(self, sparse_indices, item_indices_sampled):
        user_sparse_col = self.data_info.user_sparse_col.index
        item_sparse_col = self.data_info.item_sparse_col.index

        if user_sparse_col and item_sparse_col:
            user_sparse_indices = np.take(
                sparse_indices, user_sparse_col, axis=1)
            user_sparse_sampled = np.repeat(
                user_sparse_indices, self.num_neg + 1, axis=0)
            item_sparse_sampled = self.data_info.item_sparse_unique[
                item_indices_sampled]

            assert len(user_sparse_sampled) == len(item_sparse_sampled), (
                "num of user sampled must equal to num of item sampled")
            # keep column names in original order
            orig_cols = user_sparse_col + item_sparse_col
            col_reindex = np.arange(len(orig_cols))[np.argsort(orig_cols)]
            return np.concatenate(
                [user_sparse_sampled, item_sparse_sampled], axis=-1
            )[:, col_reindex]

        elif user_sparse_col:
            user_sparse_indices = np.take(
                sparse_indices, user_sparse_col, axis=1)
            user_sparse_sampled = np.repeat(
                user_sparse_indices, self.num_neg + 1, axis=0)
            return user_sparse_sampled

        elif item_sparse_col:
            item_sparse_sampled = self.data_info.item_sparse_unique[
                item_indices_sampled]
            return item_sparse_sampled

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
            user_dense_sampled = np.repeat(
                user_dense_values, self.num_neg + 1, axis=0)
            item_dense_sampled = self.data_info.item_dense_unique[
                item_indices_sampled]

            assert len(user_dense_sampled) == len(item_dense_sampled), (
                "num of user sampled must equal to num of item sampled")
            # keep column names in original order
            orig_cols = user_dense_col + item_dense_col
            col_reindex = np.arange(len(orig_cols))[np.argsort(orig_cols)]
            return np.concatenate(
                [user_dense_sampled, item_dense_sampled], axis=-1
            )[:, col_reindex]

        elif user_dense_col:
            user_dense_values = np.take(dense_values, user_dense_col, axis=1)
            user_dense_sampled = np.repeat(
                user_dense_values, self.num_neg + 1, axis=0)
            return user_dense_sampled

        elif item_dense_col:
            item_dense_sampled = self.data_info.item_dense_unique[
                item_indices_sampled]
            return item_dense_sampled


class PairwiseSampling(SamplingBase):
    def __init__(self, dataset, data_info, num_neg=1):
        super(PairwiseSampling, self).__init__(dataset, data_info, num_neg)

        if dataset.has_sampled:
            self.user_indices = dataset.user_indices_orig
            self.item_indices = dataset.item_indices_orig
        else:
            self.user_indices = dataset.user_indices
            self.item_indices = dataset.item_indices
        self.data_size = len(self.user_indices)

    def __call__(self, shuffle=True, batch_size=None):
        if shuffle:
            mask = np.random.permutation(range(self.data_size))
            self.user_indices = self.user_indices[mask]
            self.item_indices = self.item_indices[mask]

        user_consumed_set = {
            u: set(items) for u, items in self.data_info.user_consumed.items()
        }
        n_items = self.data_info.n_items
        return self.sample_batch(user_consumed_set, n_items, batch_size)

    def sample_batch(self, user_consumed_set, n_items, batch_size):
        for k in tqdm(range(0, self.data_size, batch_size),
                      desc="pair_sampling train"):
            batch_slice = slice(k, k + batch_size)
            batch_user_indices = self.user_indices[batch_slice]
            batch_item_indices_pos = self.item_indices[batch_slice]

            batch_item_indices_neg = list()
            for u in batch_user_indices:
                item_neg = floor(n_items * random())
                while item_neg in user_consumed_set[u]:
                    item_neg = floor(n_items * random())
                batch_item_indices_neg.append(item_neg)

            batch_item_indices_neg = np.asarray(batch_item_indices_neg)
            yield (
                batch_user_indices,
                batch_item_indices_pos,
                batch_item_indices_neg
            )


class PairwiseSamplingSeq(PairwiseSampling):
    def __init__(self, dataset, data_info, num_neg=1, mode=None, num=None):
        super(PairwiseSamplingSeq, self).__init__(dataset, data_info, num_neg)

        self.seq_mode = mode
        self.seq_num = num
        self.n_items = data_info.n_items
        self.user_consumed = data_info.user_consumed

    def sample_batch(self, user_consumed_set, n_items, batch_size):
        # avoid circular import
        from ..data.sequence import user_interacted_seq

        for k in tqdm(range(0, self.data_size, batch_size),
                      desc="pair_sampling sequence train"):
            batch_slice = slice(k, k + batch_size)
            batch_user_indices = self.user_indices[batch_slice]
            batch_item_indices_pos = self.item_indices[batch_slice]

            (
                batch_interacted,
                batch_interacted_len
            ) = user_interacted_seq(
                batch_user_indices,
                batch_item_indices_pos,
                self.user_consumed,
                self.n_items,
                self.seq_mode,
                self.seq_num,
                user_consumed_set
            )

            batch_item_indices_neg = list()
            for u in batch_user_indices:
                item_neg = floor(n_items * random())
                while item_neg in user_consumed_set[u]:
                    item_neg = floor(n_items * random())
                batch_item_indices_neg.append(item_neg)

            batch_item_indices_neg = np.asarray(batch_item_indices_neg)
            yield (
                batch_user_indices,
                batch_item_indices_pos,
                batch_item_indices_neg,
                batch_interacted,
                batch_interacted_len
            )
