import time
from math import floor
import random
import numpy as np
from .timing import time_block, time_func


class SamplingBase(object):
    def __init__(self, dataset, data_info, num_neg, batch_size=64):
        self.dataset = dataset
        self.data_info = data_info
        self.num_neg = num_neg
        self.batch_size = batch_size

    def __call__(self, seed=42, dense=True, shuffle=False, mode="random"):
        if mode not in ["random", "popular"]:
            raise ValueError("sampling mode must either be 'random' or 'popular'")
        else:
            return self.random_sampling(seed, dense, shuffle, mode)

    def _shuffle_data(self, dense, sparse_indices_sampled, dense_indices_sampled,
                      dense_values_sampled, label_sampled, seed=42):
        np.random.seed(seed)
        total_length = len(self.dataset) * (self.num_neg + 1)
        random_mask = np.random.permutation(range(total_length))
        if dense:
            return (sparse_indices_sampled[random_mask], dense_indices_sampled[random_mask],
                    dense_values_sampled[random_mask], label_sampled[random_mask])
        else:
            return sparse_indices_sampled[random_mask], None, None, label_sampled[random_mask]

    def sample_items_random(self, seed=42):
        np.random.seed(seed)
        n_items = self.data_info.n_items
        item_indices_sampled = list()
        item_append = item_indices_sampled.append  # use local variable to speed up
        # set is much faster for search contains
        user_consumed = {u: set(items) for u, items in self.dataset.user_consumed.items()}
        # sample negative items for every user
        with time_block("neg"):
            for u, i in zip(self.dataset.user_indices, self.dataset.item_indices):
                item_append(i)
                for _ in range(self.num_neg):
                    item_neg = floor(random.random() * n_items)
                    while item_neg in user_consumed[u]:
                        item_neg = floor(random.random() * n_items)
                    item_append(item_neg)
        return np.array(item_indices_sampled)

    def sample_items_popular(self, seed=42):
        np.random.seed(seed)
        data = self.data_info.get_indexed_interaction()
        item_counts = data.groupby("item")["user"].count()
        user_consumed = self.dataset.user_consumed
        items = np.arange(self.data_info.n_items)

        item_order = list()
        item_indices_sampled = list()
        t1 = time.time()
        for user, u_data in data.groupby("user", sort=False):
            item_indices = u_data.index.to_list()
            item_indices = item_indices * (self.num_neg + 1)
            item_order.extend(item_indices)

            item_indices_sampled.extend(u_data.item.tolist())  # add positive items
            u_consumed = list(user_consumed[user])
            u_item_counts = item_counts.to_numpy().copy()
            u_item_counts[u_consumed] = 0
            item_prob = u_item_counts / np.sum(u_item_counts)
            neg_size = len(u_consumed) * self.num_neg
            neg_sampled = np.random.choice(items, size=neg_size, p=item_prob, replace=True)
            item_indices_sampled.extend(neg_sampled)
        print("neg: ", time.time() - t1)

        item_indices_sampled = np.asarray(item_indices_sampled)
        item_order = np.argsort(item_order, kind="mergesort")  # must be stable sort to keep order
        return item_indices_sampled[item_order]


class NegativeSamplingFeat(SamplingBase):
    def __init__(self, dataset, data_info, num_neg, batch_size=64):
        super(NegativeSamplingFeat, self).__init__(dataset, data_info, num_neg, batch_size)

    def random_sampling(self, seed=42, dense=True, shuffle=False, mode="random"):  # "popular" may be slow
        if mode == "random":
            item_indices_sampled = self.sample_items_random(seed=seed)
        elif mode == "popular":
            item_indices_sampled = self.sample_items_popular(seed=seed)

        sparse_indices_sampled = self._sparse_indices_sampling(item_indices_sampled)
        dense_indices_sampled = self._dense_indices_sampling(item_indices_sampled) if dense else None
        dense_values_sampled = self._dense_values_sampling(item_indices_sampled) if dense else None
        label_sampled = self._label_negative_sampling()

        if shuffle:
            return self._shuffle_data(
                dense, sparse_indices_sampled, dense_indices_sampled,
                dense_values_sampled, label_sampled, seed)
        return sparse_indices_sampled, dense_indices_sampled, dense_values_sampled, label_sampled

    def _sparse_indices_sampling(self, item_indices_sampled):
        user_sparse_col = self.data_info.user_sparse_col.index
        user_sparse_indices = np.take(self.dataset.sparse_indices, user_sparse_col, axis=1)
        user_sparse_sampled = np.repeat(user_sparse_indices, self.num_neg + 1, axis=0)

        item_sparse_col = self.data_info.item_sparse_col.index
        item_sparse_sampled = self.data_info.item_sparse_unique[item_indices_sampled]

        assert len(user_sparse_sampled) == len(item_sparse_sampled), (
            "num of user sampled must equal to num of item sampled")
        # keep column names in original order
        orig_cols = user_sparse_col + item_sparse_col
        col_reindex = np.arange(len(orig_cols))[np.argsort(orig_cols)]
        return np.concatenate([user_sparse_sampled, item_sparse_sampled], axis=-1)[:, col_reindex]

    def _dense_indices_sampling(self, item_indices_sampled):
        n_samples = len(item_indices_sampled)
        user_dense_col = self.data_info.user_dense_col.index
        item_dense_col = self.data_info.item_dense_col.index
        total_dense_cols = len(user_dense_col) + len(item_dense_col)
        return np.tile(np.arange(total_dense_cols), [n_samples, 1])

    def _dense_values_sampling(self, item_indices_sampled):
        user_dense_col = self.data_info.user_dense_col.index
        item_dense_col = self.data_info.item_dense_col.index
        if user_dense_col and item_dense_col:
            user_dense_values = np.take(self.dataset.dense_values, user_dense_col, axis=1)
            user_dense_sampled = np.repeat(user_dense_values, self.num_neg + 1, axis=0)
            item_dense_sampled = self.data_info.item_dense_unique[item_indices_sampled]

            assert len(user_dense_sampled) == len(item_dense_sampled), (
                "num of user sampled must equal to num of item sampled")
            # keep column names in original order
            orig_cols = user_dense_col + item_dense_col
            col_reindex = np.arange(len(orig_cols))[np.argsort(orig_cols)]
            return np.concatenate([user_dense_sampled, item_dense_sampled], axis=-1)[:, col_reindex]
        elif user_dense_col:
            user_dense_values = np.take(self.dataset.dense_values, user_dense_col, axis=1)
            user_dense_sampled = np.repeat(user_dense_values, self.num_neg + 1, axis=0)
            return user_dense_sampled
        elif item_dense_col:
            item_dense_sampled = self.data_info.item_dense_unique[item_indices_sampled]
            return item_dense_sampled

    def _label_negative_sampling(self):
        factor = self.num_neg + 1
        total_length = len(self.dataset) * factor
        labels = np.zeros(total_length, dtype=np.float32)
        labels[::factor] = 1.0
        return labels

    def next_batch(self):
        if self.pre_sampling:
            end = min(len(self.dataset.train_indices_implicit), (self.i + 1) * self.batch_size)
            batch_feat_indices = self.dataset.train_indices_implicit[self.i * self.batch_size: end]
            batch_feat_values = self.dataset.train_values_implicit[self.i * self.batch_size: end]
            batch_labels = self.dataset.train_labels_implicit[self.i * self.batch_size: end]
            self.i += 1
            return batch_feat_indices, batch_feat_values, batch_labels
        else:
            batch_size = int(self.batch_size / (self.num_neg + 1))  # positive samples in one batch
            end = min(len(self.dataset.train_feat_indices), (self.i + 1) * batch_size)
            batch_feat_indices = self.dataset.train_feat_indices[self.i * batch_size: end]
            batch_feat_values = self.dataset.train_feat_values[self.i * batch_size: end]
            batch_feat_labels = self.dataset.train_labels[self.i * batch_size: end]

            indices, values, labels = [], [], []
            for i, sample in enumerate(batch_feat_indices):
                ss = sample.tolist()
                user = ss[-2] - self.dataset.user_offset
                indices.append(batch_feat_indices[i])
                values.append(batch_feat_values[i])
                labels.append(batch_feat_labels[i])
                for _ in range(self.num_neg):
                    item_neg = np.random.randint(0, self.dataset.n_items)
                    while item_neg in self.dataset.train_user[user]:
                        item_neg = np.random.randint(0, self.dataset.n_items)
                    item_neg += (self.dataset.user_offset + self.dataset.n_users)  # item offset
                    ss[-1] = item_neg

                    if self.item_feat:
                        dt = self.neg_indices_dict[item_neg]
                        for col, orig_col in enumerate(self.dataset.item_feature_cols):
                            ss[orig_col] = dt[col]

                    indices.append(ss)
                    vv = batch_feat_values[i]
                    if self.item_feat:
                        dv = self.neg_values_dict[item_neg]
                        for col, orig_col in enumerate(self.dataset.item_numerical_cols):
                            vv[orig_col] = dv[col]
                    values.append(vv)
                    labels.append(0.0)

            self.i += 1
            random_mask = np.random.permutation(len(indices))
            return np.array(indices)[random_mask], np.array(values)[random_mask], np.array(labels)[random_mask]







