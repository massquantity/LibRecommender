import random

import numpy as np
import torch
from tqdm import tqdm

from ..graph import pairs_from_dgl_graph
from .batch_unit import PairwiseBatch, PointwiseBatch, PointwiseSepFeatBatch
from .negatives import (
    neg_probs_from_frequency,
    negatives_from_out_batch,
    negatives_from_popular,
    negatives_from_random,
    negatives_from_unconsumed,
    pos_probs_from_frequency,
)
from .random_walks import pairs_from_random_walk


class DataGenerator:
    def __init__(
        self,
        dataset,
        data_info,
        batch_size,
        num_neg,
        sampler,
        seed,
        use_features=True,
        separate_features=False,
        temperature=0.75,
    ):
        self.data_info = data_info
        self.batch_size = batch_size
        self.num_neg = num_neg
        self.n_users = data_info.n_users
        self.n_items = data_info.n_items
        self.user_indices = dataset.user_indices
        self.item_indices = dataset.item_indices
        self.labels = dataset.labels
        self.sparse_indices = dataset.sparse_indices
        self.dense_values = dataset.dense_values
        self.data_size = len(dataset)
        self.sampler = sampler
        self.random_generator = np.random.default_rng(seed)
        self.use_features = use_features
        self.separate_features = separate_features
        if self.sampler == "unconsumed":
            self.user_consumed_set = [
                set(data_info.user_consumed[u]) for u in range(self.n_users)
            ]
        elif self.sampler == "popular":
            self.neg_probs = neg_probs_from_frequency(
                data_info.item_consumed, self.n_items, temperature
            )

    def __call__(self, shuffle=True):
        if shuffle:
            mask = self.random_generator.permutation(self.data_size)
            user_indices = self.user_indices[mask]
            item_indices = self.item_indices[mask]
            labels = self.labels[mask]
            sparse_indices = (
                self.sparse_indices[mask]
                if self.sparse_indices is not None and self.use_features
                else None
            )
            dense_values = (
                self.dense_values[mask]
                if self.dense_values is not None and self.use_features
                else None
            )
        else:
            user_indices = self.user_indices
            item_indices = self.item_indices
            labels = self.labels
            sparse_indices = self.sparse_indices if self.use_features else None
            dense_values = self.dense_values if self.use_features else None
        return self._iterate_data(
            user_indices, item_indices, sparse_indices, dense_values, labels
        )

    def _iterate_data(
        self, user_indices, item_indices, sparse_indices, dense_values, labels
    ):
        batch_cls = PointwiseSepFeatBatch if self.separate_features else PointwiseBatch
        for k in tqdm(
            range(0, self.data_size, self.batch_size), desc="pointwise data iterator"
        ):
            batch_slice = slice(k, k + self.batch_size)
            user_batch = user_indices[batch_slice]
            item_batch = item_indices[batch_slice]
            label_batch = labels[batch_slice]
            sparse_batch = (
                sparse_indices[batch_slice] if sparse_indices is not None else None
            )
            dense_batch = (
                dense_values[batch_slice] if dense_values is not None else None
            )
            if self.separate_features:
                sparse_batch = self.get_pair_features(
                    sparse_indices, batch_slice, is_sparse=True
                )
                dense_batch = self.get_pair_features(
                    dense_values, batch_slice, is_sparse=False
                )
            yield batch_cls(
                users=user_batch,
                items=item_batch,
                labels=label_batch,
                sparse_indices=sparse_batch,
                dense_values=dense_batch,
                interacted_seq=None,
                interacted_len=None,
            )

    def get_pair_features(self, features, batch_slice, is_sparse):
        if features is None:
            return None, None
        if is_sparse:
            user_col_index = self.data_info.user_sparse_col.index
            item_col_index = self.data_info.item_sparse_col.index
        else:
            user_col_index = self.data_info.user_dense_col.index
            item_col_index = self.data_info.item_dense_col.index
        batch_features = features[batch_slice]
        user_features = batch_features[:, user_col_index] if user_col_index else None
        item_features = batch_features[:, item_col_index] if item_col_index else None
        return user_features, item_features

    def get_sampled_item_features(self, item_indices_sampled, is_sparse):
        if is_sparse:
            item_features = self.data_info.item_sparse_unique
        else:
            item_features = self.data_info.item_dense_unique
        return item_features[item_indices_sampled]


class PointwiseDataGenerator(DataGenerator):
    def __init__(
        self,
        dataset,
        data_info,
        batch_size,
        num_neg,
        sampler,
        seed,
        use_features=True,
        separate_features=False,
        temperature=0.75,
    ):
        super().__init__(
            dataset,
            data_info,
            batch_size,
            num_neg,
            sampler,
            seed,
            use_features,
            separate_features,
            temperature,
        )
        self.batch_size = max(1, int(batch_size / (num_neg + 1)))

    def _iterate_data(
        self, user_indices, item_indices, sparse_indices, dense_values, *_
    ):
        batch_cls = PointwiseSepFeatBatch if self.separate_features else PointwiseBatch
        for k in tqdm(
            range(0, self.data_size, self.batch_size), desc="pointwise data iterator"
        ):
            batch_slice = slice(k, k + self.batch_size)
            user_batch = np.repeat(user_indices[batch_slice], self.num_neg + 1)
            item_batch = np.repeat(item_indices[batch_slice], self.num_neg + 1)
            label_batch = np.zeros_like(item_batch, dtype=np.float32)
            label_batch[:: (self.num_neg + 1)] = 1.0
            if self.sampler == "unconsumed":
                items_neg = negatives_from_unconsumed(
                    self.user_consumed_set,
                    user_indices[batch_slice],
                    item_indices[batch_slice],
                    self.n_items,
                    self.num_neg,
                )
            elif self.sampler == "popular":
                items_neg = negatives_from_popular(
                    self.random_generator,
                    self.n_items,
                    item_indices[batch_slice],
                    self.num_neg,
                    probs=self.neg_probs,
                )
            else:
                items_neg = negatives_from_random(
                    self.random_generator,
                    self.n_items,
                    item_indices[batch_slice],
                    self.num_neg,
                )
            for i in range(self.num_neg):
                item_batch[(i + 1) :: (self.num_neg + 1)] = items_neg[i :: self.num_neg]

            if sparse_indices is None:
                sparse_batch = (None, None) if self.separate_features else None
            else:
                sparse_batch = self.get_features(
                    sparse_indices[batch_slice], item_batch, is_sparse=True
                )
            if dense_values is None:
                dense_batch = (None, None) if self.separate_features else None
            else:
                dense_batch = self.get_features(
                    dense_values[batch_slice], item_batch, is_sparse=False
                )
            yield batch_cls(
                users=user_batch,
                items=item_batch,
                labels=label_batch,
                sparse_indices=sparse_batch,
                dense_values=dense_batch,
                interacted_seq=None,
                interacted_len=None,
            )

    def get_features(self, batch_features, item_indices_sampled, is_sparse):
        if is_sparse:
            user_col_index = self.data_info.user_sparse_col.index
            item_col_index = self.data_info.item_sparse_col.index
        else:
            user_col_index = self.data_info.user_dense_col.index
            item_col_index = self.data_info.item_dense_col.index

        user_features = (
            np.repeat(batch_features[:, user_col_index], self.num_neg + 1, axis=0)
            if user_col_index
            else None
        )
        item_features = (
            self.get_sampled_item_features(item_indices_sampled, is_sparse)
            if item_col_index
            else None
        )
        if self.separate_features:
            return user_features, item_features
        if user_col_index and item_col_index:
            return self.merge_columns(
                user_features, item_features, user_col_index, item_col_index
            )
        return user_features if user_col_index else item_features

    @staticmethod
    def merge_columns(user_features, item_features, user_col_index, item_col_index):
        if len(user_features) != len(item_features):
            raise ValueError(
                f"length of user_features and length of item_features don't match, "
                f"got {len(user_features)} and {len(item_features)}"
            )
        # keep column names in original order
        orig_cols = user_col_index + item_col_index
        col_reindex = np.arange(len(orig_cols))[np.argsort(orig_cols)]
        concat_features = np.concatenate([user_features, item_features], axis=1)
        return concat_features[:, col_reindex]


class PairwiseDataGenerator(DataGenerator):
    def __init__(
        self,
        dataset,
        data_info,
        batch_size,
        num_neg,
        sampler,
        seed,
        use_features=True,
        repeat_positives=True,
        temperature=0.75,
    ):
        super().__init__(
            dataset,
            data_info,
            batch_size,
            num_neg,
            sampler,
            seed,
            use_features,
            True,
            temperature,
        )
        self.batch_size = max(1, int(batch_size / num_neg))
        self.repeat_positives = repeat_positives

    def _iterate_data(
        self, user_indices, item_indices, sparse_indices, dense_values, *_
    ):
        for k in tqdm(
            range(0, self.data_size, self.batch_size), desc="pairwise data iterator"
        ):
            batch_slice = slice(k, k + self.batch_size)
            if self.num_neg > 1 and self.repeat_positives:
                users = np.repeat(user_indices[batch_slice], self.num_neg)
                items_pos = np.repeat(item_indices[batch_slice], self.num_neg)
            else:
                users = user_indices[batch_slice]
                items_pos = item_indices[batch_slice]

            if self.sampler == "unconsumed":
                items_neg = negatives_from_unconsumed(
                    self.user_consumed_set,
                    user_indices[batch_slice],
                    item_indices[batch_slice],
                    self.n_items,
                    self.num_neg,
                )
            elif self.sampler == "popular":
                items_neg = negatives_from_popular(
                    self.random_generator,
                    self.n_items,
                    item_indices[batch_slice],
                    self.num_neg,
                    probs=self.neg_probs,
                )
            else:
                items_neg = negatives_from_random(
                    self.random_generator,
                    self.n_items,
                    item_indices[batch_slice],
                    self.num_neg,
                )

            if sparse_indices is None:
                sparse_batch = (None, None, None)
            else:
                sparse_batch = self.get_features(
                    sparse_indices[batch_slice], items_neg, is_sparse=True
                )
            if dense_values is None:
                dense_batch = (None, None, None)
            else:
                dense_batch = self.get_features(
                    dense_values[batch_slice], items_neg, is_sparse=False
                )
            yield PairwiseBatch(
                queries=users,
                item_pairs=(items_pos, items_neg),
                sparse_indices=sparse_batch,
                dense_values=dense_batch,
                interacted_seq=None,
                interacted_len=None,
            )

    def get_features(self, batch_features, items_neg, is_sparse):
        if is_sparse:
            user_col_index = self.data_info.user_sparse_col.index
            item_col_index = self.data_info.item_sparse_col.index
        else:
            user_col_index = self.data_info.user_dense_col.index
            item_col_index = self.data_info.item_dense_col.index

        user_features = self.repeat_features(batch_features, user_col_index)
        item_pos_features = self.repeat_features(batch_features, item_col_index)
        item_neg_features = (
            self.get_sampled_item_features(items_neg, is_sparse=is_sparse)
            if item_col_index
            else None
        )
        return user_features, item_pos_features, item_neg_features

    def repeat_features(self, batch_features, col_index):
        if not col_index:
            return
        column_features = batch_features[:, col_index]
        if self.repeat_positives and self.num_neg > 1:
            column_features = np.repeat(column_features, self.num_neg, axis=0)
        return column_features


class PairwiseRandomWalkGenerator(PairwiseDataGenerator):
    def __init__(
        self,
        dataset,
        data_info,
        batch_size,
        num_neg,
        num_walks,
        walk_length,
        sampler,
        seed,
        use_features=True,
        repeat_positives=False,
        start_nodes="random",
        focus_start=False,
        temperature=0.75,
        alpha=1e-3,
        graph=None,
    ):
        super().__init__(
            dataset,
            data_info,
            batch_size,
            num_neg,
            sampler,
            seed,
            use_features,
            repeat_positives,
            temperature,
        )
        self.batch_size = max(1, int(batch_size / num_neg / num_walks / walk_length))
        self.user_consumed = data_info.user_consumed
        self.item_consumed = data_info.item_consumed
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.start_nodes = start_nodes
        self.focus_start = focus_start
        self.graph = graph
        if self.start_nodes not in ("random", "unpopular"):
            raise ValueError("`start_nodes` must either be `random` or `unpopular`")
        if self.start_nodes == "unpopular":
            self.pos_probs = pos_probs_from_frequency(
                data_info.item_consumed, self.n_users, self.n_items, alpha
            )
            if graph is not None:
                self.pos_probs = torch.tensor(self.pos_probs, dtype=torch.float)

    def __call__(self, *_):
        return self._iterate_data()

    def _iterate_data(self, *_):
        n_batches = int(
            self.data_size / self.num_walks / self.walk_length / self.batch_size
        )
        for _ in tqdm(range(n_batches), desc="pairwise random walk iterator"):
            if self.graph is not None:
                if self.start_nodes == "unpopular":
                    start_nodes = torch.multinomial(
                        self.pos_probs, self.batch_size, replacement=True
                    )
                else:
                    start_nodes = torch.randint(0, self.n_items, (self.batch_size,))
                items, items_pos = pairs_from_dgl_graph(
                    self.graph,
                    start_nodes,
                    self.num_walks,
                    self.walk_length,
                    self.focus_start,
                )
            else:
                if self.start_nodes == "unpopular":
                    start_nodes = random.choices(
                        range(self.n_items), weights=self.pos_probs, k=self.batch_size
                    )
                else:
                    start_nodes = self.random_generator.integers(
                        0, self.n_items, size=self.batch_size
                    ).tolist()
                items, items_pos = pairs_from_random_walk(
                    start_nodes,
                    self.user_consumed,
                    self.item_consumed,
                    self.num_walks,
                    self.walk_length,
                    self.focus_start,
                )

            if self.sampler == "out-batch":
                items_neg = negatives_from_out_batch(
                    self.random_generator, self.n_items, items_pos, items, self.num_neg
                )
            elif self.sampler == "popular":
                items_neg = negatives_from_popular(
                    self.random_generator,
                    self.n_items,
                    items_pos,
                    self.num_neg,
                    items=items,
                    probs=self.neg_probs,
                )
            else:
                items_neg = negatives_from_random(
                    self.random_generator,
                    self.n_items,
                    items_pos,
                    self.num_neg,
                    items=items,
                )

            if self.repeat_positives and self.num_neg > 1:
                items = np.repeat(items, self.num_neg)
                items_pos = np.repeat(items_pos, self.num_neg)

            sparse_feats = (
                self.get_feats(items, items_pos, items_neg, is_sparse=True)
                if self.use_features
                else None
            )
            dense_feats = (
                self.get_feats(items, items_pos, items_neg, is_sparse=False)
                if self.use_features
                else None
            )
            yield PairwiseBatch(
                queries=items,
                item_pairs=(items_pos, items_neg),
                sparse_indices=sparse_feats,
                dense_values=dense_feats,
                interacted_seq=None,
                interacted_len=None,
            )

    def get_feats(self, items, items_pos, items_neg, is_sparse):
        if is_sparse:
            col_index = self.data_info.item_sparse_col.index
        else:
            col_index = self.data_info.item_dense_col.index
        if col_index:
            item_feats = self.get_sampled_item_features(items, is_sparse=is_sparse)
            item_pos_feats = self.get_sampled_item_features(
                items_pos, is_sparse=is_sparse
            )
            item_neg_feats = self.get_sampled_item_features(
                items_neg, is_sparse=is_sparse
            )
        else:
            item_feats = item_pos_feats = item_neg_feats = None
        return item_feats, item_pos_feats, item_neg_feats
