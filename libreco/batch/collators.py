import random

import numpy as np
import torch

from .batch_unit import (
    PairFeats,
    PairwiseBatch,
    PointwiseBatch,
    PointwiseSepFeatBatch,
    SeqFeats,
    SparseBatch,
    SparseSeqFeats,
    TripleFeats,
)
from .enums import FeatType
from .sequence import get_interacted_seq, get_sparse_interacted
from ..graph import build_subgraphs, pairs_from_dgl_graph
from ..sampling import (
    neg_probs_from_frequency,
    negatives_from_out_batch,
    negatives_from_popular,
    negatives_from_random,
    negatives_from_unconsumed,
    pairs_from_random_walk,
    pos_probs_from_frequency,
)


class BaseCollator:
    def __init__(
        self,
        model,
        data_info,
        backend,
        separate_features=False,
        temperature=0.75,
    ):
        self.n_users = data_info.n_users
        self.n_items = data_info.n_items
        self.user_consumed = data_info.user_consumed
        self.item_consumed = data_info.item_consumed
        self.user_sparse_col_index = data_info.user_sparse_col.index
        self.item_sparse_col_index = data_info.item_sparse_col.index
        self.user_dense_col_index = data_info.user_dense_col.index
        self.item_dense_col_index = data_info.item_dense_col.index
        self.item_sparse_unique = data_info.item_sparse_unique
        self.item_dense_unique = data_info.item_dense_unique
        self.has_seq = True if model.model_category == "sequence" else False
        self.seq_mode = model.seq_mode if self.has_seq else None
        self.max_seq_len = model.max_seq_len if self.has_seq else None
        self.separate_features = separate_features
        self.backend = backend
        self.seed = model.seed
        self.temperature = temperature
        self.user_consumed_set = None
        self.neg_probs = None
        self.np_rng = None

    def __call__(self, batch):
        sparse_batch = self.get_features(batch, FeatType.SPARSE)
        dense_batch = self.get_features(batch, FeatType.DENSE)
        seq_batch = self.get_seqs(batch["user"], batch["item"])
        batch_cls = PointwiseSepFeatBatch if self.separate_features else PointwiseBatch
        batch_data = batch_cls(
            users=batch["user"],
            items=batch["item"],
            labels=batch["label"],
            sparse_indices=sparse_batch,
            dense_values=dense_batch,
            seqs=seq_batch,
            backend=self.backend,
        )
        return batch_data

    def get_col_index(self, feat_type):
        if feat_type is FeatType.SPARSE:
            user_col_index = self.user_sparse_col_index
            item_col_index = self.item_sparse_col_index
        elif feat_type is FeatType.DENSE:
            user_col_index = self.user_dense_col_index
            item_col_index = self.item_dense_col_index
        else:
            raise ValueError("`feat_type` must be sparse or dense.")
        return user_col_index, item_col_index

    def get_features(self, batch, feat_type):
        if feat_type.value not in batch:
            return
        features = batch[feat_type.value]
        if self.separate_features:
            user_col_index, item_col_index = self.get_col_index(feat_type)
            user_features = features[:, user_col_index] if user_col_index else None
            item_features = features[:, item_col_index] if item_col_index else None
            features = PairFeats(user_features, item_features)
        return features

    def get_seqs(self, user_indices, item_indices):
        if not self.has_seq:
            return
        self._set_random_seeds()
        self._set_user_consumed()
        batch_interacted, interacted_len = get_interacted_seq(
            user_indices,
            item_indices,
            self.user_consumed,
            self.n_items,
            self.seq_mode,
            self.max_seq_len,
            self.user_consumed_set,
            self.np_rng,
        )
        return SeqFeats(batch_interacted, interacted_len)

    def sample_neg_items(self, batch, sampler, num_neg):
        if sampler == "unconsumed":
            self._set_user_consumed()
            items_neg = negatives_from_unconsumed(
                self.user_consumed_set,
                batch["user"],
                batch["item"],
                self.n_items,
                num_neg,
            )
        elif sampler == "popular":
            self._set_random_seeds()
            self._set_neg_probs()
            items_neg = negatives_from_popular(
                self.np_rng,
                self.n_items,
                batch["item"],
                num_neg,
                probs=self.neg_probs,
            )
        else:
            self._set_random_seeds()
            items_neg = negatives_from_random(
                self.np_rng,
                self.n_items,
                batch["item"],
                num_neg,
            )
        return items_neg

    def _set_user_consumed(self):
        if self.user_consumed_set is None:
            self.user_consumed_set = [
                set(self.user_consumed[u]) for u in range(self.n_users)
            ]

    def _set_neg_probs(self):
        if self.neg_probs is None:
            self.neg_probs = neg_probs_from_frequency(
                self.item_consumed, self.n_items, self.temperature
            )

    def _set_random_seeds(self):
        if self.np_rng is None:
            worker_info = torch.utils.data.get_worker_info()
            seed = self.seed if worker_info is None else worker_info.seed
            seed = seed % 3407 * 11
            random.seed(seed)
            torch.manual_seed(seed)
            self.np_rng = np.random.default_rng(seed)


class SparseCollator(BaseCollator):
    def __init__(self, model, data_info, backend):
        super().__init__(model, data_info, backend)

    def __call__(self, batch):
        seq_batch = self.get_seqs(batch["user"], batch["item"])
        sparse_batch = self.get_features(batch, FeatType.SPARSE)
        dense_batch = self.get_features(batch, FeatType.DENSE)
        return SparseBatch(
            seqs=seq_batch,
            items=batch["item"],
            sparse_indices=sparse_batch,
            dense_values=dense_batch,
        )

    def get_seqs(self, user_indices, item_indices):
        batch_indices, batch_values, batch_size = get_sparse_interacted(
            user_indices,
            item_indices,
            self.user_consumed,
            self.seq_mode,
            self.max_seq_len,
        )
        return SparseSeqFeats(batch_indices, batch_values, batch_size)


class PointwiseCollator(BaseCollator):
    def __init__(self, model, data_info, backend):
        super().__init__(model, data_info, backend)
        self.sampler = model.sampler
        self.num_neg = model.num_neg

    def __call__(self, batch):
        user_batch = np.repeat(batch["user"], self.num_neg + 1)
        item_batch = np.repeat(batch["item"], self.num_neg + 1)
        label_batch = np.zeros_like(item_batch, dtype=np.float32)
        label_batch[:: (self.num_neg + 1)] = 1.0
        items_neg = self.sample_neg_items(batch, self.sampler, self.num_neg)
        for i in range(self.num_neg):
            item_batch[(i + 1) :: (self.num_neg + 1)] = items_neg[i :: self.num_neg]

        sparse_batch = self.get_pointwise_feats(batch, FeatType.SPARSE, item_batch)
        dense_batch = self.get_pointwise_feats(batch, FeatType.DENSE, item_batch)
        seq_batch = self.get_seqs(user_batch, item_batch)
        batch_data = PointwiseBatch(
            users=user_batch,
            items=item_batch,
            labels=label_batch,
            sparse_indices=sparse_batch,
            dense_values=dense_batch,
            seqs=seq_batch,
            backend=self.backend,
        )
        return batch_data

    def get_pointwise_feats(self, batch, feat_type, items):
        if feat_type.value not in batch:
            return
        batch_feats = batch[feat_type.value]
        user_col_index, item_col_index = self.get_col_index(feat_type)
        user_features = repeat_feats(batch_feats, user_col_index, self.num_neg)
        item_features = get_sampled_item_feats(self, item_col_index, items, feat_type)
        if self.separate_features:
            return PairFeats(user_features, item_features)
        if user_col_index and item_col_index:
            return merge_columns(
                user_features, item_features, user_col_index, item_col_index
            )
        return user_features if user_col_index else item_features


class PairwiseCollator(BaseCollator):
    def __init__(self, model, data_info, backend, repeat_positives):
        super().__init__(model, data_info, backend, separate_features=True)
        self.sampler = model.sampler
        self.num_neg = model.num_neg
        self.repeat_positives = repeat_positives

    def __call__(self, batch):
        if self.repeat_positives and self.num_neg > 1:
            users = np.repeat(batch["user"], self.num_neg)
            items_pos = np.repeat(batch["item"], self.num_neg)
        else:
            users = batch["user"]
            items_pos = batch["item"]
        items_neg = self.sample_neg_items(batch, self.sampler, self.num_neg)

        sparse_batch = self.get_pairwise_feats(batch, FeatType.SPARSE, items_neg)
        dense_batch = self.get_pairwise_feats(batch, FeatType.DENSE, items_neg)
        seq_batch = self.get_seqs(users, items_pos)
        if self.has_seq and not self.repeat_positives and self.num_neg > 1:
            seq_batch = seq_batch.repeat(self.num_neg)
        batch_data = PairwiseBatch(
            queries=users,
            item_pairs=(items_pos, items_neg),
            sparse_indices=sparse_batch,
            dense_values=dense_batch,
            seqs=seq_batch,
            backend=self.backend,
        )
        return batch_data

    def get_pairwise_feats(self, batch, feat_type, items_neg):
        if feat_type.value not in batch:
            return
        batch_feats = batch[feat_type.value]
        user_col_index, item_col_index = self.get_col_index(feat_type)
        if self.repeat_positives and self.num_neg > 1:
            user_feats = repeat_feats(
                batch_feats, user_col_index, self.num_neg, is_pairwise=True
            )
            item_pos_feats = repeat_feats(
                batch_feats, item_col_index, self.num_neg, is_pairwise=True
            )
        else:
            user_feats = batch_feats[:, user_col_index] if user_col_index else None
            item_pos_feats = batch_feats[:, item_col_index] if item_col_index else None
        item_neg_feats = get_sampled_item_feats(
            self, item_col_index, items_neg, feat_type
        )
        return TripleFeats(user_feats, item_pos_feats, item_neg_feats)


class GraphCollator(BaseCollator):
    def __init__(self, model, data_info, backend, alpha=1e-3):
        super().__init__(model, data_info, backend)
        self.neighbor_walker = model.neighbor_walker
        self.paradigm = model.paradigm
        self.sampler = model.sampler
        self.num_neg = model.num_neg
        self.num_walks = model.num_walks
        self.walk_length = model.sample_walk_len
        self.start_node = model.start_node
        self.focus_start = model.focus_start
        if self.start_node == "unpopular":
            self.pos_probs = pos_probs_from_frequency(
                self.item_consumed, self.n_users, self.n_items, alpha
            )

    def __call__(self, batch):
        self._set_random_seeds()
        if self.paradigm == "u2i":
            users, items_pos = batch["user"], batch["item"]
            items_neg = self.sample_neg_items(batch, self.sampler, self.num_neg)
            user_data = self.neighbor_walker.get_user_feats(users)
            item_pos_data = self.neighbor_walker(items_pos)
            item_neg_data = self.neighbor_walker(items_neg)
            return user_data, item_pos_data, item_neg_data
        else:
            start_nodes = self.get_start_nodes(batch)
            items, items_pos = pairs_from_random_walk(
                start_nodes,
                self.user_consumed,
                self.item_consumed,
                self.num_walks,
                self.walk_length,
                self.focus_start,
            )
            items_neg = self.sample_i2i_negatives(items, items_pos)
            item_data = self.neighbor_walker(items, items_pos)
            item_pos_data = self.neighbor_walker(items_pos)
            item_neg_data = self.neighbor_walker(items_neg)
            return item_data, item_pos_data, item_neg_data

    # exclude both items and items_pos
    def sample_i2i_negatives(self, items, items_pos):
        if self.sampler == "out-batch":
            items_neg = negatives_from_out_batch(
                self.np_rng, self.n_items, items_pos, items, self.num_neg
            )
        elif self.sampler == "popular":
            items_neg = negatives_from_popular(
                self.np_rng,
                self.n_items,
                items_pos,
                self.num_neg,
                items=items,
                probs=self.neg_probs,
            )
        else:
            items_neg = negatives_from_random(
                self.np_rng,
                self.n_items,
                items_pos,
                self.num_neg,
                items=items,
            )
        return items_neg

    def get_start_nodes(self, batch):
        size = len(batch["item"])
        if self.start_node == "unpopular":
            population = range(self.n_items)
            start_nodes = random.choices(population, weights=self.pos_probs, k=size)
        else:
            start_nodes = self.np_rng.integers(0, self.n_items, size=size)
            start_nodes = start_nodes.tolist()
        return start_nodes


class GraphDGLCollator(GraphCollator):
    def __init__(self, model, data_info, backend, alpha=1e-3):
        super().__init__(model, data_info, backend, alpha)
        self.graph = model.hetero_g
        self.dgl = model._dgl
        self.dgl_seed = None
        if self.start_node == "unpopular":
            self.pos_probs = torch.tensor(self.pos_probs, dtype=torch.float)

    def __call__(self, batch):
        self._set_random_seeds()
        self._set_dgl_seeds()
        if self.paradigm == "u2i":
            users, items_pos = batch["user"], batch["item"]
            items_neg = self.sample_neg_items(batch, self.sampler, self.num_neg)
            # nodes in pos_graph and neg_graph are same, difference is the connected edges
            pos_graph, neg_graph, *_ = build_subgraphs(
                users, (items_pos, items_neg), self.paradigm, self.num_neg
            )
            # user -> item heterogeneous graph, users on srcdata, items on dstdata
            all_users = pos_graph.srcdata[self.dgl.NID]
            all_items = pos_graph.dstdata[self.dgl.NID]
            user_data = self.neighbor_walker.get_user_feats(all_users)
            item_data = self.neighbor_walker(all_items)
            return user_data, item_data, pos_graph, neg_graph
        else:
            start_nodes = self.get_start_nodes(batch)
            items, items_pos = pairs_from_dgl_graph(
                self.graph,
                start_nodes,
                self.num_walks,
                self.walk_length,
                self.focus_start,
            )
            items_neg = self.sample_i2i_negatives(items, items_pos)
            # nodes in pos_graph and neg_graph are same, difference is the connected edges
            pos_graph, neg_graph, *target_nodes = build_subgraphs(
                items, (items_pos, items_neg), self.paradigm, self.num_neg
            )
            # item -> item homogeneous graph, items on all nodes
            all_items = pos_graph.ndata[self.dgl.NID]
            item_data = self.neighbor_walker(all_items, target_nodes)
            return item_data, pos_graph, neg_graph

    def get_start_nodes(self, batch):
        size = len(batch["item"])
        if self.start_node == "unpopular":
            start_nodes = torch.multinomial(self.pos_probs, size, replacement=True)
        else:
            start_nodes = torch.randint(0, self.n_items, (size,))
        return start_nodes

    def _set_dgl_seeds(self):
        if self.dgl_seed is None:
            worker_info = torch.utils.data.get_worker_info()
            seed = self.seed if worker_info is None else worker_info.seed
            seed = seed % 3407 * 11
            self.dgl.seed(seed)
            self.dgl_seed = True


def repeat_feats(batch_feats, col_index, num_neg, is_pairwise=False):
    if not col_index:
        return
    column_features = batch_feats[:, col_index]
    repeats = num_neg if is_pairwise else num_neg + 1
    return np.repeat(column_features, repeats, axis=0)


def get_sampled_item_feats(collator, item_col_index, items_sampled, feat_type):
    if not item_col_index:
        return
    if feat_type is FeatType.SPARSE:
        item_unique_features = collator.item_sparse_unique
    elif feat_type is FeatType.DENSE:
        item_unique_features = collator.item_dense_unique
    else:
        raise ValueError("`feat_type` must be sparse or dense.")
    return item_unique_features[items_sampled]


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
