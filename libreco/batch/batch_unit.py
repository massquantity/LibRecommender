from dataclasses import InitVar, dataclass
from typing import Generic, Iterable, Optional, Tuple, TypeVar

import numpy as np
import torch

from ..batch.enums import Backend

T = TypeVar("T", int, float)


@dataclass
class PairFeats(Generic[T]):
    user_feats: Optional[Iterable[T]]
    item_feats: Optional[Iterable[T]]

    def to_torch_tensor(self):
        if self.user_feats is not None:
            self.user_feats = torch.from_numpy(self.user_feats)
        if self.item_feats is not None:
            self.item_feats = torch.from_numpy(self.item_feats)
        return self


@dataclass
class TripleFeats(Generic[T]):
    query_feats: Optional[Iterable[T]]
    item_pos_feats: Optional[Iterable[T]]
    item_neg_feats: Optional[Iterable[T]]

    def to_torch_tensor(self):
        if self.query_feats is not None:
            self.query_feats = torch.from_numpy(self.query_feats)
        if self.item_pos_feats is not None:
            self.item_pos_feats = torch.from_numpy(self.item_pos_feats)
        if self.item_neg_feats is not None:
            self.item_neg_feats = torch.from_numpy(self.item_neg_feats)
        return self

    def to_device(self, device):  # pragma: no cover
        if self.query_feats is not None:
            self.query_feats = self.query_feats.to(device)
        if self.item_pos_feats is not None:
            self.item_pos_feats = self.item_pos_feats.to(device)
        if self.item_neg_feats is not None:
            self.item_neg_feats = self.item_neg_feats.to(device)
        return self


@dataclass
class SeqFeats:
    interacted_seq: Iterable[Iterable[int]]
    interacted_len: Iterable[float]

    def repeat(self, num):
        self.interacted_seq = np.repeat(self.interacted_seq, num, axis=0)
        self.interacted_len = np.repeat(self.interacted_len, num)
        return self


@dataclass
class SparseSeqFeats:
    interacted_indices: Iterable[Iterable[int]]
    interacted_values: Iterable[int]
    modified_batch_size: int


@dataclass
class PointwiseBatch:
    __slots__ = (
        "users",
        "items",
        "labels",
        "sparse_indices",
        "dense_values",
        "seqs",
    )

    users: Iterable[int]
    items: Iterable[int]
    labels: Iterable[float]
    sparse_indices: Optional[Iterable[int]]
    dense_values: Optional[Iterable[float]]
    seqs: Optional[SeqFeats]
    backend: InitVar[Backend]

    # todo: For now, no torch model uses sequence feature
    def __post_init__(self, backend):
        if backend is Backend.TORCH:
            self.users = torch.from_numpy(self.users).long()
            self.items = torch.from_numpy(self.items).long()
            self.labels = torch.from_numpy(self.labels)
            if self.sparse_indices is not None:
                self.sparse_indices = torch.from_numpy(self.sparse_indices)
            if self.dense_values is not None:
                self.dense_values = torch.from_numpy(self.dense_values)

    def to_device(self, device):  # pragma: no cover
        self.users = self.users.to(device)
        self.items = self.items.to(device)
        self.labels = self.labels.to(device)
        if self.sparse_indices is not None:
            self.sparse_indices = self.sparse_indices.to(device)
        if self.dense_values is not None:
            self.dense_values = self.dense_values.to(device)
        return self


@dataclass
class PointwiseSepFeatBatch(PointwiseBatch):
    sparse_indices: Optional[PairFeats[int]]
    dense_values: Optional[PairFeats[float]]

    def __post_init__(self, backend):
        if backend is Backend.TORCH:
            self.users = torch.from_numpy(self.users)
            self.items = torch.from_numpy(self.items)
            self.labels = torch.from_numpy(self.labels)
            if self.sparse_indices is not None:
                self.sparse_indices.to_torch_tensor()
            if self.dense_values is not None:
                self.dense_values.to_torch_tensor()


@dataclass
class PairwiseBatch:
    __slots__ = (
        "queries",
        "item_pairs",
        "sparse_indices",
        "dense_values",
        "seqs",
    )

    queries: Iterable[int]
    item_pairs: Tuple[Iterable[int], Iterable[int]]
    sparse_indices: Optional[TripleFeats[int]]
    dense_values: Optional[TripleFeats[float]]
    seqs: Optional[SeqFeats]
    backend: InitVar[Backend]

    def __post_init__(self, backend):
        if backend is Backend.TORCH:
            self.queries = torch.from_numpy(self.queries)
            self.item_pairs = (
                torch.from_numpy(self.item_pairs[0]),
                torch.from_numpy(self.item_pairs[1]),
            )
            if self.sparse_indices is not None:
                self.sparse_indices.to_torch_tensor()
            if self.dense_values is not None:
                self.dense_values.to_torch_tensor()

    def to_device(self, device):  # pragma: no cover
        self.queries = self.queries.to(device)
        self.item_pairs = (
            self.item_pairs[0].to(device),
            self.item_pairs[1].to(device),
        )
        if self.sparse_indices is not None:
            self.sparse_indices = self.sparse_indices.to_device(device)
        if self.dense_values is not None:
            self.dense_values = self.dense_values.to_device(device)
        return self


@dataclass
class SparseBatch:
    seqs: SparseSeqFeats
    items: Iterable[int]  # items are used as labels
    sparse_indices: Optional[Iterable[int]]
    dense_values: Optional[Iterable[float]]
