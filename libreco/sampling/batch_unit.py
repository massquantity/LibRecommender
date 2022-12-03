from dataclasses import dataclass
from typing import Iterable, Optional, Tuple


@dataclass
class PointwiseBatch:
    __slots__ = (
        "users",
        "items",
        "labels",
        "sparse_indices",
        "dense_values",
        "interacted_seq",
        "interacted_len",
    )

    users: Iterable[int]
    items: Iterable[int]
    labels: Iterable[float]
    sparse_indices: Optional[Iterable[int]]
    dense_values: Optional[Iterable[float]]
    interacted_seq: Optional[Iterable[Iterable[int]]]
    interacted_len: Optional[Iterable[float]]


@dataclass
class PointwiseSepFeatBatch(PointwiseBatch):
    sparse_indices: Tuple[Optional[Iterable[int]], Optional[Iterable[int]]]
    dense_values: Tuple[Optional[Iterable[float]], Optional[Iterable[float]]]


@dataclass
class PairwiseBatch:
    __slots__ = (
        "queries",
        "item_pairs",
        "sparse_indices",
        "dense_values",
        "interacted_seq",
        "interacted_len",
    )

    queries: Iterable[int]
    item_pairs: Tuple[Iterable[int], Iterable[int]]
    sparse_indices: Tuple[
        Optional[Iterable[int]],
        Optional[Iterable[int]],
        Optional[Iterable[int]],
    ]
    dense_values: Tuple[
        Optional[Iterable[float]],
        Optional[Iterable[float]],
        Optional[Iterable[float]],
    ]
    interacted_seq: Optional[Iterable[Iterable[int]]]
    interacted_len: Optional[Iterable[float]]
