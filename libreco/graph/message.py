from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

import numpy as np
import torch


def to_cpu_tensor(data, dtype=None):
    if data is None:
        return
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    else:
        assert dtype is not None
        return torch.tensor(data, dtype=dtype)


@dataclass
class UserMessage:
    users: Iterable[int]
    sparse_indices: Optional[Iterable[int]]
    dense_values: Optional[Iterable[float]]

    def to_torch_tensor(self):
        self.users = to_cpu_tensor(self.users, dtype=torch.long)
        self.sparse_indices = to_cpu_tensor(self.sparse_indices, dtype=torch.long)
        self.dense_values = to_cpu_tensor(self.dense_values, dtype=torch.float)
        return self


@dataclass
class ItemMessage:
    items: Iterable[int]
    sparse_indices: Optional[Iterable[int]]
    dense_values: Optional[Iterable[float]]
    neighbors: List[Optional[List[int]]]
    neighbors_sparse: List[Optional[Iterable[int]]]
    neighbors_dense: List[Optional[Iterable[float]]]
    offsets: List[List[int]]
    weights: Optional[List[List[float]]]

    def to_torch_tensor(self):
        self.items = to_cpu_tensor(self.items, dtype=torch.long)
        self.sparse_indices = to_cpu_tensor(self.sparse_indices, dtype=torch.long)
        self.dense_values = to_cpu_tensor(self.dense_values, dtype=torch.float)
        self.neighbors = [to_cpu_tensor(n, dtype=torch.long) for n in self.neighbors]
        self.neighbors_sparse = [
            to_cpu_tensor(n, dtype=torch.long) for n in self.neighbors_sparse
        ]
        self.neighbors_dense = [
            to_cpu_tensor(n, dtype=torch.float) for n in self.neighbors_dense
        ]
        self.offsets = [to_cpu_tensor(n, dtype=torch.long) for n in self.offsets]
        if self.weights is not None:
            self.weights = [to_cpu_tensor(n, dtype=torch.float) for n in self.weights]
        return self


@dataclass
class ItemMessageDGL:
    blocks: List[Any]
    items: Iterable[int]
    sparse_indices: Optional[Iterable[int]]
    dense_values: Optional[Iterable[float]]

    def to_torch_tensor(self):
        self.items = to_cpu_tensor(self.items, dtype=torch.long)
        self.sparse_indices = to_cpu_tensor(self.sparse_indices, dtype=torch.long)
        self.dense_values = to_cpu_tensor(self.dense_values, dtype=torch.float)
        return self
