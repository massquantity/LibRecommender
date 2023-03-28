from dataclasses import InitVar, dataclass
from typing import Iterable, List, Optional

import numpy as np
import torch

from ..batch.enums import Backend


def to_cpu_tensor(data, dtype=None):
    if data is None:
        return
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    else:
        assert dtype is not None
        return torch.tensor(data, dtype=dtype)


def to_device(data, device):
    if data is None:
        return
    return data.to(device)


class Message:
    def to_device(self, device):
        raise NotImplementedError


@dataclass
class UserMessage(Message):
    users: Iterable[int]
    sparse_indices: Optional[Iterable[int]]
    dense_values: Optional[Iterable[float]]
    backend: InitVar[Backend] = Backend.TORCH

    def __post_init__(self, backend):
        if backend is Backend.TORCH:
            self.users = to_cpu_tensor(self.users, dtype=torch.long)
            self.sparse_indices = to_cpu_tensor(self.sparse_indices, dtype=torch.long)
            self.dense_values = to_cpu_tensor(self.dense_values, dtype=torch.float)

    def to_device(self, device):
        self.users = to_device(self.users, device)
        self.sparse_indices = to_device(self.sparse_indices, device)
        self.dense_values = to_device(self.dense_values, device)
        return self


@dataclass
class ItemMessage(Message):
    items: Iterable[int]
    sparse_indices: Optional[Iterable[int]]
    dense_values: Optional[Iterable[float]]
    neighbors: List[Optional[List[int]]]
    neighbors_sparse: List[Optional[Iterable[int]]]
    neighbors_dense: List[Optional[Iterable[float]]]
    offsets: List[List[int]]
    weights: Optional[List[List[float]]]
    backend: InitVar[Backend] = Backend.TORCH

    def __post_init__(self, backend):
        if backend is Backend.TORCH:
            self.items = to_cpu_tensor(self.items, dtype=torch.long)
            self.sparse_indices = to_cpu_tensor(self.sparse_indices, dtype=torch.long)
            self.dense_values = to_cpu_tensor(self.dense_values, dtype=torch.float)
            self.neighbors = [
                to_cpu_tensor(n, dtype=torch.long) for n in self.neighbors
            ]
            self.neighbors_sparse = [
                to_cpu_tensor(n, dtype=torch.long) for n in self.neighbors_sparse
            ]
            self.neighbors_dense = [
                to_cpu_tensor(n, dtype=torch.float) for n in self.neighbors_dense
            ]
            self.offsets = [to_cpu_tensor(n, dtype=torch.long) for n in self.offsets]
            if self.weights is not None:
                self.weights = [
                    to_cpu_tensor(n, dtype=torch.float) for n in self.weights
                ]

    def to_device(self, device):
        self.items = to_device(self.items, device)
        self.sparse_indices = to_device(self.sparse_indices, device)
        self.dense_values = to_device(self.dense_values, device)
        self.neighbors = [to_device(n, device) for n in self.neighbors]
        self.neighbors_sparse = [to_device(n, device) for n in self.neighbors_sparse]
        self.neighbors_dense = [to_device(n, device) for n in self.neighbors_dense]
        self.offsets = [to_device(n, device) for n in self.offsets]
        if self.weights is not None:
            self.weights = [to_device(n, device) for n in self.weights]
        return self


# https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/pin_memory.py#L53
# noinspection PyUnresolvedReferences
@dataclass
class ItemMessageDGL(Message):
    blocks: List["dgl.DGLBlock"]  # noqa: F821
    items: Iterable[int]
    sparse_indices: Optional[Iterable[int]]
    dense_values: Optional[Iterable[float]]
    backend: InitVar[Backend] = Backend.TORCH

    def __post_init__(self, backend):
        if backend is Backend.TORCH:
            self.items = to_cpu_tensor(self.items, dtype=torch.long)
            self.sparse_indices = to_cpu_tensor(self.sparse_indices, dtype=torch.long)
            self.dense_values = to_cpu_tensor(self.dense_values, dtype=torch.float)

    def to_device(self, device):
        self.blocks = [to_device(block, device) for block in self.blocks]
        self.items = to_device(self.items, device)
        self.sparse_indices = to_device(self.sparse_indices, device)
        self.dense_values = to_device(self.dense_values, device)
        return self
