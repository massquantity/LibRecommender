import math

import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

from .collators import BaseCollator as NormalCollator
from .collators import (
    GraphCollator,
    GraphDGLCollator,
    PairwiseCollator,
    PointwiseCollator,
    SparseCollator,
)
from .enums import Backend
from ..utils.constants import FEAT_TRAIN_MODELS, TF_TRAIN_MODELS


class BatchData(torch.utils.data.Dataset):
    def __init__(self, data, use_features, factor=None):
        self.user_indices = data.user_indices
        self.item_indices = data.item_indices
        self.labels = data.labels
        self.sparse_indices = data.sparse_indices
        self.dense_values = data.dense_values
        self.use_features = use_features
        self.factor = factor

    def __getitem__(self, idx):
        batch = {
            "user": self.user_indices[idx],
            "item": self.item_indices[idx],
            "label": self.labels[idx],
        }
        if self.use_features and self.sparse_indices is not None:
            batch["sparse"] = self.sparse_indices[idx]
        if self.use_features and self.dense_values is not None:
            batch["dense"] = self.dense_values[idx]
        return batch

    def __len__(self):
        length = len(self.labels)
        return math.ceil(length / self.factor) if self.factor is not None else length


def get_batch_loader(model, data, neg_sampling, batch_size, shuffle, num_workers=0):
    use_features = True if model.model_name in FEAT_TRAIN_MODELS else False
    factor = (
        model.num_walks * model.sample_walk_len
        if "Sage" in model.model_name and model.paradigm == "i2i"
        else None
    )
    batch_data = BatchData(data, use_features, factor)
    sampler = RandomSampler(batch_data) if shuffle else SequentialSampler(batch_data)
    batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
    collate_fn = get_collate_fn(model, neg_sampling, num_workers)
    return DataLoader(
        batch_data,
        batch_size=None,  # `batch_size=None` disables automatic batching
        sampler=batch_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )


def get_collate_fn(model, neg_sampling, num_workers):
    model_name, data_info = model.model_name, model.data_info
    backend = Backend.TF if model_name in TF_TRAIN_MODELS else Backend.TORCH
    if model_name == "YouTubeRetrieval":
        collate_fn = SparseCollator(model, data_info, backend)
    elif "Sage" in model.model_name:
        if model.use_dgl:
            assert num_workers == 0, "DGL models can't use multiprocessing data loader"
            collate_fn = GraphDGLCollator(model, data_info, backend)
        else:
            collate_fn = GraphCollator(model, data_info, backend)
    elif model.task == "rating" or not neg_sampling:
        collate_fn = NormalCollator(model, data_info, backend)
    else:
        if model.loss_type in ("cross_entropy", "focal"):
            collate_fn = PointwiseCollator(model, data_info, backend)
        else:
            repeat_positives = True if backend is Backend.TF else False
            collate_fn = PairwiseCollator(model, data_info, backend, repeat_positives)
    return collate_fn


# consider negative sampling and random walks in batch_size
def adjust_batch_size(model, original_batch_size):
    if model.model_name == "YouTubeRetrieval":
        return original_batch_size
    elif "Sage" in model.model_name and model.paradigm == "i2i":
        walk_len = model.sample_walk_len
        bs = original_batch_size / model.num_neg / model.num_walks / walk_len
        return max(1, int(bs))
    elif model.sampler is not None:
        if model.loss_type in ("cross_entropy", "focal"):
            return max(1, int(original_batch_size / (model.num_neg + 1)))
        else:
            return max(1, int(original_batch_size / model.num_neg))
    return original_batch_size
