"""

Reference: Rex Ying et al. "Graph Convolutional Neural Networks for Web-Scale Recommender Systems"
           (https://arxiv.org/abs/1806.01973)

author: massquantity

"""
import numpy as np
import torch
from tqdm import tqdm

from .torch_modules import PinSageModel
from ..bases import EmbedBase
from ..sampling import bipartite_neighbors_with_weights
from ..torchops import (
    feat_to_tensor,
    user_unique_to_tensor,
    item_unique_to_tensor,
)
from ..training import SageTrainer


class PinSage(EmbedBase):
    def __init__(
        self,
        task,
        data_info,
        loss_type="max_margin",
        paradigm="i2i",
        embed_size=16,
        n_epochs=20,
        lr=0.001,
        lr_decay=False,
        epsilon=1e-8,
        amsgrad=False,
        reg=None,
        batch_size=256,
        num_neg=1,
        dropout=0.0,
        remove_edges=False,
        num_layers=2,
        num_neighbors=3,
        num_walks=10,
        neighbor_walk_len=2,
        sample_walk_len=5,
        termination_prob=0.5,
        margin=1.0,
        sampler="random",
        start_node="random",
        focus_start=False,
        seed=42,
        k=10,
        eval_batch_size=8192,
        eval_user_num=None,
        device=torch.device("cpu"),
        lower_upper_bound=None,
        with_training=True,
    ):
        super().__init__(task, data_info, embed_size, lower_upper_bound)

        self.all_args = locals()
        self.loss_type = loss_type
        self.batch_size = batch_size
        self.dropout = dropout
        self.paradigm = paradigm
        self.remove_edges = remove_edges
        self.num_layers = num_layers
        self.num_neighbors = num_neighbors
        self.num_walks = num_walks
        self.neighbor_walk_len = neighbor_walk_len
        self.sample_walk_len = sample_walk_len
        self.termination_prob = termination_prob
        self.seed = seed
        self.device = device
        self._check_params()
        if with_training:
            self.torch_model = PinSageModel(
                self.paradigm,
                self.data_info,
                self.embed_size,
                self.batch_size,
                self.num_layers,
                self.dropout,
            ).to(device)
            self.trainer = SageTrainer(
                self,
                task,
                loss_type,
                n_epochs,
                lr,
                lr_decay,
                epsilon,
                amsgrad,
                reg,
                batch_size,
                num_neg,
                paradigm,
                num_walks,
                sample_walk_len,
                margin,
                sampler,
                start_node,
                focus_start,
                k,
                eval_batch_size,
                eval_user_num,
                device,
            )

    def _check_params(self):
        if self.task != "ranking":
            raise ValueError("PinSage is only suitable for ranking")
        if self.paradigm not in ("u2i", "i2i"):
            raise ValueError("paradigm must either be `u2i` or `i2i`")
        if self.loss_type not in ("cross_entropy", "focal", "bpr", "max_margin"):
            raise ValueError(f"unsupported `loss_type` for u2i: {self.loss_type}")

    def get_user_repr(self, users, sparse_indices, dense_values):
        query_feats = feat_to_tensor(users, sparse_indices, dense_values, self.device)
        return self.torch_model.user_repr(*query_feats)

    def get_item_repr(self, items, sparse_indices, dense_values, items_pos=None):
        nodes = items
        item_indices = (
            list(range(len(nodes)))
            if self.paradigm == "i2i" and self.remove_edges
            else None
        )
        tensor_neighbors, tensor_weights, tensor_offsets = [], [], []
        tensor_neighbor_sparse_indices, tensor_neighbor_dense_values = [], []
        for _ in range(self.num_layers):
            (
                neighbors,
                weights,
                offsets,
                item_indices_in_samples,
            ) = bipartite_neighbors_with_weights(
                nodes,
                self.data_info.user_consumed,
                self.data_info.item_consumed,
                self.num_neighbors,
                self.num_walks,
                self.neighbor_walk_len,
                items,
                item_indices,
                items_pos,
            )
            neighbor_tensors = item_unique_to_tensor(
                neighbors,
                self.data_info,
                self.device,
            )
            tensor_neighbors.append(neighbor_tensors[0])
            tensor_neighbor_sparse_indices.append(neighbor_tensors[1])
            tensor_neighbor_dense_values.append(neighbor_tensors[2])
            tensor_weights.append(
                torch.tensor(weights, dtype=torch.float, device=self.device)
            )
            tensor_offsets.append(
                torch.tensor(offsets, dtype=torch.long, device=self.device)
            )
            nodes = neighbors
            item_indices = item_indices_in_samples

        item_feats_tensor = feat_to_tensor(
            items, sparse_indices, dense_values, self.device
        )
        return self.torch_model(
            *item_feats_tensor,
            tensor_neighbors,
            tensor_neighbor_sparse_indices,
            tensor_neighbor_dense_values,
            tensor_weights,
            tensor_offsets,
        )

    @torch.no_grad()
    def set_embeddings(self):
        self.torch_model.eval()
        all_items = list(range(self.n_items))
        item_embed = []
        for i in tqdm(range(0, self.n_items, self.batch_size), desc="item embedding"):
            batch_items = all_items[i : i + self.batch_size]
            nodes = batch_items
            tensor_neighbors, tensor_weights, tensor_offsets = [], [], []
            tensor_neighbor_sparse_indices, tensor_neighbor_dense_values = [], []
            for _ in range(self.num_layers):
                neighbors, weights, offsets, _, = bipartite_neighbors_with_weights(
                    nodes,
                    self.data_info.user_consumed,
                    self.data_info.item_consumed,
                    num_neighbors=self.num_neighbors,
                    num_walks=self.num_walks,
                    walk_length=self.neighbor_walk_len,
                )
                (
                    neighbor_tensor,
                    neighbor_sparse_indices,
                    neighbor_dense_values,
                ) = item_unique_to_tensor(
                    neighbors,
                    self.data_info,
                    self.device,
                )
                tensor_neighbors.append(neighbor_tensor)
                tensor_neighbor_sparse_indices.append(neighbor_sparse_indices)
                tensor_neighbor_dense_values.append(neighbor_dense_values)
                tensor_weights.append(
                    torch.tensor(weights, dtype=torch.float, device=self.device)
                )
                tensor_offsets.append(
                    torch.tensor(offsets, dtype=torch.long, device=self.device)
                )
                nodes = neighbors

            item_tensor, item_sparse_indices, item_dense_values = item_unique_to_tensor(
                batch_items, self.data_info, self.device
            )
            item_reprs = self.torch_model(
                item_tensor,
                item_sparse_indices,
                item_dense_values,
                tensor_neighbors,
                tensor_neighbor_sparse_indices,
                tensor_neighbor_dense_values,
                tensor_weights,
                tensor_offsets,
            )
            item_embed.append(item_reprs.cpu().numpy())
        self.item_embed = np.concatenate(item_embed, axis=0)
        self.user_embed = self.get_user_embeddings()

    @torch.no_grad()
    def get_user_embeddings(self):
        self.torch_model.eval()
        user_embed = []
        if self.paradigm == "u2i":
            for i in range(0, self.n_users, self.batch_size):
                users = np.arange(i, min(i + self.batch_size, self.n_users))
                user_tensors = user_unique_to_tensor(users, self.data_info, self.device)
                user_reprs = self.torch_model.user_repr(*user_tensors)
                user_embed.append(user_reprs.cpu().numpy())
            return np.concatenate(user_embed, axis=0)
        else:
            for u in range(self.n_users):
                items = self.user_consumed[u]
                user_embed.append(np.mean(self.item_embed[items], axis=0))
                # user_embed.append(self.item_embed[items[-1]])
            return np.array(user_embed)
