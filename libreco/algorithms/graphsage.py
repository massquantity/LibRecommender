"""

Reference: William L. Hamilton et al. "Inductive Representation Learning on Large Graphs"
           (https://arxiv.org/abs/1706.02216)

author: massquantity

"""
import numpy as np
import torch
from tqdm import tqdm

from ..bases import EmbedBase, ModelMeta
from ..sampling import bipartite_neighbors
from ..torchops import feat_to_tensor, item_unique_to_tensor, user_unique_to_tensor
from ..training import SageTrainer
from .torch_modules import GraphSageModel


class GraphSage(EmbedBase, metaclass=ModelMeta, backend="torch"):
    def __init__(
        self,
        task,
        data_info,
        loss_type="cross_entropy",
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
        dropout_rate=0.0,
        remove_edges=False,
        num_layers=2,
        num_neighbors=3,
        num_walks=10,
        sample_walk_len=5,
        margin=1.0,
        sampler="random",
        start_node="random",
        focus_start=False,
        seed=42,
        device=torch.device("cpu"),
        lower_upper_bound=None,
        with_training=True,
    ):
        super().__init__(task, data_info, embed_size, lower_upper_bound)

        self.all_args = locals()
        self.loss_type = loss_type
        self.batch_size = batch_size
        self.paradigm = paradigm
        self.dropout_rate = dropout_rate
        self.remove_edges = remove_edges
        self.num_layers = num_layers
        self.num_neighbors = num_neighbors
        self.seed = seed
        self.device = device
        self._check_params()
        if with_training:
            self.torch_model = self.build_model().to(device)
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
                device,
            )

    def _check_params(self):
        if self.task != "ranking":
            raise ValueError(f"{self.model_name} is only suitable for ranking")
        if self.paradigm not in ("u2i", "i2i"):
            raise ValueError("paradigm must either be `u2i` or `i2i`")
        if self.loss_type not in ("cross_entropy", "focal", "bpr", "max_margin"):
            raise ValueError(f"unsupported `loss_type`: {self.loss_type}")

    def build_model(self):
        return GraphSageModel(
            self.paradigm,
            self.data_info,
            self.embed_size,
            self.batch_size,
            self.num_layers,
            self.dropout_rate,
        )

    def get_user_repr(self, users, sparse_indices, dense_values):
        user_feats = feat_to_tensor(users, sparse_indices, dense_values, self.device)
        return self.torch_model.user_repr(*user_feats)

    def sample_neighbors(self, items):
        nodes = items
        tensor_neighbors, tensor_offsets = [], []
        tensor_neighbor_sparse_indices, tensor_neighbor_dense_values = [], []
        for _ in range(self.num_layers):
            neighbors, offsets = bipartite_neighbors(
                nodes,
                self.data_info.user_consumed,
                self.data_info.item_consumed,
                self.num_neighbors,
            )

            (
                neighbor_tensor,
                neighbor_sparse_indices,
                neighbor_dense_values,
            ) = item_unique_to_tensor(neighbors, self.data_info, self.device)
            tensor_neighbors.append(neighbor_tensor)
            tensor_neighbor_sparse_indices.append(neighbor_sparse_indices)
            tensor_neighbor_dense_values.append(neighbor_dense_values)
            tensor_offsets.append(
                torch.tensor(offsets, dtype=torch.long, device=self.device)
            )
            nodes = neighbors
        return (
            tensor_neighbors,
            tensor_neighbor_sparse_indices,
            tensor_neighbor_dense_values,
            tensor_offsets,
        )

    def get_item_repr(self, items, sparse_indices=None, dense_values=None, **_):
        (
            tensor_neighbors,
            tensor_neighbor_sparse_indices,
            tensor_neighbor_dense_values,
            tensor_offsets,
        ) = self.sample_neighbors(items)

        if sparse_indices is not None or dense_values is not None:
            item_tensor, item_sparse_indices, item_dense_values = feat_to_tensor(
                items, sparse_indices, dense_values, self.device
            )
        else:
            item_tensor, item_sparse_indices, item_dense_values = item_unique_to_tensor(
                items, self.data_info, self.device
            )
        return self.torch_model(
            item_tensor,
            item_sparse_indices,
            item_dense_values,
            tensor_neighbors,
            tensor_neighbor_sparse_indices,
            tensor_neighbor_dense_values,
            tensor_offsets,
        )

    @torch.no_grad()
    def set_embeddings(self):
        self.torch_model.eval()
        all_items = list(range(self.n_items))
        item_embed = []
        for i in tqdm(range(0, self.n_items, self.batch_size), desc="item embedding"):
            batch_items = all_items[i : i + self.batch_size]
            item_reprs = self.get_item_repr(batch_items)
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
