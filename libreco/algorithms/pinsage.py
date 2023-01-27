"""

Reference: Rex Ying et al. "Graph Convolutional Neural Networks for Web-Scale Recommender Systems"
           (https://arxiv.org/abs/1806.01973)

author: massquantity

"""
import torch

from ..bases import ModelMeta
from ..sampling import bipartite_neighbors_with_weights
from ..torchops import feat_to_tensor, item_unique_to_tensor
from .graphsage import GraphSage
from .torch_modules import PinSageModel


class PinSage(GraphSage, metaclass=ModelMeta, backend="torch"):
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
        dropout_rate=0.0,
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
        device=torch.device("cpu"),
        lower_upper_bound=None,
        with_training=True,
    ):
        super().__init__(
            task,
            data_info,
            loss_type,
            paradigm,
            embed_size,
            n_epochs,
            lr,
            lr_decay,
            epsilon,
            amsgrad,
            reg,
            batch_size,
            num_neg,
            dropout_rate,
            remove_edges,
            num_layers,
            num_neighbors,
            num_walks,
            sample_walk_len,
            margin,
            sampler,
            start_node,
            focus_start,
            seed,
            device,
            lower_upper_bound,
            with_training,
        )
        self.all_args = locals()
        self.num_walks = num_walks
        self.neighbor_walk_len = neighbor_walk_len
        self.termination_prob = termination_prob

    def _check_params(self):
        if self.task != "ranking":
            raise ValueError("PinSage is only suitable for ranking")
        if self.paradigm not in ("u2i", "i2i"):
            raise ValueError("paradigm must either be `u2i` or `i2i`")
        if self.loss_type not in ("cross_entropy", "focal", "bpr", "max_margin"):
            raise ValueError(f"unsupported `loss_type`: {self.loss_type}")

    def build_model(self):
        return PinSageModel(
            self.paradigm,
            self.data_info,
            self.embed_size,
            self.batch_size,
            self.num_layers,
            self.dropout_rate,
        )

    def sample_neighbors(self, items, item_indices=None, items_pos=None):
        nodes = items
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
                self.termination_prob,
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
            item_indices = item_indices_in_samples
        return (
            tensor_neighbors,
            tensor_neighbor_sparse_indices,
            tensor_neighbor_dense_values,
            tensor_weights,
            tensor_offsets,
        )

    def get_item_repr(
        self, items, sparse_indices=None, dense_values=None, items_pos=None
    ):
        if self.paradigm == "i2i" and self.remove_edges and items_pos is not None:
            item_indices = list(range(len(items)))
        else:
            item_indices = None

        (
            tensor_neighbors,
            tensor_neighbor_sparse_indices,
            tensor_neighbor_dense_values,
            tensor_weights,
            tensor_offsets,
        ) = self.sample_neighbors(items, item_indices, items_pos)

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
            tensor_weights,
            tensor_offsets,
        )
