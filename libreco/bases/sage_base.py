from dataclasses import astuple
from typing import Union

import numpy as np
import torch
from tqdm import tqdm

from ..bases import EmbedBase
from ..graph import NeighborWalker
from ..graph.message import ItemMessage, ItemMessageDGL, UserMessage
from ..torchops import device_config


class SageBase(EmbedBase):
    """Base class for GraphSage and PinSage.

    Graph neural network algorithms using neighbor sampling and node features.

    See Also
    --------
    ~libreco.algorithms.GraphSage
    ~libreco.algorithms.PinSage
    ~libreco.algorithms.GraphSageDGL
    ~libreco.algorithms.PinSageDGL
    """

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
        device="cuda",
        lower_upper_bound=None,
    ):
        super().__init__(task, data_info, embed_size, lower_upper_bound)

        self.all_args = locals()
        self.loss_type = loss_type
        self.paradigm = paradigm
        self.n_epochs = n_epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        self.reg = reg
        self.batch_size = batch_size
        self.num_neg = num_neg
        self.dropout_rate = dropout_rate
        self.remove_edges = remove_edges
        self.num_layers = num_layers
        self.num_neighbors = num_neighbors
        self.num_walks = num_walks
        self.sample_walk_len = sample_walk_len
        self.margin = margin
        self.sampler = sampler
        self.start_node = start_node
        self.focus_start = focus_start
        self.neighbor_walker = None
        self.seed = seed
        self.device = device_config(device)
        self.use_dgl = "DGL" in self.model_name
        self.torch_model = None
        self._check_params()

    def _check_params(self):
        if self.task != "ranking":
            raise ValueError(f"`{self.model_name}` is only suitable for ranking")
        if self.paradigm not in ("u2i", "i2i"):
            raise ValueError("`paradigm` must either be `u2i` or `i2i`")
        if self.loss_type not in ("cross_entropy", "focal", "bpr", "max_margin"):
            raise ValueError(f"unsupported `loss_type`: {self.loss_type}")
        if self.paradigm == "i2i" and self.start_node not in ("random", "unpopular"):
            raise ValueError("`start_nodes` must either be `random` or `unpopular`")
        if not self.sampler:
            raise ValueError(
                f"`{self.model_name}` must use negative sampling, make sure data "
                f"only contains positive samples when using negative sampling."
            )

    def build_model(self):
        raise NotImplementedError

    def get_user_repr(self, user_data: UserMessage):
        users, sparse_indices, dense_values = astuple(user_data)
        return self.torch_model.user_repr(users, sparse_indices, dense_values)

    def get_item_repr(self, item_data: Union[ItemMessage, ItemMessageDGL]):
        if isinstance(item_data, ItemMessage):
            (
                item,
                sparse_indices,
                dense_values,
                neighbors,
                neighbor_sparse,
                neighbor_dense,
                offsets,
                weights,
            ) = astuple(item_data)
            return self.torch_model(
                item,
                sparse_indices,
                dense_values,
                neighbors,
                neighbor_sparse,
                neighbor_dense,
                offsets,
                weights,
            )
        else:
            blocks, start_nodes, sparse_indices, dense_values = astuple(item_data)
            return self.torch_model(blocks, start_nodes, sparse_indices, dense_values)

    @torch.no_grad()
    def set_embeddings(self):
        assert isinstance(self.neighbor_walker, NeighborWalker)
        self.torch_model.eval()
        item_embed = []
        all_items = list(range(self.n_items))
        for i in tqdm(range(0, self.n_items, self.batch_size), desc="item embedding"):
            batch_items = all_items[i : i + self.batch_size]
            if self.use_dgl:
                batch_items = torch.tensor(batch_items, dtype=torch.long)
            item_data = self.neighbor_walker(batch_items)
            item_data = item_data.to_device(self.device)
            item_reprs = self.get_item_repr(item_data)
            item_embed.append(item_reprs.detach().cpu().numpy())
        self.item_embed = np.concatenate(item_embed, axis=0)
        self.user_embed = self.get_user_embeddings()

    @torch.no_grad()
    def get_user_embeddings(self):
        self.torch_model.eval()
        user_embed = []
        if self.paradigm == "u2i":
            for i in range(0, self.n_users, self.batch_size):
                users = np.arange(i, min(i + self.batch_size, self.n_users))
                user_data = self.neighbor_walker.get_user_feats(users)
                user_data = user_data.to_device(self.device)
                user_reprs = self.get_user_repr(user_data)
                user_embed.append(user_reprs.detach().cpu().numpy())
            return np.concatenate(user_embed, axis=0)
        else:
            for u in range(self.n_users):
                items = self.user_consumed[u]
                user_embed.append(np.mean(self.item_embed[items], axis=0))
                # user_embed.append(self.item_embed[items[-1]])
            return np.array(user_embed)
