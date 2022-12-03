"""

Reference: Rex Ying et al. "Graph Convolutional Neural Networks for Web-Scale Recommender Systems"
           (https://arxiv.org/abs/1806.01973)

author: massquantity

"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from ..bases import EmbedBase
from ..sampling.random_walks import bipartite_neighbors_with_weights
from ..torchops.features import (
    feat_to_tensor,
    user_unique_to_tensor,
    item_unique_to_tensor,
)
from ..training.torch_trainer import SageTrainer
from ..utils.validate import (
    check_dense_values,
    check_sparse_indices,
    dense_field_size,
    sparse_feat_size,
)


class PinSage(EmbedBase):
    def __init__(
        self,
        task,
        data_info,
        loss_type="max_margin",
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
        paradigm="i2i",
        remove_edges=False,
        full_repr=True,
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
        self.n_epochs = n_epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.num_neg = num_neg
        self.dropout = dropout
        self.paradigm = paradigm
        self.remove_edges = remove_edges
        self.full_repr = full_repr
        self.num_layers = num_layers
        self.num_neighbors = num_neighbors
        self.num_walks = num_walks
        self.neighbor_walk_len = neighbor_walk_len
        self.sample_walk_len = sample_walk_len
        self.termination_prob = termination_prob
        self.margin = margin
        self.sampler = sampler
        self.start_node = start_node
        self.focus_start = focus_start
        self.k = k
        self.eval_batch_size = eval_batch_size
        self.eval_user_num = eval_user_num
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
                full_repr,
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
        if self.paradigm == "u2i":
            if self.loss_type not in ("cross_entropy", "focal", "bpr", "max_margin"):
                raise ValueError(f"unsupported `loss_type` for u2i: {self.loss_type}")
        elif self.paradigm == "i2i":
            if self.loss_type not in ("cross_entropy", "bpr", "max_margin"):
                raise ValueError(f"unsupported `loss_type` for i2i: {self.loss_type}")

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
            tensor_weights.append(torch.FloatTensor(weights, device=self.device))
            tensor_offsets.append(torch.LongTensor(offsets, device=self.device))
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
                    num_neighbors=100,
                    num_walks=100,
                    walk_length=self.neighbor_walk_len,
                )
                neighbor_tensors = item_unique_to_tensor(
                    neighbors,
                    self.data_info,
                    self.device,
                )
                tensor_neighbors.append(neighbor_tensors[0])
                tensor_neighbor_sparse_indices.append(neighbor_tensors[1])
                tensor_neighbor_dense_values.append(neighbor_tensors[2])
                tensor_weights.append(torch.FloatTensor(weights, device=self.device))
                tensor_offsets.append(torch.LongTensor(offsets, device=self.device))
                nodes = neighbors

            item_feats_tensor = item_unique_to_tensor(
                batch_items, self.data_info, self.device
            )
            item_reprs = self.torch_model(
                *item_feats_tensor,
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


class PinSageModel(nn.Module):
    def __init__(
        self, paradigm, data_info, embed_size, batch_size, num_layers, dropout_rate
    ):
        super(PinSageModel, self).__init__()
        self.paradigm = paradigm
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_rate)
        self.item_embeds = nn.Embedding(data_info.n_items, embed_size)
        item_input_dim = (len(data_info.item_col) + 1) * embed_size
        self.item_proj = nn.Linear(item_input_dim, embed_size)
        self.q_linears = nn.ModuleList(
            [nn.Linear(embed_size, embed_size) for _ in range(num_layers)]
        )
        self.w_linears = nn.ModuleList(
            [nn.Linear(embed_size * 2, embed_size) for _ in range(num_layers)]
        )
        self.G1 = nn.Linear(embed_size, embed_size)
        self.G2 = nn.Linear(embed_size, embed_size, bias=False)
        if paradigm == "u2i":
            self.user_embeds = nn.Embedding(data_info.n_users, embed_size)
            user_input_dim = (len(data_info.user_col) + 1) * embed_size
            self.user_proj = nn.Linear(user_input_dim, embed_size)
            self.U1 = nn.Linear(embed_size, embed_size)
            self.U2 = nn.Linear(embed_size, embed_size, bias=False)
        self.sparse = check_sparse_indices(data_info)
        self.dense = check_dense_values(data_info)
        if self.sparse:
            self.sparse_embeds = nn.Embedding(sparse_feat_size(data_info), embed_size)
        if self.dense:
            # self.dense_embeds = nn.Parameter(
            #    torch.empty(self.batch_size, dense_field_size(data_info), embed_size)
            # )
            self.dense_embeds = nn.Parameter(
                torch.empty(dense_field_size(data_info), embed_size)
            )
        self.init_parameters()

    def init_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if self.paradigm == "u2i":
            nn.init.xavier_uniform_(self.user_embeds.weight)
            nn.init.xavier_uniform_(self.user_proj.weight)
            nn.init.zeros_(self.user_proj.bias)
            nn.init.xavier_uniform_(self.U1.weight, gain=gain)
            nn.init.zeros_(self.U1.bias)
            nn.init.xavier_uniform_(self.U2.weight)
        nn.init.xavier_uniform_(self.item_embeds.weight)
        nn.init.xavier_uniform_(self.item_proj.weight)
        nn.init.zeros_(self.item_proj.bias)
        for q, w in zip(self.q_linears, self.w_linears):
            nn.init.xavier_uniform_(q.weight, gain=gain)
            nn.init.zeros_(q.bias)
            nn.init.xavier_uniform_(w.weight, gain=gain)
            nn.init.zeros_(w.bias)
        nn.init.xavier_uniform_(self.G1.weight, gain=gain)
        nn.init.zeros_(self.G1.bias)
        nn.init.xavier_uniform_(self.G2.weight)
        if self.sparse:
            nn.init.xavier_uniform_(self.sparse_embeds.weight)
        if self.dense:
            nn.init.xavier_uniform_(self.dense_embeds)

    def forward(
        self,
        items,
        sparse_indices,
        dense_values,
        neighbors,
        neighbor_sparse_indices,
        neighbor_dense_values,
        weights,
        offsets,
    ):
        hidden = [
            self.get_raw_features(items, sparse_indices, dense_values, is_user=False)
        ]
        for n, s, d in zip(neighbors, neighbor_sparse_indices, neighbor_dense_values):
            hidden.append(self.get_raw_features(n, s, d, is_user=False))
        for layer in range(self.num_layers):
            q_linear, w_linear = self.q_linears[layer], self.w_linears[layer]
            next_hidden = []
            depth = self.num_layers - layer
            for k in range(depth):
                current_embeds = hidden[k]
                neighbor_embeds = F.relu(q_linear(self.dropout(hidden[k + 1])))
                weighted_neighbors = F.embedding_bag(
                    torch.arange(neighbor_embeds.shape[0]),
                    neighbor_embeds,
                    offsets=offsets[k],
                    per_sample_weights=weights[k],
                    mode="sum",
                )
                z = torch.cat([current_embeds, weighted_neighbors], dim=1)
                z = F.relu(w_linear(self.dropout(z)))
                z_norm = z.norm(dim=1, keepdim=True)
                default_norm = torch.tensor(1.0).to(z_norm)
                z_norm = torch.where(z_norm == 0, default_norm, z_norm)
                next_hidden.append(z / z_norm)
            hidden = next_hidden
        return self.G2(F.relu(self.G1(hidden[0])))

    def user_repr(self, users, sparse_indices, dense_values):
        raw_features = self.get_raw_features(
            users, sparse_indices, dense_values, is_user=True
        )
        return self.U2(F.relu(self.U1(raw_features)))

    def get_raw_features(self, ids, sparse_indices, dense_values, is_user):
        concat_features = []
        if sparse_indices is not None:
            sparse_feature = self.sparse_embeds(sparse_indices)
            concat_features.append(sparse_feature.flatten(start_dim=1))
        if dense_values is not None:
            batch_size = dense_values.shape[0]
            # B * F_dense * K
            dense_embeds = self.dense_embeds.repeat(batch_size, 1, 1)
            # dense_embeds = self.dense_embeds[:b_size] if b_size != self.batch_size else self.dense_embeds
            # B * F_dense * 1
            dense_vals = dense_values.unsqueeze(2)
            dense_feature = torch.mul(dense_embeds, dense_vals)
            concat_features.append(dense_feature.flatten(start_dim=1))

        if is_user:
            concat_features.append(self.user_embeds(ids))
            concat_features = torch.cat(concat_features, dim=1)
            proj_features = self.user_proj(concat_features)
        else:
            concat_features.append(self.item_embeds(ids))
            concat_features = torch.cat(concat_features, dim=1)
            proj_features = self.item_proj(concat_features)
        return proj_features
