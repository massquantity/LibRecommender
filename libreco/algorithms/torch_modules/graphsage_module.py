import torch
from torch import nn
from torch.nn import functional as F

from ...utils.validate import (
    check_dense_values,
    check_sparse_indices,
    dense_field_size,
    sparse_feat_size,
)


class GraphSageModel(nn.Module):
    def __init__(
        self,
        paradigm,
        data_info,
        embed_size,
        batch_size,
        num_layers,
        dropout_rate,
    ):
        super(GraphSageModel, self).__init__()
        self.paradigm = paradigm
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_rate)
        self.item_embeds = nn.Embedding(data_info.n_items, embed_size)
        item_input_dim = (len(data_info.item_col) + 1) * embed_size
        self.item_proj = nn.Linear(item_input_dim, embed_size)
        self.item_dense_col_indices = data_info.item_dense_col.index
        self.w_linears = nn.ModuleList(
            [nn.Linear(embed_size * 2, embed_size) for _ in range(num_layers)]
        )
        if paradigm == "u2i":
            self.user_embeds = nn.Embedding(data_info.n_users, embed_size)
            user_input_dim = (len(data_info.user_col) + 1) * embed_size
            self.user_proj = nn.Linear(user_input_dim, embed_size)
            self.user_dense_col_indices = data_info.user_dense_col.index
        self.sparse = check_sparse_indices(data_info)
        self.dense = check_dense_values(data_info)
        if self.sparse:
            self.sparse_embeds = nn.Embedding(sparse_feat_size(data_info), embed_size)
        if self.dense:
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
        nn.init.xavier_uniform_(self.item_embeds.weight)
        nn.init.xavier_uniform_(self.item_proj.weight)
        nn.init.zeros_(self.item_proj.bias)
        for i, w in enumerate(self.w_linears):
            if i == self.num_layers - 1:
                nn.init.xavier_uniform_(w.weight)
            else:
                nn.init.xavier_uniform_(w.weight, gain=gain)
            nn.init.zeros_(w.bias)
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
        offsets,
        weights=None,
    ):
        # paper author's implementation: https://github.com/williamleif/GraphSAGE/blob/master/graphsage/models.py#L299
        hidden = [
            self.get_raw_features(items, sparse_indices, dense_values, is_user=False)
        ]
        for n, s, d in zip(neighbors, neighbor_sparse_indices, neighbor_dense_values):
            hidden.append(self.get_raw_features(n, s, d, is_user=False))
        for layer in range(self.num_layers):
            w_linear = self.w_linears[layer]
            next_hidden = []
            depth = self.num_layers - layer
            for k in range(depth):
                current_embeds = self.dropout(hidden[k])
                neighbor_embeds = self.dropout(hidden[k + 1])
                mean_neighbors = F.embedding_bag(
                    torch.arange(neighbor_embeds.shape[0]).to(neighbor_embeds.device),
                    neighbor_embeds,
                    offsets=offsets[k],
                    mode="mean",
                )
                h = torch.cat([current_embeds, mean_neighbors], dim=1)
                if layer == self.num_layers - 1:
                    h = w_linear(h)
                else:
                    h = F.relu(w_linear(h))
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0]

    def user_repr(self, users, sparse_indices, dense_values):
        return self.get_raw_features(users, sparse_indices, dense_values, is_user=True)

    def get_raw_features(self, ids, sparse_indices, dense_values, is_user):
        concat_features = []
        if sparse_indices is not None:
            sparse_feature = self.sparse_embeds(sparse_indices)
            concat_features.append(sparse_feature.flatten(start_dim=1))
        if dense_values is not None:
            batch_size = dense_values.shape[0]
            # B * F_dense * K
            index = (
                self.user_dense_col_indices if is_user else self.item_dense_col_indices
            )
            dense_embeds = self.dense_embeds[index].repeat(batch_size, 1, 1)
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


class GraphSageDGLModel(GraphSageModel):
    def __init__(
        self,
        paradigm,
        data_info,
        embed_size,
        batch_size,
        num_layers,
        dropout_rate,
        aggregator_type,
    ):
        # noinspection PyUnresolvedReferences
        from dgl.nn import SAGEConv

        super().__init__(
            paradigm, data_info, embed_size, batch_size, num_layers, dropout_rate
        )
        self.layers = nn.ModuleList(
            [
                SAGEConv(
                    in_feats=embed_size,
                    out_feats=embed_size,
                    aggregator_type=aggregator_type,
                    feat_drop=dropout_rate,
                    bias=True,
                    norm=None,
                    activation=None,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, blocks, nodes, sparse_indices, dense_values, *_):
        h = self.get_raw_features(nodes, sparse_indices, dense_values, is_user=False)
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i != self.num_layers - 1:
                h = F.relu(h)
        return h
