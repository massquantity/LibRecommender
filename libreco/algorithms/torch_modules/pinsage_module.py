import torch
from torch import nn
from torch.nn import functional as F

from ...utils.validate import (
    check_dense_values,
    check_sparse_indices,
    dense_field_size,
    sparse_feat_size,
)


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
        self.item_dense_col_indices = data_info.item_dense_col.index
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
            self.user_dense_col_indices = data_info.user_dense_col.index
            self.U1 = nn.Linear(embed_size, embed_size)
            self.U2 = nn.Linear(embed_size, embed_size, bias=False)
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
        offsets,
        weights,
    ):
        # graphsage paper author's implementation: https://github.com/williamleif/GraphSAGE/blob/master/graphsage/models.py#L299
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
                neighbor_embeds = self.dropout(F.relu(q_linear(hidden[k + 1])))
                weighted_neighbors = F.embedding_bag(
                    torch.arange(neighbor_embeds.shape[0]).to(neighbor_embeds.device),
                    neighbor_embeds,
                    offsets=offsets[k],
                    per_sample_weights=weights[k],
                    mode="sum",
                )
                z = torch.cat([current_embeds, weighted_neighbors], dim=1)
                z = F.relu(w_linear(z))
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


class PinSageDGLModel(PinSageModel):
    def __init__(
        self, paradigm, data_info, embed_size, batch_size, num_layers, dropout_rate
    ):
        super().__init__(
            paradigm, data_info, embed_size, batch_size, num_layers, dropout_rate
        )

    def forward(self, blocks, nodes, sparse_indices, dense_values, *_):
        import dgl.function as dfn

        h_src = self.get_raw_features(
            nodes, sparse_indices, dense_values, is_user=False
        )
        for layer, block in enumerate(blocks):
            q_linear, w_linear = self.q_linears[layer], self.w_linears[layer]
            h_dst = h_src[: block.num_dst_nodes()]
            with block.local_scope():
                block.srcdata["n"] = self.dropout(F.relu(q_linear(h_src)))
                block.edata["w"] = block.edata["weights"].float()
                block.update_all(dfn.u_mul_e("n", "w", "m"), dfn.sum("m", "n"))
                block.update_all(dfn.copy_e("w", "m"), dfn.sum("m", "ws"))
                n = block.dstdata["n"]
                ws = block.dstdata["ws"].unsqueeze(1).clamp(min=1)
                z = torch.cat([h_dst, n / ws], dim=1)
                z = F.relu(w_linear(z))
                z_norm = z.norm(dim=1, keepdim=True)
                default_norm = torch.tensor(1.0).to(z_norm)
                z_norm = torch.where(z_norm == 0, default_norm, z_norm)
                z = z / z_norm
            h_src = z
        items = h_src  # h_src[:blocks[-1].num_dst_nodes()]
        return self.G2(F.relu(self.G1(items)))
