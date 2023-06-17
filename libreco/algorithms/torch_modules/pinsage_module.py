import torch
from torch import nn
from torch.nn import functional as F

from .graphsage_module import GraphSageModelBase


class PinSageModelBase(GraphSageModelBase):
    def __init__(self, paradigm, data_info, embed_size, batch_size, num_layers):
        super().__init__(paradigm, data_info, embed_size, batch_size, num_layers)
        self.G1 = nn.Linear(embed_size, embed_size)
        self.G2 = nn.Linear(embed_size, embed_size, bias=False)
        if paradigm == "u2i":
            self.U1 = nn.Linear(embed_size, embed_size)
            self.U2 = nn.Linear(embed_size, embed_size, bias=False)

    def init_parameters(self):
        super().init_parameters()
        gain = nn.init.calculate_gain("relu")
        if self.paradigm == "u2i":
            nn.init.xavier_uniform_(self.U1.weight, gain=gain)
            nn.init.zeros_(self.U1.bias)
            nn.init.xavier_uniform_(self.U2.weight)
        nn.init.xavier_uniform_(self.G1.weight, gain=gain)
        nn.init.zeros_(self.G1.bias)
        nn.init.xavier_uniform_(self.G2.weight)

    def user_repr(self, users, sparse_indices, dense_values):
        raw_features = self.get_raw_features(
            users, sparse_indices, dense_values, is_user=True
        )
        return self.U2(F.relu(self.U1(raw_features)))


class PinSageModel(PinSageModelBase):
    def __init__(
        self, paradigm, data_info, embed_size, batch_size, num_layers, dropout_rate
    ):
        super().__init__(paradigm, data_info, embed_size, batch_size, num_layers)
        self.dropout = nn.Dropout(dropout_rate)
        self.q_linears = nn.ModuleList(
            [nn.Linear(embed_size, embed_size) for _ in range(num_layers)]
        )
        self.w_linears = nn.ModuleList(
            [nn.Linear(embed_size * 2, embed_size) for _ in range(num_layers)]
        )
        self.init_parameters()

    def init_parameters(self):
        super().init_parameters()
        gain = nn.init.calculate_gain("relu")
        for q, w in zip(self.q_linears, self.w_linears):
            nn.init.xavier_uniform_(q.weight, gain=gain)
            nn.init.zeros_(q.bias)
            nn.init.xavier_uniform_(w.weight, gain=gain)
            nn.init.zeros_(w.bias)

    # graphsage paper author's implementation: https://github.com/williamleif/GraphSAGE/blob/master/graphsage/models.py#L299
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


class PinSageDGLModel(PinSageModelBase):
    def __init__(
        self, paradigm, data_info, embed_size, batch_size, num_layers, dropout_rate
    ):
        super().__init__(paradigm, data_info, embed_size, batch_size, num_layers)
        self.layers = nn.ModuleList(
            [PinSageConv(embed_size, dropout_rate) for _ in range(num_layers)]
        )
        self.init_parameters()

    def forward(self, blocks, nodes, sparse_indices, dense_values, *_):
        h = self.get_raw_features(nodes, sparse_indices, dense_values, is_user=False)
        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)  # h_src[:blocks[-1].num_dst_nodes()]
        return self.final_linear(h)

    def final_linear(self, x):
        return self.G2(F.relu(self.G1(x)))


class PinSageConv(nn.Module):
    def __init__(self, embed_size, dropout_rate):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.q_linear = nn.Linear(embed_size, embed_size)
        self.w_linear = nn.Linear(embed_size * 2, embed_size)
        self.init_parameters()

    def init_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.q_linear.weight, gain=gain)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.w_linear.weight, gain=gain)
        nn.init.zeros_(self.w_linear.bias)

    def forward(self, graph, h_src):
        import dgl.function as dfn

        with graph.local_scope():
            h_dst = h_src[: graph.num_dst_nodes()]
            graph.srcdata["n"] = self.dropout(F.relu(self.q_linear(h_src)))
            graph.edata["w"] = graph.edata["weights"].float()
            graph.update_all(dfn.u_mul_e("n", "w", "m"), dfn.sum("m", "n"))
            graph.update_all(dfn.copy_e("w", "m"), dfn.sum("m", "ws"))
            n = graph.dstdata["n"]
            ws = graph.dstdata["ws"].unsqueeze(1).clamp(min=1)
            z = torch.cat([h_dst, n / ws], dim=1)
            z = F.relu(self.w_linear(z))
            z_norm = z.norm(dim=1, keepdim=True)
            default_norm = torch.tensor(1.0).to(z_norm)
            z_norm = torch.where(z_norm == 0, default_norm, z_norm)
            return z / z_norm
