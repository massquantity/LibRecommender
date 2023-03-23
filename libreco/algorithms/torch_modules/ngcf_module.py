import numpy as np
import torch
from scipy import sparse as ssp
from torch import nn as nn
from torch.nn import functional as F


class NGCFModel(nn.Module):
    def __init__(
        self,
        n_users,
        n_items,
        embed_size,
        layers,
        node_dropout,
        message_dropout,
        user_consumed,
        device,
    ):
        super(NGCFModel, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_size = embed_size
        self.layers = layers
        self.node_dropout = node_dropout
        self.message_dropout = message_dropout
        self.user_consumed = user_consumed
        self.device = device
        self.embedding_dict, self.weight_dict = self.init_weights()
        self.laplacian_matrix = self._build_laplacian_matrix()

    def init_weights(self):
        embedding_dict = nn.ParameterDict(
            {
                "user_embed": nn.Parameter(
                    nn.init.xavier_uniform_(torch.empty(self.n_users, self.embed_size))
                ),
                "item_embed": nn.Parameter(
                    nn.init.xavier_uniform_(torch.empty(self.n_items, self.embed_size))
                ),
            }
        )

        weight_dict = nn.ParameterDict()
        layers = [self.embed_size, *self.layers]
        for k in range(len(self.layers)):
            weight_dict[f"W_self_{k}"] = nn.Parameter(
                nn.init.xavier_uniform_(torch.empty(layers[k], layers[k + 1]))
            )
            weight_dict[f"b_self_{k}"] = nn.Parameter(
                nn.init.zeros_(torch.empty(1, layers[k + 1]))
            )
            weight_dict[f"W_pair_{k}"] = nn.Parameter(
                nn.init.xavier_uniform_(torch.empty(layers[k], layers[k + 1]))
            )
            weight_dict[f"b_pair_{k}"] = nn.Parameter(
                nn.init.zeros_(torch.empty(1, layers[k + 1]))
            )
        return embedding_dict.to(self.device), weight_dict.to(self.device)

    def _build_laplacian_matrix(self):
        R = ssp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        for u in range(self.n_users):
            items = self.user_consumed[u]
            R[u, items] = 1.0
        R = R.tolil()

        adj_matrix = ssp.lil_matrix(
            (self.n_users + self.n_items, self.n_items + self.n_users), dtype=np.float32
        )
        adj_matrix[: self.n_users, self.n_users :] = R
        adj_matrix[self.n_users :, : self.n_users] = R.T
        adj_matrix = adj_matrix.tocsr() + ssp.eye(adj_matrix.shape[0])

        row_sum = np.array(adj_matrix.sum(axis=1))
        diag_inv = np.power(row_sum, -1).flatten()
        diag_inv[np.isinf(diag_inv)] = 0.0
        diag_matrix_inv = ssp.diags(diag_inv)

        coo = diag_matrix_inv.dot(adj_matrix).tocoo()
        indices = torch.from_numpy(np.array([coo.row, coo.col]))
        values = torch.from_numpy(coo.data)
        laplacian_matrix = torch.sparse_coo_tensor(
            indices, values, coo.shape, dtype=torch.float32, device=self.device
        )
        return laplacian_matrix

    def forward(self, use_dropout):
        return self.embedding_propagation(use_dropout=use_dropout)

    def embedding_propagation(self, use_dropout):
        if use_dropout and self.node_dropout > 0:
            laplacian_norm = self.sparse_dropout(
                self.laplacian_matrix, self.laplacian_matrix._nnz()
            )
        else:
            laplacian_norm = self.laplacian_matrix

        all_embeddings = [
            torch.cat(
                [self.embedding_dict["user_embed"], self.embedding_dict["item_embed"]],
                dim=0,
            )
        ]
        for k in range(len(self.layers)):
            side_embeddings = torch.sparse.mm(laplacian_norm, all_embeddings[-1])
            self_embeddings = (
                torch.matmul(side_embeddings, self.weight_dict[f"W_self_{k}"])
                + self.weight_dict[f"b_self_{k}"]
            )
            pair_embeddings = (
                torch.matmul(
                    torch.mul(side_embeddings, all_embeddings[-1]),
                    self.weight_dict[f"W_pair_{k}"],
                )
                + self.weight_dict[f"b_pair_{k}"]
            )
            embed_messages = F.leaky_relu(
                self_embeddings + pair_embeddings, negative_slope=0.2
            )
            if use_dropout and self.message_dropout > 0:
                embed_messages = F.dropout(embed_messages, p=self.message_dropout)
            norm_embeddings = F.normalize(embed_messages, p=2, dim=1)
            all_embeddings.append(norm_embeddings)

        all_embeddings = torch.cat(all_embeddings, dim=1)
        user_embeds = all_embeddings[: self.n_users]
        item_embeds = all_embeddings[self.n_users :]
        return user_embeds, item_embeds

    def sparse_dropout(self, x, noise_shape):
        keep_prob = 1 - self.node_dropout
        random_tensor = (torch.rand(noise_shape) + keep_prob).to(x.device)
        dropout_mask = torch.floor(random_tensor).bool()
        indices = x._indices()[:, dropout_mask]
        values = x._values()[dropout_mask] / keep_prob
        return torch.sparse_coo_tensor(indices, values, x.shape, device=x.device)
