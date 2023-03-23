import numpy as np
import torch
from scipy import sparse as ssp
from torch import nn as nn


class LightGCNModel(nn.Module):
    def __init__(
        self,
        n_users,
        n_items,
        embed_size,
        n_layers,
        dropout_rate,
        user_consumed,
        device,
    ):
        super(LightGCNModel, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.user_consumed = user_consumed
        self.device = device
        self.user_init_embeds, self.item_init_embeds = self.init_embeds()
        self.laplacian_matrix = self._build_laplacian_matrix()

    def init_embeds(self):
        user_embeds = nn.Embedding(self.n_users, self.embed_size)
        item_embeds = nn.Embedding(self.n_items, self.embed_size)
        nn.init.normal_(user_embeds.weight, 0.0, 0.1)
        nn.init.normal_(item_embeds.weight, 0.0, 0.1)
        return user_embeds.to(self.device), item_embeds.to(self.device)

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
        adj_matrix = adj_matrix.tocsr()

        row_sum = np.array(adj_matrix.sum(axis=1))
        diag_inv = np.power(row_sum, -0.5).flatten()
        diag_inv[np.isinf(diag_inv)] = 0.0
        diag_matrix_inv = ssp.diags(diag_inv)

        coo = diag_matrix_inv.dot(adj_matrix).dot(diag_matrix_inv).tocoo()
        indices = torch.from_numpy(np.array([coo.row, coo.col]))
        values = torch.from_numpy(coo.data)
        laplacian_matrix = torch.sparse_coo_tensor(
            indices, values, coo.shape, dtype=torch.float32, device=self.device
        )
        return laplacian_matrix

    def forward(self, use_dropout):
        return self.embedding_propagation(use_dropout=use_dropout)

    def embedding_propagation(self, use_dropout):
        if use_dropout and self.dropout_rate > 0:
            laplacian_norm = self.sparse_dropout(
                self.laplacian_matrix, self.laplacian_matrix._nnz()
            )
        else:
            laplacian_norm = self.laplacian_matrix

        all_embeddings = [
            torch.cat(
                [self.user_init_embeds.weight, self.item_init_embeds.weight], dim=0
            )
        ]
        for _ in range(self.n_layers):
            layer_embed = torch.sparse.mm(laplacian_norm, all_embeddings[-1])
            all_embeddings.append(layer_embed)

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_embeds, item_embeds = torch.split(
            all_embeddings, [self.n_users, self.n_items]
        )
        return user_embeds, item_embeds

    def sparse_dropout(self, x, noise_shape):
        keep_prob = 1 - self.dropout_rate
        random_tensor = (torch.rand(noise_shape) + keep_prob).to(x.device)
        dropout_mask = torch.floor(random_tensor).bool()
        indices = x._indices()[:, dropout_mask]
        values = x._values()[dropout_mask] / keep_prob
        return torch.sparse_coo_tensor(indices, values, x.shape, device=x.device)
