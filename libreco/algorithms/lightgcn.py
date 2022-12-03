"""

Reference: Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"
           (https://arxiv.org/pdf/2002.02126.pdf)

author: massquantity

"""
import numpy as np
import scipy
import torch
import torch.nn as nn

from ..bases import EmbedBase, ModelMeta
from ..training import TorchTrainer


class LightGCN(EmbedBase, metaclass=ModelMeta, backend="torch"):
    def __init__(
        self,
        task,
        data_info,
        loss_type="bpr",
        embed_size=16,
        n_epochs=20,
        lr=0.01,
        lr_decay=False,
        epsilon=1e-8,
        amsgrad=False,
        reg=None,
        batch_size=256,
        num_neg=1,
        dropout=0.0,
        n_layers=3,
        margin=1.0,
        sampler="random",
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
        self.dropout = dropout
        self.n_layers = n_layers
        self.seed = seed
        self.device = device
        self._check_params()
        if with_training:
            self.torch_model = LightGCNModel(
                self.n_users,
                self.n_items,
                self.embed_size,
                self.n_layers,
                self.dropout,
                self.user_consumed,
                self.device,
            )
            self.trainer = TorchTrainer(
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
                margin,
                sampler,
                k,
                eval_batch_size,
                eval_user_num,
                device,
            )

    def _check_params(self):
        if self.task != "ranking":
            raise ValueError("LightGCN is only suitable for ranking")
        if self.loss_type not in ("cross_entropy", "focal", "bpr", "max_margin"):
            raise ValueError(f"unsupported `loss_type` for LightGCN: {self.loss_type}")

    @torch.no_grad()
    def set_embeddings(self):
        self.torch_model.eval()
        embeddings = self.torch_model.embedding_propagation(use_dropout=False)
        self.user_embed = embeddings[0].numpy()
        self.item_embed = embeddings[1].numpy()


class LightGCNModel(nn.Module):
    def __init__(
        self,
        n_users,
        n_items,
        embed_size,
        n_layers,
        dropout,
        user_consumed,
        device=torch.device("cpu"),
    ):
        super(LightGCNModel, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.user_consumed = user_consumed
        self.device = device
        self.user_init_embeds, self.item_init_embeds = self.init_embeds()
        self.laplacian_matrix = self._build_laplacian_matrix()

    def init_embeds(self):
        user_embeds = nn.Embedding(self.n_users, self.embed_size)
        item_embeds = nn.Embedding(self.n_items, self.embed_size)
        nn.init.normal_(user_embeds.weight, 0.0, 0.1)
        nn.init.normal_(item_embeds.weight, 0.0, 0.1)
        return user_embeds, item_embeds

    # noinspection PyUnresolvedReferences
    def _build_laplacian_matrix(self):
        R = scipy.sparse.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        for u, items in self.user_consumed.items():
            R[u, items] = 1.0
        R = R.tolil()

        adj_matrix = scipy.sparse.lil_matrix(
            (self.n_users + self.n_items, self.n_items + self.n_users), dtype=np.float32
        )
        adj_matrix[: self.n_users, self.n_users :] = R
        adj_matrix[self.n_users :, : self.n_users] = R.T
        adj_matrix = adj_matrix.tocsr()

        row_sum = np.array(adj_matrix.sum(axis=1))
        diag_inv = np.power(row_sum, -0.5).flatten()
        diag_inv[np.isinf(diag_inv)] = 0.0
        diag_matrix_inv = scipy.sparse.diags(diag_inv)

        coo = diag_matrix_inv.dot(adj_matrix).dot(diag_matrix_inv).tocoo()
        indices = torch.LongTensor(np.array([coo.row, coo.col]))
        values = torch.from_numpy(coo.data)
        laplacian_matrix = torch.sparse_coo_tensor(
            indices, values, coo.shape, dtype=torch.float32, device=self.device
        )
        return laplacian_matrix

    def forward(self, use_dropout):
        return self.embedding_propagation(use_dropout=use_dropout)

    def embedding_propagation(self, use_dropout):
        if use_dropout and self.dropout > 0:
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
        keep_prob = 1 - self.dropout
        random_tensor = (torch.rand(noise_shape) + keep_prob).to(x.device)
        dropout_mask = torch.floor(random_tensor).bool()
        indices = x._indices()[:, dropout_mask]
        values = x._values()[dropout_mask] / keep_prob
        return torch.sparse_coo_tensor(indices, values, x.shape, device=x.device)
