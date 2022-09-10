"""

Reference: Xiang Wang et al. "Neural Graph Collaborative Filtering"
           (https://arxiv.org/pdf/1905.08108.pdf)

author: massquantity

"""
import os

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from ..bases import EmbedBase
from ..evaluation import print_metrics
from ..utils.misc import time_block, colorize
from ..utils.sampling import PairwiseSampling
from ..utils.save_load import save_params


class NGCF(EmbedBase):
    def __init__(
        self,
        task,
        data_info,
        embed_size=16,
        n_epochs=20,
        lr=0.01,
        lr_decay=False,
        reg=None,
        batch_size=256,
        num_neg=1,
        node_dropout=None,
        message_dropout=None,
        hidden_units="64,64,64",
        seed=42,
        k=10,
        eval_batch_size=8192,
        eval_user_num=None,
        device=torch.device("cpu"),
        lower_upper_bound=None,
        with_training=True,
    ):
        super().__init__(task, data_info, embed_size, lower_upper_bound)

        assert self.task == "ranking", "NGCF is only suitable for ranking"
        self.all_args = locals()
        self.n_epochs = n_epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.reg = reg
        self.batch_size = batch_size
        self.num_neg = num_neg
        self.node_dropout = node_dropout
        self.message_dropout = message_dropout
        self.hidden_units = list(eval(hidden_units))
        self.seed = seed
        self.k = k
        self.eval_batch_size = eval_batch_size
        self.eval_user_num = eval_user_num
        self.device = device
        if with_training:
            self.model = self._build_model()

    def _build_model(self):
        return NGCFModel(
            self.n_users,
            self.n_items,
            self.embed_size,
            self.hidden_units,
            self.node_dropout,
            self.message_dropout,
            self.user_consumed,
            self.device,
        )

    def fit(
        self,
        train_data,
        verbose=1,
        shuffle=True,
        eval_data=None,
        metrics=None,
        **kwargs,
    ):
        self.show_start_time()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        data_generator = PairwiseSampling(train_data, self.data_info, self.num_neg)
        for epoch in range(1, self.n_epochs + 1):
            with time_block(f"Epoch {epoch}", verbose):
                self.model.train()
                train_total_loss = []
                for (user, item_pos, item_neg) in data_generator(
                    shuffle, self.batch_size
                ):
                    user_embeds, pos_item_embeds, neg_item_embeds = self.model(
                        user, item_pos, item_neg, use_dropout=True
                    )
                    loss = self.bpr_loss(user_embeds, pos_item_embeds, neg_item_embeds)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_total_loss.append(loss.detach().cpu().item())

            if verbose > 1:
                train_loss_str = "train_loss: " + str(
                    round(float(np.mean(train_total_loss)), 4)
                )
                print(f"\t {colorize(train_loss_str, 'green')}")
                # for evaluation
                self.set_embeddings()
                print_metrics(
                    model=self,
                    eval_data=eval_data,
                    metrics=metrics,
                    eval_batch_size=self.eval_batch_size,
                    k=self.k,
                    sample_user_num=self.eval_user_num,
                    seed=self.seed,
                )
                print("=" * 30)

        # for prediction and recommendation
        self.set_embeddings()
        self.assign_embedding_oov()

    def bpr_loss(self, user_embeds, pos_item_embeds, neg_item_embeds):
        pos_scores = torch.sum(torch.mul(user_embeds, pos_item_embeds), axis=1)
        neg_scores = torch.sum(torch.mul(user_embeds, neg_item_embeds), axis=1)
        log_sigmoid = F.logsigmoid(pos_scores - neg_scores)
        loss = torch.negative(torch.mean(log_sigmoid))
        if self.reg:
            embed_reg = (
                torch.linalg.norm(user_embeds).pow(2)
                + torch.linalg.norm(pos_item_embeds).pow(2)
                + torch.linalg.norm(neg_item_embeds).pow(2)
            ) / 2
            loss += (self.reg * embed_reg) / float(len(user_embeds))
        return loss

    @torch.no_grad()
    def set_embeddings(self):
        self.model.eval()
        embeddings = self.model.embedding_propagation(use_dropout=False)
        self.user_embed = embeddings[0].numpy()
        self.item_embed = embeddings[1].numpy()

    def save(self, path, model_name, manual=True, inference_only=False):
        if not os.path.isdir(path):
            print(f"file folder {path} doesn't exists, creating a new one...")
            os.makedirs(path)
        save_params(self, path, model_name)
        variable_path = os.path.join(path, model_name)
        np.savez_compressed(
            variable_path, user_embed=self.user_embed, item_embed=self.item_embed
        )


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
        device=torch.device("cpu"),
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
        layers = [self.embed_size] + self.layers
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
        return embedding_dict, weight_dict

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
        adj_matrix = adj_matrix.tocsr() + scipy.sparse.eye(adj_matrix.shape[0])

        row_sum = np.array(adj_matrix.sum(axis=1))
        diag_inv = np.power(row_sum, -1).flatten()
        diag_inv[np.isinf(diag_inv)] = 0.0
        diag_matrix_inv = scipy.sparse.diags(diag_inv)

        coo = diag_matrix_inv.dot(adj_matrix).tocoo()
        indices = torch.LongTensor([coo.row, coo.col])
        values = torch.from_numpy(coo.data)
        laplacian_matrix = torch.sparse_coo_tensor(
            indices, values, coo.shape, dtype=torch.float32, device=self.device
        )
        return laplacian_matrix

    def forward(self, users, pos_items, neg_items, use_dropout):
        user_embeds, item_embeds = self.embedding_propagation(use_dropout=use_dropout)
        return user_embeds[users], item_embeds[pos_items], item_embeds[neg_items]

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
