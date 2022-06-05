"""

Reference: Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"
           (https://arxiv.org/pdf/2002.02126.pdf)

author: massquantity

"""
import os
from itertools import islice

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from .base import Base
from ..evaluation.evaluate import EvalMixin
from ..utils.misc import time_block, colorize, assign_oov_vector
from ..utils.sampling import PairwiseSampling


class LightGCN(Base, EvalMixin):
    user_variables_np = ["user_embed"]
    item_variables_np = ["item_embed"]

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
            dropout=None,
            n_layers=3,
            seed=42,
            device=torch.device("cpu"),
            lower_upper_bound=None,
    ):
        Base.__init__(self, task, data_info, lower_upper_bound)
        EvalMixin.__init__(self, task, data_info)

        self.task = task
        self.data_info = data_info
        self.embed_size = embed_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.reg = reg
        self.batch_size = batch_size
        self.num_neg = num_neg
        self.dropout = dropout
        self.n_layers = n_layers
        self.n_users = data_info.n_users
        self.n_items = data_info.n_items
        self.seed = seed
        self.user_consumed = data_info.user_consumed
        self.device = device
        self.model = None
        self.user_embed = None
        self.item_embed = None
        self.all_args = locals()

    def _build_model(self):
        self.model = LightGCNModel(
            self.n_users,
            self.n_items,
            self.embed_size,
            self.n_layers,
            self.dropout,
            self.user_consumed,
            self.device
        )

    def fit(
            self,
            train_data,
            verbose=1,
            shuffle=True,
            eval_data=None,
            metrics=None,
            **kwargs
    ):
        assert self.task == "ranking", "LightGCN is only suitable for ranking"
        self.show_start_time()
        self._build_model()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        data_generator = PairwiseSampling(train_data, self.data_info, self.num_neg)
        for epoch in range(1, self.n_epochs + 1):
            with time_block(f"Epoch {epoch}", verbose):
                self.model.train()
                train_total_loss = []
                for (
                        user,
                        item_pos,
                        item_neg
                ) in data_generator(shuffle, self.batch_size):
                    user_embeds, pos_item_embeds, neg_item_embeds = self.model(
                        user, item_pos, item_neg, use_dropout=True
                    )
                    loss = self.bpr_loss(
                        user_embeds,
                        pos_item_embeds,
                        neg_item_embeds,
                        user,
                        item_pos,
                        item_neg
                    )
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
                self._set_latent_vectors()
                self.print_metrics(eval_data=eval_data, metrics=metrics, **kwargs)
                print("=" * 30)

        # for prediction and recommendation
        self._set_latent_vectors()
        assign_oov_vector(self)

    def bpr_loss(
            self,
            user_embeds,
            pos_item_embeds,
            neg_item_embeds,
            user,
            item_pos,
            item_neg
    ):
        pos_scores = torch.sum(torch.mul(user_embeds, pos_item_embeds), axis=1)
        neg_scores = torch.sum(torch.mul(user_embeds, neg_item_embeds), axis=1)
        log_sigmoid = F.logsigmoid(pos_scores - neg_scores)
        loss = torch.negative(torch.mean(log_sigmoid))
        if self.reg:
            user_reg = self.model.get_embed(user, "user")
            item_pos_reg = self.model.get_embed(item_pos, "item")
            item_neg_reg = self.model.get_embed(item_neg, "item")
            embed_reg = (torch.linalg.norm(user_reg).pow(2)
                         + torch.linalg.norm(item_pos_reg).pow(2)
                         + torch.linalg.norm(item_neg_reg).pow(2)) / 2
            loss += ((self.reg * embed_reg) / float(len(user_embeds)))
        return loss

    def predict(self, user, item, cold_start="average", inner_id=False):
        user, item = self.convert_id(user, item, inner_id)
        unknown_num, unknown_index, user, item = self._check_unknown(user, item)
        preds = np.sum(
            np.multiply(self.user_embed[user], self.item_embed[item]),
            axis=1
        )
        preds = 1 / (1 + np.exp(-preds))

        if unknown_num > 0 and cold_start == "popular":
            preds[unknown_index] = self.default_prediction
        return preds

    def recommend_user(self, user, n_rec, cold_start="average", inner_id=False):
        user_id = self._check_unknown_user(user, inner_id)
        if user_id is None:
            if cold_start == "average":
                user_id = self.n_users
            elif cold_start == "popular":
                return self.popular_recommends(inner_id, n_rec)
            else:
                raise ValueError(user)

        consumed = set(self.user_consumed[user_id])
        count = n_rec + len(consumed)
        recos = self.user_embed[user_id] @ self.item_embed.T
        recos = 1 / (1 + np.exp(-recos))

        ids = np.argpartition(recos, -count)[-count:]
        rank = sorted(zip(ids, recos[ids]), key=lambda x: -x[1])
        recs_and_scores = islice(
            (rec if inner_id else (self.data_info.id2item[rec[0]], rec[1])
             for rec in rank if rec[0] not in consumed),
            n_rec
        )
        return list(recs_and_scores)

    @torch.no_grad()
    def _set_latent_vectors(self):
        self.model.eval()
        embeddings = self.model.embedding_propagation(use_dropout=False)
        self.user_embed = embeddings[0].numpy()
        self.item_embed = embeddings[1].numpy()

    def save(self, path, model_name, manual=True, inference_only=False):
        if not os.path.isdir(path):
            print(f"file folder {path} doesn't exists, creating a new one...")
            os.makedirs(path)
        self.save_params(path)
        variable_path = os.path.join(path, model_name)
        np.savez_compressed(variable_path,
                            user_embed=self.user_embed,
                            item_embed=self.item_embed)

    @classmethod
    def load(cls, path, model_name, data_info, manual=True):
        variable_path = os.path.join(path, f"{model_name}.npz")
        variables = np.load(variable_path)
        hparams = cls.load_params(path, data_info)
        model = cls(**hparams)
        model.user_embed = variables["user_embed"]
        model.item_embed = variables["item_embed"]
        return model


class LightGCNModel(nn.Module):
    def __init__(
            self,
            n_users,
            n_items,
            embed_size,
            n_layers,
            dropout,
            user_consumed,
            device=torch.device("cpu")
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
            R[u, items] = 1.
        R = R.tolil()

        adj_matrix = scipy.sparse.lil_matrix(
            (self.n_users + self.n_items, self.n_items + self.n_users),
            dtype=np.float32
        )
        adj_matrix[:self.n_users, self.n_users:] = R
        adj_matrix[self.n_users:, :self.n_users] = R.T
        adj_matrix = adj_matrix.tocsr()

        row_sum = np.array(adj_matrix.sum(axis=1))
        diag_inv = np.power(row_sum, -0.5).flatten()
        diag_inv[np.isinf(diag_inv)] = 0.
        diag_matrix_inv = scipy.sparse.diags(diag_inv)

        coo = diag_matrix_inv.dot(adj_matrix).dot(diag_matrix_inv).tocoo()
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
        if use_dropout and self.dropout > 0:
            laplacian_norm = self.sparse_dropout(
                self.laplacian_matrix, self.laplacian_matrix._nnz()
            )
        else:
            laplacian_norm = self.laplacian_matrix

        all_embeddings = [
            torch.cat([self.user_init_embeds.weight,
                       self.item_init_embeds.weight], dim=0)
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

    def get_embed(self, indices, embed_type):
        if embed_type == "user":
            return self.user_init_embeds(torch.LongTensor(indices))
        elif embed_type == "item":
            return self.item_init_embeds(torch.LongTensor(indices))
