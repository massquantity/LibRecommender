import math
from statistics import mean

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from .trainer import BaseTrainer
from ..batch import get_batch_loader
from ..batch.batch_unit import PairwiseBatch, PointwiseBatch
from ..evaluation import print_metrics
from ..graph import compute_i2i_edge_scores, compute_u2i_edge_scores
from ..graph.message import Message
from ..torchops import (
    binary_cross_entropy_loss,
    bpr_loss,
    compute_pair_scores,
    focal_loss,
    max_margin_loss,
    pairwise_bce_loss,
    pairwise_focal_loss,
)
from ..utils.misc import colorize, time_block


class TorchTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        task,
        loss_type,
        n_epochs,
        lr,
        lr_decay,
        epsilon,
        amsgrad,
        reg,
        batch_size,
        sampler,
        num_neg,
        margin,
        device,
    ):
        super().__init__(
            model,
            task,
            loss_type,
            n_epochs,
            lr,
            lr_decay,
            epsilon,
            batch_size,
            sampler,
            num_neg,
        )
        self.amsgrad = amsgrad
        self.reg = reg or 0
        self.margin = margin
        self.sampler = sampler
        self.device = device
        self.torch_model = model.torch_model
        self.optimizer = Adam(
            params=self.torch_model.parameters(),
            lr=self.lr,
            eps=self.epsilon,
            weight_decay=self.reg,
            amsgrad=self.amsgrad,
        )
        # lr_schedular based on paper SGDR: https://arxiv.org/abs/1608.03983
        self.lr_scheduler = (
            CosineAnnealingWarmRestarts(self.optimizer, T_0=1, T_mult=2)
            if self.lr_decay
            else None
        )

    def run(
        self,
        train_data,
        neg_sampling,
        verbose,
        shuffle,
        eval_data,
        metrics,
        k,
        eval_batch_size,
        eval_user_num,
        num_workers,
    ):
        self._check_params()
        data_loader = get_batch_loader(
            self.model, train_data, neg_sampling, self.batch_size, shuffle, num_workers
        )
        n_batches = math.ceil(len(train_data) / self.batch_size)
        for epoch in range(1, self.n_epochs + 1):
            if self.lr_decay:
                print(
                    f"With lr_decay, epoch {epoch} learning rate: "
                    f"{self.optimizer.param_groups[0]['lr']}"
                )
            with time_block(f"Epoch {epoch}", verbose):
                self.torch_model.train()
                train_total_loss = []
                for i, batch_data in enumerate(tqdm(data_loader, desc="train")):
                    loss = self._compute_loss(batch_data)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        # noinspection PyTypeChecker
                        self.lr_scheduler.step(epoch + i / n_batches)
                    train_total_loss.append(loss.detach().cpu().item())

            if verbose > 1:
                train_loss_str = f"train_loss: {round(mean(train_total_loss), 4)}"
                print(f"\t {colorize(train_loss_str, 'green')}")
                # get embedding for evaluation
                self.model.set_embeddings()
                print_metrics(
                    model=self.model,
                    neg_sampling=neg_sampling,
                    eval_data=eval_data,
                    metrics=metrics,
                    eval_batch_size=eval_batch_size,
                    k=k,
                    sample_user_num=eval_user_num,
                    seed=self.model.seed,
                )
                print("=" * 30)

    def _compute_loss(self, data):
        if "cpu" not in self.device.type:  # pragma: no cover
            data.to_device(self.device)
        user_embeds, item_embeds = self.torch_model(use_dropout=True)
        if isinstance(data, PointwiseBatch):
            users, items = data.users, data.items
            users, items = user_embeds[users], item_embeds[items]
            logits = torch.sum(torch.mul(users, items), dim=1)
            labels = torch.as_tensor(data.labels, dtype=torch.float, device=self.device)
            if self.loss_type == "cross_entropy":
                return binary_cross_entropy_loss(logits, labels)
            else:
                return focal_loss(logits, labels)
        elif isinstance(data, PairwiseBatch):
            users = user_embeds[data.queries]
            items_pos, items_neg = data.item_pairs[0], data.item_pairs[1]
            items_pos, items_neg = item_embeds[items_pos], item_embeds[items_neg]
            pos_scores, neg_scores = compute_pair_scores(users, items_pos, items_neg)
            if self.loss_type == "bpr":
                return bpr_loss(pos_scores, neg_scores)
            else:
                return max_margin_loss(pos_scores, neg_scores, self.margin)


class GraphTrainer(TorchTrainer):
    def __init__(
        self,
        model,
        task,
        loss_type,
        n_epochs,
        lr,
        lr_decay,
        epsilon,
        amsgrad,
        reg,
        batch_size,
        sampler,
        num_neg,
        margin,
        device,
    ):
        super().__init__(
            model,
            task,
            loss_type,
            n_epochs,
            lr,
            lr_decay,
            epsilon,
            amsgrad,
            reg,
            batch_size,
            sampler,
            num_neg,
            margin,
            device,
        )
        self.paradigm = model.paradigm

    def _check_params(self):
        n_items = self.model.data_info.n_items
        assert 0 < self.num_neg < n_items, (
            f"`num_neg` should be positive and smaller than total items, "
            f"got {self.num_neg}, {n_items}"
        )
        if self.paradigm == "u2i" and self.sampler not in (
            "random",
            "unconsumed",
            "popular",
        ):
            raise ValueError(
                f"`sampler` must be one of (`random`, `unconsumed`, `popular`) "
                f"for u2i, got {self.sampler}"
            )
        if self.paradigm == "i2i" and self.sampler not in (
            "random",
            "out-batch",
            "popular",
        ):
            raise ValueError(
                f"`sampler` must be one of (`random`, `out-batch`, `popular`) "
                f"for i2i, got {self.sampler}"
            )

    def _to_device(self, data):  # pragma: no cover
        if "cpu" in self.device.type:
            return data
        device_data = []
        for d in data:
            if isinstance(d, Message):
                d = d.to_device(self.device)
            else:
                d = d.to(self.device)
            device_data.append(d)
        return device_data

    def _compute_loss(self, data):
        if self.model.use_dgl:
            if self.paradigm == "u2i":
                user_data, item_data, pos_graph, neg_graph = self._to_device(data)
                user_reprs = self.model.get_user_repr(user_data)
                item_reprs = self.model.get_item_repr(item_data)
                pos_scores = compute_u2i_edge_scores(pos_graph, user_reprs, item_reprs)
                neg_scores = compute_u2i_edge_scores(neg_graph, user_reprs, item_reprs)
            else:
                item_data, pos_graph, neg_graph = self._to_device(data)
                item_reprs = self.model.get_item_repr(item_data)
                pos_scores = compute_i2i_edge_scores(pos_graph, item_reprs)
                neg_scores = compute_i2i_edge_scores(neg_graph, item_reprs)
            if self.loss_type in ("bpr", "max_margin") and self.num_neg > 1:
                pos_scores = pos_scores.repeat_interleave(self.num_neg)
            return self._loss_scores(pos_scores, neg_scores)
        else:
            if self.paradigm == "u2i":
                user_data, item_pos_data, item_neg_data = self._to_device(data)
                query_reprs = self.model.get_user_repr(user_data)
            else:
                item_data, item_pos_data, item_neg_data = self._to_device(data)
                query_reprs = self.model.get_item_repr(item_data)
            item_pos_reprs = self.model.get_item_repr(item_pos_data)
            item_neg_reprs = self.model.get_item_repr(item_neg_data)
            repeat_positives = (
                True if self.loss_type in ("bpr", "max_margin") else False
            )
            pos_scores, neg_scores = compute_pair_scores(
                query_reprs, item_pos_reprs, item_neg_reprs, repeat_positives
            )
            return self._loss_scores(pos_scores, neg_scores)

    def _loss_scores(self, pos_scores, neg_scores):
        if self.loss_type in ("cross_entropy", "focal"):
            loss_func = (
                pairwise_bce_loss
                if self.loss_type == "cross_entropy"
                else pairwise_focal_loss
            )
            mean = True if self.paradigm == "u2i" else False
            return loss_func(pos_scores, neg_scores, mean=mean)
        if self.loss_type == "bpr":
            return bpr_loss(pos_scores, neg_scores)
        else:
            return max_margin_loss(pos_scores, neg_scores, self.margin)
