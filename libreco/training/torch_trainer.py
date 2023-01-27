import math
from statistics import mean

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from ..evaluation import print_metrics
from ..graph import build_subgraphs, compute_i2i_edge_scores, compute_u2i_edge_scores
from ..sampling import (
    DataGenerator,
    PairwiseDataGenerator,
    PairwiseRandomWalkGenerator,
    PointwiseDataGenerator,
)
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
from .trainer import BaseTrainer


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
        num_neg,
        margin,
        sampler,
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

    def _check_params(self):
        if self.loss_type in ("bpr", "max_margin") and not self.sampler:
            raise ValueError(f"{self.loss_type} loss must use negative sampling")
        if self.sampler:
            n_items = self.model.data_info.n_items
            assert 0 < self.num_neg < n_items, (
                f"`num_neg` should be positive and smaller than total items, "
                f"got {self.num_neg}, {n_items}"
            )
            if self.sampler not in ("random", "unconsumed", "popular"):
                raise ValueError(
                    f"`sampler` must be one of (`random`, `unconsumed`, `popular`), "
                    f"got {self.sampler}"
                )

    def get_data_generator(self, train_data):
        if not self.sampler:
            return DataGenerator(
                train_data,
                self.model.data_info,
                self.batch_size,
                self.num_neg,
                self.sampler,
                self.model.seed,
                separate_features=False,
            )
        if self.loss_type in ("cross_entropy", "focal"):
            return PointwiseDataGenerator(
                train_data,
                self.model.data_info,
                self.batch_size,
                self.num_neg,
                self.sampler,
                self.model.seed,
                separate_features=True,
            )
        elif self.loss_type in ("bpr", "max_margin"):
            return PairwiseDataGenerator(
                train_data,
                self.model.data_info,
                self.batch_size,
                self.num_neg,
                self.sampler,
                self.model.seed,
                repeat_positives=False,
            )
        else:
            raise ValueError(f"unknown `loss_type`: {self.loss_type}")

    def run(
        self,
        train_data,
        verbose,
        shuffle,
        eval_data,
        metrics,
        k,
        eval_batch_size,
        eval_user_num,
    ):
        self._check_params()
        data_generator = self.get_data_generator(train_data)
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
                for i, data in enumerate(data_generator(shuffle)):
                    loss = self.compute_loss(data)
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
                    eval_data=eval_data,
                    metrics=metrics,
                    eval_batch_size=eval_batch_size,
                    k=k,
                    sample_user_num=eval_user_num,
                    seed=self.model.seed,
                )
                print("=" * 30)

    def compute_loss(self, data):
        user_embeds, item_embeds = self.torch_model(use_dropout=True)
        if self.loss_type in ("cross_entropy", "focal"):
            users = torch.as_tensor(data.users, dtype=torch.long, device=self.device)
            users = user_embeds[users]
            items = torch.as_tensor(data.items, dtype=torch.long, device=self.device)
            items = item_embeds[items]
            logits = torch.sum(torch.mul(users, items), dim=1)
            labels = torch.as_tensor(data.labels, dtype=torch.float, device=self.device)
            if self.loss_type == "cross_entropy":
                return binary_cross_entropy_loss(logits, labels)
            else:
                return focal_loss(logits, labels)
        elif self.loss_type in ("bpr", "max_margin"):
            users = torch.as_tensor(data.queries, dtype=torch.long, device=self.device)
            users = user_embeds[users]
            items_pos = torch.as_tensor(
                data.item_pairs[0], dtype=torch.long, device=self.device
            )
            items_pos = item_embeds[items_pos]
            items_neg = torch.as_tensor(
                data.item_pairs[1], dtype=torch.long, device=self.device
            )
            items_neg = item_embeds[items_neg]
            pos_scores, neg_scores = compute_pair_scores(users, items_pos, items_neg)
            if self.loss_type == "bpr":
                return bpr_loss(pos_scores, neg_scores)
            else:
                return max_margin_loss(pos_scores, neg_scores, self.margin)
        else:
            raise ValueError(f"unknown `loss_type`: {self.loss_type}")


class SageTrainer(TorchTrainer):
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
        num_neg,
        paradigm,
        num_walks,
        walk_len,
        margin,
        sampler,
        start_node,
        focus_start,
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
            num_neg,
            margin,
            sampler,
            device,
        )
        self.data_info = model.data_info
        self.paradigm = paradigm
        self.num_walks = num_walks
        self.walk_len = walk_len
        self.start_node = start_node
        self.focus_start = focus_start

    def _check_params(self):
        if self.loss_type in ("bpr", "max_margin") and not self.sampler:
            raise ValueError(f"{self.loss_type} loss must use negative sampling")
        if self.sampler:
            self._check_sampler()

    def _check_sampler(self):
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
                f"for i2i, got {self.sampler}, consider using u2i for no sampling"
            )

    def get_data_generator(self, train_data):
        if self.paradigm == "u2i":
            if not self.sampler:
                return DataGenerator(
                    train_data,
                    self.model.data_info,
                    self.batch_size,
                    self.num_neg,
                    self.sampler,
                    self.model.seed,
                    separate_features=True,
                )
            else:
                return PairwiseDataGenerator(
                    train_data,
                    self.model.data_info,
                    self.batch_size,
                    self.num_neg,
                    self.sampler,
                    self.model.seed,
                    repeat_positives=False,
                )
        elif self.paradigm == "i2i":
            return PairwiseRandomWalkGenerator(
                train_data,
                self.data_info,
                self.batch_size,
                self.num_neg,
                self.num_walks,
                self.walk_len,
                self.sampler,
                self.model.seed,
                repeat_positives=False,
                start_nodes=self.start_node,
                focus_start=self.focus_start,
            )

    def compute_loss(self, data):
        if (
            self.paradigm == "u2i"
            and not self.sampler
            and self.loss_type in ("cross_entropy", "focal")
        ):
            user_feats = data.users, data.sparse_indices[0], data.dense_values[0]
            user_reprs = self.model.get_user_repr(*user_feats)
            item_feats = data.items, data.sparse_indices[1], data.dense_values[1]
            item_reprs = self.model.get_item_repr(*item_feats)
            logits = torch.sum(torch.mul(user_reprs, item_reprs), dim=1)
            labels = torch.as_tensor(data.labels, dtype=torch.float, device=self.device)
            if self.loss_type == "cross_entropy":
                return binary_cross_entropy_loss(logits, labels)
            else:
                return focal_loss(logits, labels)
        else:
            query_feats = data.queries, data.sparse_indices[0], data.dense_values[0]
            if self.paradigm == "i2i":
                query_reprs = self.model.get_item_repr(
                    *query_feats, items_pos=data.item_pairs[0]
                )
            else:
                query_reprs = self.model.get_user_repr(*query_feats)
            item_pos_feats = (
                data.item_pairs[0],
                data.sparse_indices[1],
                data.dense_values[1],
            )
            item_neg_feats = (
                data.item_pairs[1],
                data.sparse_indices[2],
                data.dense_values[2],
            )
            item_pos_reprs = self.model.get_item_repr(*item_pos_feats)
            item_neg_reprs = self.model.get_item_repr(*item_neg_feats)
            repeat_positives = (
                True if self.loss_type in ("bpr", "max_margin") else False
            )
            pos_scores, neg_scores = compute_pair_scores(
                query_reprs, item_pos_reprs, item_neg_reprs, repeat_positives
            )
            return self._get_loss(pos_scores, neg_scores)

    def _get_loss(self, pos_scores, neg_scores):
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


class SageDGLTrainer(SageTrainer):
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
        num_neg,
        paradigm,
        num_walks,
        walk_len,
        margin,
        sampler,
        start_node,
        focus_start,
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
            num_neg,
            paradigm,
            num_walks,
            walk_len,
            margin,
            sampler,
            start_node,
            focus_start,
            device,
        )

    def _check_params(self):
        if not self.sampler:
            raise ValueError(f"{self.model.model_name} must use sampling")
        self._check_sampler()

    def get_data_generator(self, train_data):
        if self.paradigm == "u2i":
            return PairwiseDataGenerator(
                train_data,
                self.model.data_info,
                self.batch_size,
                self.num_neg,
                self.sampler,
                self.model.seed,
                use_features=False,
                repeat_positives=False,
            )
        elif self.paradigm == "i2i":
            return PairwiseRandomWalkGenerator(
                train_data,
                self.data_info,
                self.batch_size,
                self.num_neg,
                self.num_walks,
                self.walk_len,
                self.sampler,
                self.model.seed,
                use_features=False,
                repeat_positives=False,
                start_nodes=self.start_node,
                focus_start=self.focus_start,
                graph=self.model.hetero_g,
            )

    def compute_loss(self, data):
        import dgl

        # nodes in pos_graph and neg_graph are same, difference is the connected edges
        pos_graph, neg_graph, *target_nodes = build_subgraphs(
            data.queries, data.item_pairs, self.paradigm, self.num_neg, self.device
        )
        if self.paradigm == "u2i":
            # user -> item heterogeneous graph, users on srcdata, items on dstdata
            users, items = pos_graph.srcdata[dgl.NID], pos_graph.dstdata[dgl.NID]
            user_reprs = self.model.get_user_repr(users)
            item_reprs = self.model.get_item_repr(items)
            pos_scores = compute_u2i_edge_scores(pos_graph, user_reprs, item_reprs)
            neg_scores = compute_u2i_edge_scores(neg_graph, user_reprs, item_reprs)
        else:
            # item -> item homogeneous graph, items on all nodes
            items = pos_graph.ndata[dgl.NID]
            item_reprs = self.model.get_item_repr(items, target_nodes)
            pos_scores = compute_i2i_edge_scores(pos_graph, item_reprs)
            neg_scores = compute_i2i_edge_scores(neg_graph, item_reprs)
        if self.loss_type in ("bpr", "max_margin") and self.num_neg > 1:
            pos_scores = pos_scores.repeat_interleave(self.num_neg)
        return self._get_loss(pos_scores, neg_scores)
