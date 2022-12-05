from statistics import mean

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from .trainer import BaseTrainer
from ..evaluation import print_metrics
from ..sampling.data_sampler import (
    DataGenerator,
    PairwiseDataGenerator,
    PairwiseRandomWalkGenerator,
    PointwiseDataGenerator,
)
from ..torchops.features import feat_to_tensor
from ..torchops.loss import (
    binary_cross_entropy_loss,
    bpr_loss,
    focal_loss,
    graphsage_unsupervised_loss,
    max_margin_loss,
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
        num_neg,
        margin,
        sampler,
        k,
        eval_batch_size,
        eval_user_num,
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
            k,
            eval_batch_size,
            eval_user_num,
        )
        self.amsgrad = amsgrad
        self.reg = reg or 0
        self.margin = margin
        self.sampler = sampler
        self.device = device
        self.torch_model = model.torch_model
        self._check_params()
        self.optimizer = Adam(
            params=self.torch_model.parameters(),
            lr=self.lr,
            eps=self.epsilon,
            weight_decay=self.reg,
            amsgrad=self.amsgrad,
        )
        self.lr_scheduler = (
            ExponentialLR(self.optimizer, gamma=0.96) if self.lr_decay else None
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
            if (
                self.sampler not in ("random", "unconsumed", "popular")
                and self.sampler != "out-batch"
            ):
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

    def run(self, train_data, verbose, shuffle, eval_data, metrics, **kwargs):
        data_generator = self.get_data_generator(train_data)
        for epoch in range(1, self.n_epochs + 1):
            if self.lr_decay:
                print(
                    f"With lr_decay, epoch {epoch} learning rate: "
                    f"{self.optimizer.param_groups[0]['lr']}"
                )
            with time_block(f"Epoch {epoch}", verbose):
                self.torch_model.train()
                train_total_loss = []
                for data in data_generator(shuffle):
                    loss = self.compute_loss(data)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
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
                    eval_batch_size=self.eval_batch_size,
                    k=self.k,
                    sample_user_num=self.eval_user_num,
                    seed=self.model.seed,
                )
                print("=" * 30)

            if self.lr_scheduler:
                self.lr_scheduler.step()

    def compute_loss(self, data):
        user_embeds, item_embeds = self.torch_model(use_dropout=True)
        if self.loss_type in ("cross_entropy", "focal"):
            users = torch.LongTensor(data.users, device=self.device)
            users = user_embeds[users]
            items = torch.LongTensor(data.items, device=self.device)
            items = item_embeds[items]
            logits = torch.sum(torch.mul(users, items), dim=1)
            labels = torch.FloatTensor(data.labels, device=self.device)
            if self.loss_type == "cross_entropy":
                return binary_cross_entropy_loss(logits, labels)
            else:
                return focal_loss(logits, labels)
        elif self.loss_type in ("bpr", "max_margin"):
            users = torch.LongTensor(data.queries, device=self.device)
            users = user_embeds[users]
            items_pos = torch.LongTensor(data.item_pairs[0], device=self.device)
            items_pos = item_embeds[items_pos]
            items_neg = torch.LongTensor(data.item_pairs[1], device=self.device)
            items_neg = item_embeds[items_neg]
            if self.loss_type == "bpr":
                return bpr_loss(users, items_pos, items_neg)
            else:
                return max_margin_loss(users, items_pos, items_neg, self.margin)
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
        full_repr,
        num_walks,
        walk_len,
        margin,
        sampler,
        start_node,
        focus_start,
        k,
        eval_batch_size,
        eval_user_num,
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
            k,
            eval_batch_size,
            eval_user_num,
            device,
        )
        self.data_info = model.data_info
        self.paradigm = paradigm
        self.full_repr = full_repr
        self.num_walks = num_walks
        self.walk_len = walk_len
        self.start_node = start_node
        self.focus_start = focus_start
        self._check_sampler()

    def _check_sampler(self):
        if self.paradigm == "i2i" and self.sampler not in (
            "random",
            "out-batch",
            "popular",
        ):
            raise ValueError(
                f"`sampler` must be one of (`random`, `out-batch`, `popular`) for i2i, "
                f"got {self.sampler}, consider using u2i for no sampling"
            )
        if self.paradigm == "i2i" and self.loss_type == "focal":
            raise ValueError("i2i doesn't support focal loss")

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
        if self.paradigm == "u2i" and self.loss_type in ("cross_entropy", "focal"):
            user_feats = data.users, data.sparse_indices[0], data.dense_values[0]
            user_reprs = self.model.get_user_repr(*user_feats)
            item_feats = data.items, data.sparse_indices[1], data.dense_values[1]
            item_reprs = self.model.get_item_repr(*item_feats)
            logits = torch.sum(torch.mul(user_reprs, item_reprs), dim=1)
            labels = torch.FloatTensor(data.labels, device=self.device)
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
            if self.full_repr:
                item_pos_reprs = self.model.get_item_repr(*item_pos_feats)
                item_neg_reprs = self.model.get_item_repr(*item_neg_feats)
            else:
                item_pos_feats = feat_to_tensor(*item_pos_feats, device=self.device)
                item_pos_reprs = self.torch_model.get_raw_features(
                    *item_pos_feats, is_user=False
                )
                item_neg_feats = feat_to_tensor(*item_neg_feats, device=self.device)
                item_neg_reprs = self.torch_model.get_raw_features(
                    *item_neg_feats, is_user=False
                )

            if self.paradigm == "i2i" and self.loss_type == "cross_entropy":
                return graphsage_unsupervised_loss(
                    query_reprs, item_pos_reprs, item_neg_reprs
                )
            if self.loss_type == "bpr":
                return bpr_loss(query_reprs, item_pos_reprs, item_neg_reprs)
            else:
                return max_margin_loss(
                    query_reprs, item_pos_reprs, item_neg_reprs, self.margin
                )
