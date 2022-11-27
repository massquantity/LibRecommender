import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from .trainer import BaseTrainer
from ..evaluation import print_metrics
from ..utils.misc import colorize, time_block
from ..utils.sampling import PairwiseSampling


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
        k,
        eval_batch_size,
        eval_user_num,
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
        self.torch_model = model.torch_model
        self.optimizer = Adam(
            params=self.torch_model.parameters(),
            lr=self.lr,
            eps=self.epsilon,
            weight_decay=self.reg,
            amsgrad=self.amsgrad,
        )

    def run(self, train_data, verbose, shuffle, eval_data, metrics, **kwargs):
        scheduler = None
        if self.lr_decay:
            gamma = kwargs.get("gamma", 0.96)
            scheduler = ExponentialLR(self.optimizer, gamma=gamma)

        data_generator = PairwiseSampling(
            train_data, self.model.data_info, self.num_neg
        )
        for epoch in range(1, self.n_epochs + 1):
            if self.lr_decay:
                print(
                    f"With lr_decay, epoch {epoch} learning rate: "
                    f"{self.optimizer.param_groups[0]['lr']}"
                )
            with time_block(f"Epoch {epoch}", verbose):
                self.torch_model.train()
                train_total_loss = []
                for user, item_pos, item_neg in data_generator(
                    shuffle, self.batch_size
                ):
                    loss = self.bpr_loss(user, item_pos, item_neg)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    train_total_loss.append(loss.detach().cpu().item())

            if verbose > 1:
                train_loss_str = "train_loss: " + str(
                    round(float(np.mean(train_total_loss)), 4)
                )
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

            if scheduler:
                scheduler.step()

    def bpr_loss(self, user_indices, pos_item_indices, neg_item_indices):
        user_embeds, item_embeds = self.torch_model(use_dropout=True)
        pos_scores = torch.sum(
            torch.mul(user_embeds[user_indices], item_embeds[pos_item_indices]), axis=1
        )
        neg_scores = torch.sum(
            torch.mul(user_embeds[user_indices], item_embeds[neg_item_indices]), axis=1
        )
        log_sigmoid = F.logsigmoid(pos_scores - neg_scores)
        return torch.negative(torch.mean(log_sigmoid))
