"""

Reference: Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"
           (https://arxiv.org/pdf/2002.02126.pdf)

author: massquantity

"""
import torch

from ..bases import EmbedBase, ModelMeta
from ..training import TorchTrainer
from .torch_modules import LightGCNModel


class LightGCN(EmbedBase, metaclass=ModelMeta, backend="torch"):
    def __init__(
        self,
        task,
        data_info,
        loss_type="bpr",
        embed_size=16,
        n_epochs=20,
        lr=0.001,
        lr_decay=False,
        epsilon=1e-8,
        amsgrad=False,
        reg=None,
        batch_size=256,
        num_neg=1,
        dropout_rate=0.0,
        n_layers=3,
        margin=1.0,
        sampler="random",
        seed=42,
        device=torch.device("cpu"),
        lower_upper_bound=None,
        with_training=True,
    ):
        super().__init__(task, data_info, embed_size, lower_upper_bound)

        self.all_args = locals()
        self.loss_type = loss_type
        self.dropout_rate = dropout_rate
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
                self.dropout_rate,
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
