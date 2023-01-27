"""

Reference: Rex Ying et al. "Graph Convolutional Neural Networks for Web-Scale Recommender Systems"
           (https://arxiv.org/abs/1806.01973)

author: massquantity

"""
import importlib

import torch

from ..bases import ModelMeta
from ..graph import check_dgl
from .graphsage_dgl import GraphSageDGL
from .torch_modules import PinSageDGLModel


@check_dgl
class PinSageDGL(GraphSageDGL, metaclass=ModelMeta, backend="torch"):
    def __new__(cls, *args, **kwargs):
        if cls.dgl_error is not None:
            raise cls.dgl_error
        cls._dgl = importlib.import_module("dgl")
        return super().__new__(cls)

    def __init__(
        self,
        task,
        data_info,
        loss_type="max_margin",
        paradigm="i2i",
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
        remove_edges=False,
        num_layers=2,
        num_neighbors=3,
        num_walks=10,
        neighbor_walk_len=2,
        sample_walk_len=5,
        termination_prob=0.5,
        margin=1.0,
        sampler="random",
        start_node="random",
        focus_start=False,
        seed=42,
        device=torch.device("cpu"),
        lower_upper_bound=None,
        with_training=True,
    ):
        super().__init__(
            task,
            data_info,
            loss_type,
            paradigm,
            "mean",
            embed_size,
            n_epochs,
            lr,
            lr_decay,
            epsilon,
            amsgrad,
            reg,
            batch_size,
            num_neg,
            dropout_rate,
            remove_edges,
            num_layers,
            num_neighbors,
            num_walks,
            sample_walk_len,
            margin,
            sampler,
            start_node,
            focus_start,
            seed,
            device,
            lower_upper_bound,
            with_training,
        )
        self.all_args = locals()
        self.num_walks = num_walks
        self.neighbor_walk_len = neighbor_walk_len
        self.termination_prob = termination_prob

    def build_model(self):
        return PinSageDGLModel(
            self.paradigm,
            self.data_info,
            self.embed_size,
            self.batch_size,
            self.num_layers,
            self.dropout_rate,
        )

    def sample_frontier(self, nodes):
        sampler = self._dgl.sampling.PinSAGESampler(
            self.hetero_g,
            ntype="item",
            other_type="user",
            num_traversals=self.neighbor_walk_len,
            termination_prob=self.termination_prob,
            num_random_walks=self.num_walks,
            num_neighbors=self.num_neighbors,
        )
        return sampler(nodes)
