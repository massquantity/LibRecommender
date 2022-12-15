"""

Reference: Rex Ying et al. "Graph Convolutional Neural Networks for Web-Scale Recommender Systems"
           (https://arxiv.org/abs/1806.01973)

author: massquantity

"""
import importlib
import itertools

import numpy as np
import torch
from tqdm import tqdm

from .torch_modules import PinSageDGLModel
from ..bases import EmbedBase
from ..graph import check_dgl
from ..torchops import user_unique_to_tensor, item_unique_to_tensor
from ..training import SageDGLTrainer


@check_dgl
class PinSageDGL(EmbedBase):
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
        dropout=0.0,
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
        self.batch_size = batch_size
        self.paradigm = paradigm
        self.remove_edges = remove_edges
        self.seed = seed
        self.device = device
        self._check_params()
        if with_training:
            self.torch_model = PinSageDGLModel(
                paradigm,
                data_info,
                embed_size,
                batch_size,
                num_layers,
                dropout,
            ).to(device)
            self.trainer = SageDGLTrainer(
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
                paradigm,
                num_walks,
                sample_walk_len,
                margin,
                sampler,
                start_node,
                focus_start,
                k,
                eval_batch_size,
                eval_user_num,
                device,
            )
            self.g = self.build_graph()
            self.samplers = [
                self._dgl.sampling.PinSAGESampler(
                    self.g,
                    ntype="item",
                    other_type="user",
                    num_traversals=neighbor_walk_len,
                    termination_prob=termination_prob,
                    num_random_walks=num_walks,
                    num_neighbors=num_neighbors,
                )
                for _ in range(num_layers)
            ]

    def _check_params(self):
        if self.task != "ranking":
            raise ValueError("PinSage is only suitable for ranking")
        if self.paradigm not in ("u2i", "i2i"):
            raise ValueError("paradigm must either be `u2i` or `i2i`")
        if self.loss_type not in ("cross_entropy", "focal", "bpr", "max_margin"):
            raise ValueError(f"unsupported `loss_type`: {self.loss_type}")

    def build_graph(self):
        items = [list(self.user_consumed[u]) for u in range(self.n_users)]
        counts = [len(i) for i in items]
        users = torch.arange(self.n_users).repeat_interleave(torch.tensor(counts))
        items = list(itertools.chain.from_iterable(items))
        items = torch.tensor(items, dtype=torch.long)
        graph_data = {
            ("user", "consumed", "item"): (users, items),
            ("item", "consumed-by", "user"): (items, users),
        }
        num_nodes = {"user": self.n_users, "item": self.n_items}
        return self._dgl.heterograph(graph_data, num_nodes)

    def sample_blocks(self, nodes, target_nodes=None):
        """# noqa: W605
        bipartite graph block: (items(nodes) -> sampled neighbor nodes)
        -------------
        |     / ... |
        |    /  src |
        |dst -  src |
        |    \  src |
        |     \ ... |
        -------------
        """
        dgl = self._dgl
        blocks = []
        for sampler in self.samplers:
            frontier = sampler(nodes)
            if (
                self.paradigm == "i2i"
                and self.remove_edges
                and target_nodes is not None
            ):
                heads_pos, heads_neg, tails_pos, tails_neg = target_nodes
                eids = frontier.edge_ids(
                    torch.cat([heads_pos, heads_neg]),
                    torch.cat([tails_pos, tails_neg]),
                    return_uv=True,
                )[2]
                if len(eids) > 0:
                    frontier = dgl.remove_edges(frontier, eids)
            block = dgl.to_block(frontier, dst_nodes=nodes)
            nodes = block.srcdata[dgl.NID]
            blocks.append(block)
        blocks.reverse()
        return blocks

    def get_user_repr(self, users):
        user_feat_tensors = user_unique_to_tensor(users, self.data_info, self.device)
        return self.torch_model.user_repr(*user_feat_tensors)

    def get_item_repr(self, nodes, target_nodes=None):
        blocks = self.sample_blocks(nodes, target_nodes)
        start_neighbor_nodes = blocks[0].srcdata[self._dgl.NID]
        start_nodes, sparse_indices, dense_values = item_unique_to_tensor(
            start_neighbor_nodes, self.data_info, self.device
        )
        for i in range(len(blocks)):
            blocks[i] = blocks[i].to(self.device)
        return self.torch_model(blocks, start_nodes, sparse_indices, dense_values)

    @torch.no_grad()
    def set_embeddings(self):
        self.torch_model.eval()
        all_items = list(range(self.n_items))
        item_embed = []
        for i in tqdm(range(0, self.n_items, self.batch_size), desc="item embedding"):
            items = torch.tensor(all_items[i : i + self.batch_size], dtype=torch.long)
            item_reprs = self.get_item_repr(items)
            item_embed.append(item_reprs.cpu().numpy())
        self.item_embed = np.concatenate(item_embed, axis=0)
        self.user_embed = self.get_user_embeddings()

    @torch.no_grad()
    def get_user_embeddings(self):
        self.torch_model.eval()
        user_embed = []
        if self.paradigm == "u2i":
            for i in range(0, self.n_users, self.batch_size):
                users = np.arange(i, min(i + self.batch_size, self.n_users))
                user_reprs = self.get_user_repr(users).cpu().numpy()
                user_embed.append(user_reprs)
            return np.concatenate(user_embed, axis=0)
        else:
            for u in range(self.n_users):
                items = self.user_consumed[u]
                user_embed.append(np.mean(self.item_embed[items], axis=0))
                # user_embed.append(self.item_embed[items[-1]])
            return np.array(user_embed)
