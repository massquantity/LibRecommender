"""

Reference: William L. Hamilton et al. "Inductive Representation Learning on Large Graphs"
           (https://arxiv.org/abs/1706.02216)

author: massquantity

"""
import importlib
import itertools

import numpy as np
import torch
from tqdm import tqdm

from ..bases import EmbedBase, ModelMeta
from ..graph import check_dgl
from ..torchops import item_unique_to_tensor, user_unique_to_tensor
from ..training import SageDGLTrainer
from .torch_modules import GraphSageDGLModel


@check_dgl
class GraphSageDGL(EmbedBase, metaclass=ModelMeta, backend="torch"):
    def __new__(cls, *args, **kwargs):
        if cls.dgl_error is not None:
            raise cls.dgl_error
        cls._dgl = importlib.import_module("dgl")
        return super().__new__(cls)

    def __init__(
        self,
        task,
        data_info,
        loss_type="cross_entropy",
        paradigm="i2i",
        aggregator_type="mean",
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
        sample_walk_len=5,
        margin=1.0,
        sampler="random",
        start_node="random",
        focus_start=False,
        seed=42,
        device=torch.device("cpu"),
        lower_upper_bound=None,
        with_training=True,
    ):
        super().__init__(task, data_info, embed_size, lower_upper_bound)

        self.all_args = locals()
        self.loss_type = loss_type
        self.batch_size = batch_size
        self.paradigm = paradigm
        self.aggregator_type = aggregator_type
        self.dropout_rate = dropout_rate
        self.remove_edges = remove_edges
        self.num_layers = num_layers
        self.num_neighbors = num_neighbors
        self.seed = seed
        self.device = device
        self._check_params()
        if with_training:
            self.homo_g = self.build_homo_graph()
            self.hetero_g = self.build_hetero_graph()
            self.torch_model = self.build_model().to(device)
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
                device,
            )

    def _check_params(self):
        if self.task != "ranking":
            raise ValueError(f"{self.model_name} is only suitable for ranking")
        if self.paradigm not in ("u2i", "i2i"):
            raise ValueError("paradigm must either be `u2i` or `i2i`")
        if self.loss_type not in ("cross_entropy", "focal", "bpr", "max_margin"):
            raise ValueError(f"unsupported `loss_type`: {self.loss_type}")
        if self.model_name == "GraphSageDGL" and self.aggregator_type not in (
            "mean",
            "gcn",
            "pool",
            "lstm",
        ):
            raise ValueError(
                f"unsupported `aggregator_type`: {self.aggregator_type} for GraphSageDGL"
            )

    def build_homo_graph(self):
        src_items, dst_items = [], []
        for i in range(self.n_items):
            neighbors = set()
            for u in self.data_info.item_consumed[i]:
                neighbors.update(self.user_consumed[u])
            src_items.extend(neighbors)
            dst_items.extend([i] * len(neighbors))
        src = torch.tensor(src_items, dtype=torch.long)
        dst = torch.tensor(dst_items, dtype=torch.long)
        g = self._dgl.graph((src, dst), num_nodes=self.n_items)
        return g

    def build_hetero_graph(self):
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

    def build_model(self):
        return GraphSageDGLModel(
            self.paradigm,
            self.data_info,
            self.embed_size,
            self.batch_size,
            self.num_layers,
            self.dropout_rate,
            self.aggregator_type,
        )

    def sample_frontier(self, nodes):
        return self._dgl.sampling.sample_neighbors(
            g=self.homo_g,
            nodes=nodes,
            fanout=self.num_neighbors,
            edge_dir="in",
        )

    def transform_blocks(self, nodes, target_nodes=None):
        """
        bipartite graph block: (items(nodes) -> sampled neighbor nodes)
        -------------
        |     / ... |
        |    /  src |
        |dst -  src |
        |    \  src |
        |     \ ... |
        -------------
        """  # noqa: W605
        dgl = self._dgl
        blocks = []
        for _ in range(self.num_layers):
            frontier = self.sample_frontier(nodes)
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
        blocks = self.transform_blocks(nodes, target_nodes)
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
            return np.array(user_embed)
