"""Implementation of GraphSageDGL."""
import importlib
import itertools

import numpy as np
import torch
from tqdm import tqdm

from ..bases import EmbedBase, ModelMeta
from ..graph import check_dgl
from ..torchops import device_config, item_unique_to_tensor, user_unique_to_tensor
from .torch_modules import GraphSageDGLModel


@check_dgl
class GraphSageDGL(EmbedBase, metaclass=ModelMeta, backend="torch"):
    """*GraphSageDGL* algorithm.

    .. NOTE::
        This algorithm is implemented in `DGL <https://github.com/dmlc/dgl>`_.

    .. CAUTION::
        GraphSageDGL can only be used in ``ranking`` task.

    .. versionadded:: 0.12.0

    Parameters
    ----------
    task : {'ranking'}
        Recommendation task. See :ref:`Task`.
    data_info : :class:`~libreco.data.DataInfo` object
        Object that contains useful information for training and inference.
    loss_type : {'cross_entropy', 'focal', 'bpr', 'max_margin'}, default: 'cross_entropy'
        Loss for model training.
    paradigm : {'u2i', 'i2i'}, default: 'i2i'
        Choice for features in model.

        - ``'u2i'`` will combine user features and item features.
        - ``'i2i'`` will only use item features, this is the setting in the original paper.

    aggregator_type : {'mean', 'gcn', 'pool', 'lstm'}, default: 'mean'
        Aggregator type to use in GraphSage. Refer to `SAGEConv
        <https://docs.dgl.ai/en/latest/generated/dgl.nn.pytorch.conv.SAGEConv.html>`_ in DGL.
    embed_size: int, default: 16
        Vector size of embeddings.
    n_epochs: int, default: 10
        Number of epochs for training.
    lr : float, default 0.001
        Learning rate for training.
    lr_decay : bool, default: False
        Whether to use learning rate decay.
    epsilon : float, default: 1e-8
        A small constant added to the denominator to improve numerical stability in
        Adam optimizer.
    amsgrad : bool, default: False
        Whether to use the AMSGrad variant from the paper
        `On the Convergence of Adam and Beyond <https://openreview.net/forum?id=ryQu7f-RZ>`_.
    reg : float or None, default: None
        Regularization parameter, must be non-negative or None.
    batch_size : int, default: 256
        Batch size for training.
    num_neg : int, default: 1
        Number of negative samples for each positive sample.
    dropout_rate : float, default: 0.0
        Probability of a node being dropped. 0.0 means dropout is not used.
    remove_edges : bool, default: False
        Whether to remove edges between target node and its positive pair nodes
        when target node's sampled neighbor nodes contain positive pair nodes.
        This only applies in 'i2i' paradigm.
    num_layers : int, default: 2
        Number of GCN layers.
    num_neighbors : int, default: 3
        Number of sampled neighbors in each layer
    num_walks : int, default: 10
        Number of random walks to sample positive item pairs. This only applies in
        'i2i' paradigm.
    sample_walk_len : int, default: 5
        Length of each random walk to sample positive item pairs.
    margin : float, default: 1.0
        Margin used in `max_margin` loss.
    sampler : {'random', 'unconsumed', 'popular', 'out-batch'}, default: 'random'
        Negative sampling strategy. The ``'u2i'`` paradigm can use ``'random'``, ``'unconsumed'``,
        ``'popular'``, and the ``'i2i'`` paradigm can use ``'random'``, ``'out-batch'``, ``'popular'``.

        - ``'random'`` means random sampling.
        - ``'unconsumed'`` samples items that the target user did not consume before.
          This can't be used in ``'i2i'`` since it has no users.
        - ``'popular'`` has a higher probability to sample popular items as negative samples.
        - ``'out-batch'`` samples items that didn't appear in the batch.
          This can only be used in ``'i2i'`` paradigm.

    start_node : {'random', 'unpopular'}, default: 'random'
        Strategy for choosing start nodes in random walks. ``'unpopular'`` will place a higher
        probability on unpopular items, which may increase diversity but hurt metrics.
        This only applies in ``'i2i'`` paradigm.
    focus_start : bool, default: False
        Whether to keep the start nodes in random walk sampling. The purpose of the
        parameter ``start_node`` and ``focus_start`` is oversampling unpopular items.
        If you set ``start_node='popular'`` and ``focus_start=True``, unpopular items will
        be kept in positive samples, which may increase diversity.
    seed : int, default: 42
        Random seed.
    device : {'cpu', 'cuda'}, default: 'cuda'
        Refer to `torch.device <https://pytorch.org/docs/stable/tensor_attributes.html#torch.device>`_.

        .. versionchanged:: 1.0.0
           Accept str type ``'cpu'`` or ``'cuda'``, instead of ``torch.device(...)``.

    lower_upper_bound : tuple or None, default: None
        Lower and upper score bound for `rating` task.

    See Also
    --------
    GraphSage

    References
    ----------
    *William L. Hamilton et al.* `Inductive Representation Learning on Large Graphs
    <https://arxiv.org/abs/1706.02216>`_.
    """

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
        device="cuda",
        lower_upper_bound=None,
    ):
        super().__init__(task, data_info, embed_size, lower_upper_bound)

        self.all_args = locals()
        self.loss_type = loss_type
        self.paradigm = paradigm
        self.aggregator_type = aggregator_type
        self.n_epochs = n_epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        self.reg = reg
        self.batch_size = batch_size
        self.num_neg = num_neg
        self.dropout_rate = dropout_rate
        self.remove_edges = remove_edges
        self.num_layers = num_layers
        self.num_neighbors = num_neighbors
        self.num_walks = num_walks
        self.sample_walk_len = sample_walk_len
        self.margin = margin
        self.sampler = sampler
        self.start_node = start_node
        self.focus_start = focus_start
        self.seed = seed
        self.device = device_config(device)
        self._check_params()

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
        self.homo_g = self.build_homo_graph()
        self.hetero_g = self.build_hetero_graph()
        self.torch_model = GraphSageDGLModel(
            self.paradigm,
            self.data_info,
            self.embed_size,
            self.batch_size,
            self.num_layers,
            self.dropout_rate,
            self.aggregator_type,
        ).to(self.device)

    def sample_frontier(self, nodes):
        return self._dgl.sampling.sample_neighbors(
            g=self.homo_g,
            nodes=nodes,
            fanout=self.num_neighbors,
            edge_dir="in",
        )

    def transform_blocks(self, nodes, target_nodes=None):  # noqa: D400, D415
        r"""Bipartite graph block: items(nodes) -> sampled neighbor nodes

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
