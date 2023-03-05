"""Implementation of PinSage."""
from .torch_modules import PinSageModel
from ..bases import ModelMeta, SageBase
from ..graph import NeighborWalker


class PinSage(SageBase, metaclass=ModelMeta, backend="torch"):
    """*PinSage* algorithm.

    .. NOTE::
        This algorithm is implemented in PyTorch.

    .. CAUTION::
        PinSage can only be used in ``ranking`` task.

    .. versionadded:: 0.12.0

    Parameters
    ----------
    task : {'ranking'}
        Recommendation task. See :ref:`Task`.
    data_info : :class:`~libreco.data.DataInfo` object
        Object that contains useful information for training and inference.
    loss_type : {'cross_entropy', 'focal', 'bpr', 'max_margin'}, default: 'max_margin'
        Loss for model training.
    paradigm : {'u2i', 'i2i'}, default: 'i2i'
        Choice for features in model.

        - ``'u2i'`` will combine user features and item features.
        - ``'i2i'`` will only use item features, this is the setting in the original paper.

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
    neighbor_walk_len : int, default: 2
        Length of random walk to sample neighbor nodes for target node.
    sample_walk_len : int, default: 5
        Length of each random walk to sample positive item pairs.
    termination_prob : float, default: 0.5
        Termination probability after one walk for neighbor random walk sampling.
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
    PinSageDGL

    References
    ----------
    *Rex Ying et al.* `Graph Convolutional Neural Networks for Web-Scale Recommender Systems
    <https://arxiv.org/abs/1806.01973>`_.
    """

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
        device="cuda",
        lower_upper_bound=None,
    ):
        super().__init__(
            task,
            data_info,
            loss_type,
            paradigm,
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
        )
        self.all_args = locals()
        self.neighbor_walk_len = neighbor_walk_len
        self.termination_prob = termination_prob

    def build_model(self):
        self.neighbor_walker = NeighborWalker(self, self.data_info)
        self.torch_model = PinSageModel(
            self.paradigm,
            self.data_info,
            self.embed_size,
            self.batch_size,
            self.num_layers,
            self.dropout_rate,
        ).to(self.device)
