"""Implementation of LightGCN."""
import torch

from .torch_modules import LightGCNModel
from ..bases import EmbedBase, ModelMeta
from ..torchops import device_config


class LightGCN(EmbedBase, metaclass=ModelMeta, backend="torch"):
    """*LightGCN* algorithm.

    .. CAUTION::
        LightGCN can only be used in ``ranking`` task.

    Parameters
    ----------
    task : {'ranking'}
        Recommendation task. See :ref:`Task`.
    data_info : :class:`~libreco.data.DataInfo` object
        Object that contains useful information for training and inference.
    loss_type : {'cross_entropy', 'focal', 'bpr', 'max_margin'}, default: 'bpr'
        Loss for model training.
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
    n_layers : int, default: 3
        Number of GCN layer.
    margin : float, default: 1.0
        Margin used in `max_margin` loss.
    sampler : {'random', 'unconsumed', 'popular'}, default: 'random'
        Negative sampling strategy.

        - ``'random'`` means random sampling.
        - ``'unconsumed'`` samples items that the target user did not consume before.
        - ``'popular'`` has a higher probability to sample popular items as negative samples.

    seed : int, default: 42
        Random seed.
    device : {'cpu', 'cuda'}, default: 'cuda'
        Refer to `torch.device <https://pytorch.org/docs/stable/tensor_attributes.html#torch.device>`_.

        .. versionchanged:: 1.0.0
           Accept str type ``'cpu'`` or ``'cuda'``, instead of ``torch.device(...)``.

    lower_upper_bound : tuple or None, default: None
        Lower and upper score bound for `rating` task.

    References
    ----------
    *Xiangnan He et al.* `LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
    <https://arxiv.org/pdf/2002.02126.pdf>`_.
    """

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
        device="cuda",
        lower_upper_bound=None,
        with_training=True,
    ):
        super().__init__(task, data_info, embed_size, lower_upper_bound)

        self.all_args = locals()
        self.loss_type = loss_type
        self.n_epochs = n_epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        self.reg = reg
        self.batch_size = batch_size
        self.num_neg = num_neg
        self.dropout_rate = dropout_rate
        self.n_layers = n_layers
        self.margin = margin
        self.sampler = sampler
        self.seed = seed
        self.device = device_config(device)
        self._check_params()

    def build_model(self):
        self.torch_model = LightGCNModel(
            self.n_users,
            self.n_items,
            self.embed_size,
            self.n_layers,
            self.dropout_rate,
            self.user_consumed,
            self.device,
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
        self.user_embed = embeddings[0].detach().cpu().numpy()
        self.item_embed = embeddings[1].detach().cpu().numpy()
