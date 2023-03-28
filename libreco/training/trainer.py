import abc

from ..batch import adjust_batch_size


class BaseTrainer(abc.ABC):
    def __init__(
        self,
        model,
        task,
        loss_type,
        n_epochs,
        lr,
        lr_decay,
        epsilon,
        batch_size,
        sampler,
        num_neg,
    ):
        self.model = model
        self.task = task
        self.loss_type = loss_type
        self.n_epochs = n_epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.epsilon = epsilon
        self.batch_size = adjust_batch_size(model, batch_size)
        self.sampler = sampler
        self.num_neg = num_neg

    def _check_params(self):
        if self.model.model_name != "YouTubeRetrieval":
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

    @abc.abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError
