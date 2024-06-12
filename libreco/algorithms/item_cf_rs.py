"""Implementation of RsItemCF."""
from ..bases import RsCfBase


class RsItemCF(RsCfBase):
    """*Item Collaborative Filtering* algorithm implemented in Rust.

    Parameters
    ----------
    task : {'rating', 'ranking'}
        Recommendation task. See :ref:`Task`.
    data_info : :class:`~libreco.data.DataInfo` object
        Object that contains useful information for training and inference.
    k_sim : int, default: 20
        Number of similar items to use.
    num_threads : int, default: 1
        Number of threads to use.
    min_common : int, default: 1
        Number of minimum common items to consider when computing similarities.
    mode : {'forward', 'invert'}, default: 'invert'
        Whether to use forward index or invert index.
    seed : int, default: 42
        Random seed.
    lower_upper_bound : tuple or None, default: None
        Lower and upper score bound for `rating` task.
    """

    def __init__(
        self,
        task,
        data_info,
        k_sim=20,
        num_threads=1,
        min_common=1,
        mode="invert",
        seed=42,
        lower_upper_bound=None,
    ):
        super().__init__(
            task,
            data_info,
            k_sim,
            num_threads,
            min_common,
            mode,
            seed,
            lower_upper_bound,
        )
        self.all_args = locals()
