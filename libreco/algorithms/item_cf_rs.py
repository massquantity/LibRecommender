"""Implementation of RsItemCF."""
from ..bases import RsCfBase


class RsItemCF(RsCfBase):
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
