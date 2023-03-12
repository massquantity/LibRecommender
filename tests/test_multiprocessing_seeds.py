import multiprocessing
import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader


class BatchData(torch.utils.data.Dataset):
    def __init__(self, data_size):
        self.data_size = data_size

    def __getitem__(self, idx):
        return 1

    def __len__(self):
        return self.data_size


class Collator:
    def __init__(self, same_seed):
        self.np_rng = None
        self.seed = 42
        self.same_seed = same_seed

    def __call__(self, batch):
        self._set_random_seeds()
        np_random = self.np_rng.integers(0, 2**32 - 1)
        torch_random = torch.randint(0, 2**32 - 1, (1,)).item()
        torch_seed = torch.initial_seed()
        return np_random, torch_random, torch_seed

    def _set_random_seeds(self):
        if self.np_rng is None or self.same_seed:
            worker_info = torch.utils.data.get_worker_info()
            seed = self.seed if worker_info is None else worker_info.seed
            seed = seed % 3407 * 11
            random.seed(seed)
            torch.manual_seed(seed)
            self.np_rng = np.random.default_rng(seed)


@pytest.fixture
def get_data(request):
    data_size = 20
    same_seed = request.param["same_seed"]
    batch_size = request.param["batch_size"]
    num_workers = request.param["num_workers"]
    batch_data = BatchData(data_size)
    collate_fn = Collator(same_seed=same_seed)
    data_loader = DataLoader(
        batch_data,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    return data_loader, same_seed, num_workers


# `data_loader` should generate different random number in different seeds and processes
@pytest.mark.parametrize(
    "get_data",
    [
        {"same_seed": True, "num_workers": 4, "batch_size": 3},
        {"same_seed": True, "num_workers": 4, "batch_size": 1},
        {"same_seed": True, "num_workers": 2, "batch_size": 3},
        {"same_seed": True, "num_workers": 2, "batch_size": 1},
        {"same_seed": False, "num_workers": 4, "batch_size": 3},
        {"same_seed": False, "num_workers": 4, "batch_size": 1},
        {"same_seed": False, "num_workers": 2, "batch_size": 3},
        {"same_seed": False, "num_workers": 2, "batch_size": 1},
    ],
    indirect=True,
)
def test_multiprocessing_seeds(get_data):
    data_loader, same_seed, num_workers = get_data
    cpu_cores = multiprocessing.cpu_count()
    if num_workers < cpu_cores:
        np_random, torch_random, torch_seeds = [], [], []
        for d in data_loader:
            np_random.append(d[0])
            torch_random.append(d[1])
            torch_seeds.append(d[2])

        if same_seed:
            assert len(np.unique(np_random)) == num_workers
            assert len(np.unique(torch_random)) == num_workers
            assert len(np.unique(torch_seeds)) == num_workers
        else:
            assert len(np.unique(np_random)) == len(np_random)
            assert len(np.unique(torch_random)) == len(torch_random)
            assert len(np.unique(torch_seeds)) == num_workers
