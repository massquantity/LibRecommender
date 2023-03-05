import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from libreco.data import DatasetFeat
from libreco.utils.exception import NotSamplingError
from libreco.utils.sampling import NegativeSampling


def test_sampling():
    data_path = os.path.join(
        str(Path(os.path.realpath(__file__)).parent),
        "sample_data",
        "sample_movielens_merged.csv",
    )
    pd_data = pd.read_csv(data_path, sep=",", header=0)
    pd_data["item_dense_col"] = np.random.randint(0, 10000, size=len(pd_data))

    multi_sparse_col = [["genre1", "genre2", "genre3"]]
    sparse_col = []
    dense_col = ["age", "item_dense_col"]
    user_col = ["age"]
    item_col = ["genre1", "genre2", "genre3", "item_dense_col"]
    train_data, data_info = DatasetFeat.build_trainset(
        train_data=pd_data,
        user_col=user_col,
        item_col=item_col,
        sparse_col=sparse_col,
        dense_col=dense_col,
        multi_sparse_col=multi_sparse_col,
        shuffle=False,
    )

    neg_sampling = NegativeSampling(
        train_data, data_info, num_neg=1, sparse=True, dense=True, batch_sampling=False
    )
    sampled_data = neg_sampling.generate_all(item_gen_mode="popular")
    assert len(sampled_data[0]) == len(train_data) * 2

    # test batch sampling
    train_data.build_negative_samples(
        data_info, item_gen_mode="random", num_neg=1, seed=2022
    )
    neg_sampling = NegativeSampling(
        train_data, data_info, num_neg=1, sparse=True, dense=True, batch_sampling=True
    )
    batch_size = 64
    sampled_batch = neg_sampling(shuffle=True, batch_size=batch_size)
    assert len(next(sampled_batch)[0]) == batch_size * 2
