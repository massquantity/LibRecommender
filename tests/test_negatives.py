import os

import numpy as np
import pandas as pd
import pytest

from libreco.data import DatasetFeat
from libreco.sampling.data_sampler import (
    PairwiseDataGenerator,
    PairwiseRandomWalkGenerator,
    PointwiseDataGenerator,
)
from libreco.sampling.negatives import negatives_from_unconsumed


def test_negatives_exceed_sampling_tolerance():
    users = [0, 1, 2]
    items = [1, 2, 4]
    user_consumed_set = {0: {1}, 1: {3, 4}, 2: {1, 2, 3}}
    n_items = 5
    num_neg = 5
    tolerance = 100
    negatives = np.array_split(
        negatives_from_unconsumed(
            user_consumed_set, users, items, n_items, num_neg, tolerance
        ),
        3,
    )
    assert 1 not in negatives[0][:4]
    assert 2 not in negatives[1][:4]
    assert 4 not in negatives[2][:4]


@pytest.mark.parametrize(
    "config_feat_data",
    [
        {"sparse_col": ["sex"], "dense_col": ["age"], "user_col": ["sex", "age"]},
        {
            "sparse_col": ["genre1"],
            "dense_col": ["item_dense_col"],
            "item_col": ["genre1", "item_dense_col"],
        },
        {
            "sparse_col": ["sex", "genre1"],
            "dense_col": ["age", "item_dense_col"],
            "user_col": ["sex", "age"],
            "item_col": ["genre1", "item_dense_col"],
        },
    ],
    indirect=True,
)
def test_feature_negatives(config_feat_data):
    train_data, data_info = config_feat_data
    data_generator = PointwiseDataGenerator(
        train_data,
        data_info,
        batch_size=2,
        num_neg=1,
        sampler="random",
        seed=42,
        separate_features=False,
    )
    sparse_col_num = len(data_info.sparse_col.index)
    dense_col_num = len(data_info.dense_col.index)
    for data in data_generator(shuffle=False):
        assert data.sparse_indices.shape[1] == sparse_col_num
        assert data.dense_values.shape[1] == dense_col_num


@pytest.mark.parametrize(
    "config_feat_data",
    [
        {"sparse_col": ["sex"], "dense_col": ["age"], "user_col": ["sex", "age"]},
        {
            "sparse_col": ["genre1"],
            "dense_col": ["item_dense_col"],
            "item_col": ["genre1", "item_dense_col"],
        },
        {
            "sparse_col": ["sex", "genre1"],
            "dense_col": ["age", "item_dense_col"],
            "user_col": ["sex", "age"],
            "item_col": ["genre1", "item_dense_col"],
        },
    ],
    indirect=True,
)
def test_repeat_positive_features(config_feat_data):
    train_data, data_info = config_feat_data
    data_generator = PairwiseDataGenerator(
        train_data,
        data_info,
        batch_size=2,
        num_neg=3,
        sampler="random",
        seed=42,
        repeat_positives=True,
    )
    for data in data_generator():
        users = data.queries
        items_pos = data.item_pairs[0]
        items_neg = data.item_pairs[1]
        assert len(users) == len(items_pos) == len(items_neg)

    data_generator = PairwiseRandomWalkGenerator(
        train_data,
        data_info,
        batch_size=2,
        num_neg=3,
        num_walks=10,
        walk_length=5,
        sampler="random",
        seed=42,
        repeat_positives=True,
    )
    for data in data_generator():
        users = data.queries
        items_pos = data.item_pairs[0]
        items_neg = data.item_pairs[1]
        assert len(users) == len(items_pos) == len(items_neg)


@pytest.fixture
def config_feat_data(request):
    data_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "sample_data",
        "sample_movielens_merged.csv",
    )
    pd_data = pd.read_csv(data_path, sep=",", header=0)
    pd_data["item_dense_col"] = np.random.random(len(pd_data))
    train_data, data_info = DatasetFeat.build_trainset(
        train_data=pd_data, **request.param
    )
    return train_data, data_info
