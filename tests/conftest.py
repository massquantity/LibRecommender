import os

import numpy as np
import pandas as pd
import pytest

from libreco.data import DatasetFeat, DatasetPure, split_by_ratio_chrono
from tests.utils_path import SAVE_PATH, remove_path


@pytest.fixture
def prepare_pure_data():
    data_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "sample_data",
        "sample_movielens_rating.dat",
    )
    pd_data = pd.read_csv(
        data_path, sep="::", names=["user", "item", "label", "time"], engine="python"
    )
    train_data, eval_data = split_by_ratio_chrono(pd_data, test_size=0.2)
    train_data, data_info = DatasetPure.build_trainset(train_data)
    eval_data = DatasetPure.build_evalset(eval_data)
    yield pd_data, train_data, eval_data, data_info
    remove_path(SAVE_PATH)


@pytest.fixture
def prepare_feat_data(read_feat_data):
    pd_data, (train_data, eval_data) = read_feat_data
    train_data, data_info = DatasetFeat.build_trainset(
        train_data=train_data,
        sparse_col=["sex", "occupation", "genre1", "genre2", "genre3"],
        dense_col=["age", "item_dense_feat"],
        user_col=["sex", "age", "occupation"],
        item_col=["genre1", "genre2", "genre3", "item_dense_feat"],
    )
    eval_data = DatasetFeat.build_testset(eval_data)
    yield pd_data, train_data, eval_data, data_info
    remove_path(SAVE_PATH)


@pytest.fixture
def prepare_multi_sparse_data(read_feat_data):
    pd_data, (train_data, eval_data) = read_feat_data
    train_data, data_info = DatasetFeat.build_trainset(
        train_data=train_data,
        sparse_col=["sex", "occupation"],
        multi_sparse_col=[["genre1", "genre2", "genre3"]],
        dense_col=["age", "item_dense_feat"],
        user_col=["sex", "age", "occupation"],
        item_col=["genre1", "genre2", "genre3", "item_dense_feat"],
        pad_val=["missing"],
    )
    eval_data = DatasetFeat.build_testset(eval_data)
    yield pd_data, train_data, eval_data, data_info
    remove_path(SAVE_PATH)


@pytest.fixture
def read_feat_data():
    data_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "sample_data",
        "sample_movielens_merged.csv",
    )
    pd_data = pd.read_csv(data_path, sep=",", header=0)
    pd_data["item_dense_feat"] = np.random.randn(len(pd_data), 1)
    return pd_data, split_by_ratio_chrono(pd_data, test_size=0.2)
