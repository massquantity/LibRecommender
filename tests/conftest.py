from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from libreco.data import DatasetFeat, DatasetPure, split_by_ratio_chrono
from tests.utils_data import SAVE_PATH, remove_path


@pytest.fixture
def prepare_pure_data():
    data_path = Path(__file__).parent / "sample_data" / "sample_movielens_rating.dat"
    pd_data = pd.read_csv(
        data_path, sep="::", names=["user", "item", "label", "time"], engine="python"
    )
    train_data, eval_data = split_by_ratio_chrono(pd_data, test_size=0.2)
    train_data, data_info = DatasetPure.build_trainset(train_data)
    eval_data = DatasetPure.build_evalset(eval_data)
    yield pd_data, train_data, eval_data, data_info
    remove_path(SAVE_PATH)


@pytest.fixture
def read_feat_data():
    data_path = Path(__file__).parent / "sample_data" / "sample_movielens_merged.csv"
    pd_data = pd.read_csv(data_path, sep=",", header=0)
    pd_data["item_dense_feat"] = np.random.default_rng(42).normal(size=len(pd_data))
    return pd_data, split_by_ratio_chrono(pd_data, test_size=0.2)


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
def make_synthetic_data():
    size = 200
    np_rng = np.random.default_rng(42)
    genres = ["crime", "drama", "action", "comedy", "missing"]
    data = pd.DataFrame(
        {
            "user": np_rng.integers(0, 20, size),
            "item": np_rng.integers(0, 60, size),
            "label": np_rng.integers(0, 5, size) + 1,
            "time": np_rng.integers(10000, 20000, size),
            "sex": np_rng.choice(["male", "female"], size),
            "occupation": np_rng.choice(list("abcdefg"), size),
            "age": np_rng.integers(0, 100, size),
            "genre1": np_rng.choice(genres, size),
            "genre2": np_rng.choice(genres, size),
            "genre3": np_rng.choice(genres, size),
            "profit": np_rng.random(size) * 10000,
        }
    )
    return data


@pytest.fixture
def pure_data_small(make_synthetic_data):
    pd_data = make_synthetic_data
    train_data, eval_data = split_by_ratio_chrono(pd_data, test_size=0.2)
    train_data, data_info = DatasetPure.build_trainset(train_data)
    eval_data = DatasetPure.build_evalset(eval_data)
    yield pd_data, train_data, eval_data, data_info
    remove_path(SAVE_PATH)


@pytest.fixture
def feat_data_small(make_synthetic_data):
    pd_data = make_synthetic_data
    train_data, eval_data = split_by_ratio_chrono(pd_data, test_size=0.2)
    train_data, data_info = DatasetFeat.build_trainset(
        train_data=train_data,
        sparse_col=["sex", "occupation", "genre1", "genre2", "genre3"],
        dense_col=["age", "profit"],
        user_col=["sex", "age", "occupation"],
        item_col=["genre1", "genre2", "genre3", "profit"],
    )
    eval_data = DatasetFeat.build_testset(eval_data)
    yield pd_data, train_data, eval_data, data_info
    remove_path(SAVE_PATH)


@pytest.fixture
def multi_sparse_data_small(make_synthetic_data):
    pd_data = make_synthetic_data
    train_data, eval_data = split_by_ratio_chrono(pd_data, test_size=0.2)
    train_data, data_info = DatasetFeat.build_trainset(
        train_data=train_data,
        sparse_col=["sex", "occupation"],
        multi_sparse_col=[["genre1", "genre2", "genre3"]],
        dense_col=["age", "profit"],
        user_col=["sex", "age", "occupation"],
        item_col=["genre1", "genre2", "genre3", "profit"],
        pad_val=["missing"],
    )
    eval_data = DatasetFeat.build_testset(eval_data)
    yield pd_data, train_data, eval_data, data_info
    remove_path(SAVE_PATH)


@pytest.fixture
def config_feat_data_small(make_synthetic_data, request):
    pd_data = make_synthetic_data
    train_data, eval_data = split_by_ratio_chrono(pd_data, test_size=0.2)
    train_data, data_info = DatasetFeat.build_trainset(
        train_data=train_data,
        pad_val=["missing"],
        **request.param,
    )
    eval_data = DatasetFeat.build_testset(eval_data)
    yield pd_data, train_data, eval_data, data_info
    remove_path(SAVE_PATH)
