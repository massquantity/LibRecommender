import os.path

import pytest
import pandas as pd

from libreco.data import split_by_ratio_chrono, DatasetPure, DatasetFeat


@pytest.fixture
def prepare_pure_data():
    data_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "sample_data",
        "sample_movielens_rating.dat"
    )
    pd_data = pd.read_csv(data_path, sep="::", names=["user", "item", "label", "time"])
    train_data, eval_data = split_by_ratio_chrono(pd_data, test_size=0.2)
    train_data, data_info = DatasetPure.build_trainset(train_data)
    eval_data = DatasetPure.build_evalset(eval_data)
    return pd_data, train_data, eval_data, data_info


@pytest.fixture
def prepare_feat_data():
    data_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "sample_data",
        "sample_movielens_merged.csv"
    )
    pd_data = pd.read_csv(data_path, sep=",", header=0)
    train_data, eval_data = split_by_ratio_chrono(pd_data, test_size=0.2)
    train_data, data_info = DatasetFeat.build_trainset(
        train_data=train_data,
        sparse_col=["sex", "occupation", "genre1", "genre2", "genre3"],
        dense_col=["age"],
        user_col=["sex", "age", "occupation"],
        item_col=["genre1", "genre2", "genre3"]
    )
    eval_data = DatasetFeat.build_testset(eval_data)
    return pd_data, train_data, eval_data, data_info
