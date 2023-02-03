import os
from pathlib import Path

import numpy as np
import pandas as pd

from libreco.data import DatasetFeat
from libreco.data.data_info import EmptyFeature, Feature
from libreco.feature.column import interaction_consumed, recover_sparse_cols
from libreco.feature.unique_features import (
    get_dense_indices,
    get_dense_values,
    get_sparse_indices,
)


def test_feature():
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
    _, data_info = DatasetFeat.build_trainset(
        train_data=pd_data,
        user_col=user_col,
        item_col=item_col,
        sparse_col=sparse_col,
        dense_col=dense_col,
        multi_sparse_col=multi_sparse_col,
        shuffle=False,
    )

    assert data_info.user_sparse_col == EmptyFeature
    assert data_info.user_dense_col == Feature(name=["age"], index=[0])
    assert data_info.item_sparse_col == Feature(
        name=["genre1", "genre2", "genre3"], index=[0, 1, 2]
    )
    assert data_info.item_dense_col == Feature(name=["item_dense_col"], index=[1])
    assert data_info.user_col == {"age"}
    assert data_info.item_col == {"genre3", "genre1", "genre2", "item_dense_col"}
    assert data_info.sparse_col == Feature(
        name=["genre1", "genre2", "genre3"], index=[0, 1, 2]
    )
    assert data_info.dense_col == Feature(name=["age", "item_dense_col"], index=[0, 1])

    print(
        get_sparse_indices(
            data_info, user=[1], item=[2], n_items=data_info.n_items, mode="predict"
        ).shape
    )
    print(
        get_sparse_indices(
            data_info, user=[1], item=[2], n_items=data_info.n_items, mode="recommend"
        ).shape
    )
    print(
        get_dense_values(
            data_info, user=[1], item=[2], n_items=data_info.n_items, mode="predict"
        ).shape
    )
    print(
        get_dense_values(
            data_info, user=[1], item=[2], n_items=data_info.n_items, mode="recommend"
        ).shape
    )
    print(
        get_dense_indices(
            data_info, user=[1], n_items=data_info.n_items, mode="predict"
        ).shape
    )
    print(
        get_dense_indices(
            data_info, user=[1], n_items=data_info.n_items, mode="recommend"
        ).shape
    )

    sparse_cols, multi_sparse_cols = recover_sparse_cols(data_info)
    assert sparse_cols is None
    assert multi_sparse_cols == [["genre1", "genre2", "genre3"]]

    interaction_data = data_info.interaction_data.copy()
    interaction_data.loc[interaction_data.user == 1, "user"] = pd.NA
    interaction_data.loc[interaction_data.item == 2, "item"] = pd.NA
    data_info.interaction_data = interaction_data
    assert data_info.interaction_data.user.isnull().any()
    assert data_info.interaction_data.item.isnull().any()
    new_data = data_info.get_indexed_interaction()
    assert new_data.user.notnull().all()
    assert new_data.item.notnull().all()


def test_remove_duplicates():
    user_indices = [1, 1, 1, 2, 2, 1, 2, 3, 2, 3]
    item_indices = [11, 11, 999, 0, 11, 11, 999, 11, 999, 0]
    user_consumed, item_consumed = interaction_consumed(user_indices, item_indices)
    assert isinstance(user_consumed, dict)
    assert isinstance(item_consumed, dict)
    assert isinstance(user_consumed[1], list)
    assert isinstance(item_consumed[11], list)
    assert user_consumed[1] == [11, 999]
    assert user_consumed[2] == [0, 11, 999]
    assert user_consumed[3] == [11, 0]
    assert item_consumed[11] == [1, 2, 3]
    assert item_consumed[999] == [1, 2]
    assert item_consumed[0] == [2, 3]
