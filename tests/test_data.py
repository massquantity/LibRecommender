import os.path
from io import StringIO

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix

from libreco.data import (
    DataInfo,
    DatasetFeat,
    DatasetPure,
    TransformedEvalSet,
    TransformedSet,
    process_data,
)
from libreco.data.data_info import OldInfo, store_old_info

sparse_col = ["sex", "occupation", "genre1", "genre2", "genre3"]
dense_col = ["age"]
user_col = ["sex", "age", "occupation"]
item_col = ["genre1", "genre2", "genre3"]

raw_data = StringIO(
    """
user,item,label,time,sex,age,occupation,genre1,genre2,genre3
4617,296,2,964138229,F,25,6,crime,drama,missing
1298,208,4,974849526,M,35,6,action,adventure,missing
4585,1769,4,964322774,M,35,7,action,thriller,missing
3706,1136,5,966376465,M,25,12,comedy,missing,missing
2137,1215,3,974640099,F,1,10,action,adventure,comedy
2461,1257,4,974170662,M,18,4,comedy,missing,missing
242,3148,3,977854274,F,18,4,drama,missing,missing
2211,932,4,974607346,M,45,6,romance,missing,missing
263,2115,2,976651827,F,25,7,action,adventure,missing
5184,866,5,961735308,M,18,20,crime,drama,romance
"""
)
# noinspection PyTypeChecker
pd_data = pd.read_csv(raw_data, header=0)


@pytest.fixture(params=[pd_data])
def pure_train_data(request):
    train_data, data_info = DatasetPure.build_trainset(request.param, shuffle=True)
    _ = DatasetPure.build_testset(request.param, shuffle=True)
    _, new_data_info = DatasetPure.merge_trainset(
        request.param, data_info, merge_behavior=False, shuffle=True
    )
    _ = DatasetPure.merge_testset(request.param, new_data_info, shuffle=True)
    return train_data, data_info


@pytest.fixture(
    params=[
        {
            "train_data": pd_data,
            "sparse_col": sparse_col,
            "dense_col": dense_col,
            "user_col": user_col,
            "item_col": item_col,
        }
    ]
)
def feat_train_data(request):
    train_data, data_info = DatasetFeat.build_trainset(**request.param, shuffle=True)
    _ = DatasetFeat.build_testset(request.param["train_data"], shuffle=True)
    _, new_data_info = DatasetFeat.merge_trainset(
        request.param["train_data"], data_info, merge_behavior=False, shuffle=True
    )
    _ = DatasetFeat.merge_testset(
        request.param["train_data"], new_data_info, shuffle=True
    )
    return train_data, data_info


def test_dataset_pure(pure_train_data):
    data, data_info = pure_train_data
    print(data_info)
    assert DatasetPure.train_called
    np.testing.assert_array_equal(
        DatasetPure.user_unique_vals,
        [242, 263, 1298, 2137, 2211, 2461, 3706, 4585, 4617, 5184],
    )

    assert isinstance(data, TransformedSet)
    assert isinstance(data_info, DataInfo)
    with pytest.raises(IndexError):
        assert len(data) == 10
        _ = data[11]
    with pytest.raises(AssertionError):
        _ = DatasetPure.build_trainset(pd_data[["user", "item"]])

    with pytest.raises(ValueError):
        data_without_item = pd_data.drop(columns="item")
        DatasetPure._check_col_names(data_without_item, "train")

    with pytest.raises(RuntimeError):
        DatasetPure.train_called = False
        DatasetPure.build_testset(pd_data, shuffle=True)
    DatasetPure.build_trainset(pd_data, shuffle=True)


def test_dataset_feat(feat_train_data):
    data, data_info = feat_train_data
    assert isinstance(data, TransformedSet)
    assert isinstance(data.sparse_interaction, csr_matrix)
    assert isinstance(data_info, DataInfo)
    assert len(data_info.col_name_mapping["sparse_col"]) == 5
    assert len(data_info.col_name_mapping["dense_col"]) == 1
    assert len(data_info.col_name_mapping["user_sparse_col"]) == 2
    assert len(data_info.col_name_mapping["user_dense_col"]) == 1
    assert len(data_info.col_name_mapping["item_sparse_col"]) == 3
    assert "item_dense_col" not in data_info.col_name_mapping

    assert data_info.user_sparse_col.name == ["sex", "occupation"]
    assert data_info.user_dense_col.name == ["age"]
    assert data_info.item_sparse_col.name == ["genre1", "genre2", "genre3"]
    assert data_info.item_dense_col.name == []
    assert data_info.n_users == data_info.n_items == data_info.data_size == 10
    assert data_info.item2id[208] == 0
    with pytest.raises(KeyError):
        _ = data_info.user2id[-999]

    with pytest.raises(RuntimeError):
        DatasetFeat.train_called = False
        DatasetFeat.build_testset(pd_data, shuffle=True)
    DatasetFeat.build_trainset(
        pd_data, user_col, item_col, sparse_col, dense_col, shuffle=True
    )


def test_data_info(feat_train_data):
    _, data_info = feat_train_data
    assert np.all(np.isin(data_info.item_unique_vals, data_info.popular_items))
    data_info.old_info = OldInfo(0, 0, 0, 0, popular_items=[-1, -9, 100])
    data_info._popular_items = None
    data_info.old_info = store_old_info(data_info)
    assert np.all(np.isin([-1, -9, 100], data_info.popular_items))

    # test save load DataInfo
    data_info.save(os.path.curdir, "test")
    data_info2 = DataInfo.load(os.path.curdir, "test")
    os.remove(os.path.join(os.path.curdir, "test_data_info.npz"))
    os.remove(os.path.join(os.path.curdir, "test_data_info_name_mapping.json"))
    os.remove(os.path.join(os.path.curdir, "test_user_consumed.pkl"))
    os.remove(os.path.join(os.path.curdir, "test_item_consumed.pkl"))
    assert data_info2.data_size == 10
    assert data_info.col_name_mapping == data_info2.col_name_mapping


def test_processing(feat_train_data):
    with pytest.raises(ValueError):
        process_data(pd_data, dense_col="age")
    with pytest.raises(ValueError):
        process_data(pd_data, dense_col=["age"], normalizer="unknown")

    for normalizer in ("min_max", "standard", "robust", "power"):
        train_data = pd_data.copy()
        process_data(
            train_data,
            dense_col=["age"],
            normalizer=normalizer,
            transformer=("log", "sqrt", "square"),
        )

        train_data, eval_data = pd_data.copy(), pd_data.copy()
        process_data(
            (train_data, eval_data),
            dense_col=["age", "label"],
            normalizer=normalizer,
            transformer=("log", "sqrt", "square"),
        )


def test_transformed_evalset():
    user_indices = [1, 2, 3, 4, 5]
    item_indices = [2, 3, 1, 6, 8]
    labels = [1, 1, 1, 1, 1]
    data1 = TransformedEvalSet(user_indices, item_indices, labels)
    data1.build_negatives(100, num_neg=3, seed=2222)

    data2 = TransformedEvalSet(user_indices, item_indices, labels)
    data2.build_negatives(100, num_neg=3, seed=2222)
    assert_array_equal(data1.item_indices, data2.item_indices)

    data3 = TransformedEvalSet(user_indices, item_indices, labels)
    data3.build_negatives(100, num_neg=3, seed=1111)
    assert np.any(data1.item_indices != data3.item_indices)

    user_indices = [1, 2, 1, 4, 1]
    item_indices = [2, 3, 1, 6, 8]
    labels = [1, 1, 0, 0, 1]
    data4 = TransformedEvalSet(user_indices, item_indices, labels)
    data4.build_negatives(100, num_neg=2, seed=3333)
    assert np.sort(data4.positive_consumed[1]).tolist() == [2, 8]
    assert data4.positive_consumed[2] == [3]
    assert 4 not in data4.positive_consumed

    user_indices = [1, 2, 1, 4, 1]
    item_indices = [2, 3, 1, 6, 8]
    labels = [0, 0, 0, 0, 0]
    data5 = TransformedEvalSet(user_indices, item_indices, labels)
    data5.build_negatives(100, num_neg=2, seed=3333)
    assert np.sort(data5.positive_consumed[1]).tolist() == [1, 2, 8]
    assert data5.positive_consumed[2] == [3]
    assert data5.positive_consumed[4] == [6]
