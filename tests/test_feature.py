import os
from dataclasses import astuple
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from libreco.data import DatasetFeat
from libreco.data.data_info import EmptyFeature, Feature, store_old_info
from libreco.feature.multi_sparse import (
    get_multi_sparse_indices_matrix,
    recover_sparse_cols,
)
from libreco.feature.sparse import (
    column_sparse_indices,
    get_last_offset,
    get_oov_pos,
    get_sparse_indices_matrix,
    merge_offset,
    merge_sparse_indices,
)
from libreco.feature.update import (
    update_id_unique,
    update_multi_sparse_unique,
    update_sparse_unique,
    update_unique_feats,
)
from libreco.prediction.preprocess import (
    features_from_batch,
    get_original_feats,
    set_temp_feats,
)
from libreco.recommendation.preprocess import _get_original_feats as get_rec_feats
from libreco.recommendation.preprocess import process_embed_feat


def test_invalid_features():
    data_path = Path(__file__).parent / "sample_data" / "sample_movielens_merged.csv"
    pd_data = pd.read_csv(data_path, sep=",", header=0)
    sparse_col = ["genre1", "occupation"]
    dense_col = ["age"]
    user_col = ["age", "sex"]
    item_col = ["genre1"]
    with pytest.raises(
        ValueError, match="Got inconsistent columns: ['occupation' 'sex']*"
    ):
        DatasetFeat.build_trainset(
            train_data=pd_data,
            user_col=user_col,
            item_col=item_col,
            sparse_col=sparse_col,
            dense_col=dense_col,
        )

    sparse_col = ["genre1", "occupation", "age"]
    dense_col = ["age"]
    user_col = ["age", "occupation"]
    item_col = ["genre1"]
    with pytest.raises(ValueError, match="Please make sure length of columns match*"):
        DatasetFeat.build_trainset(
            train_data=pd_data,
            user_col=user_col,
            item_col=item_col,
            sparse_col=sparse_col,
            dense_col=dense_col,
        )

    multi_sparse_col = [["genre1", "genre2", "genre3"]]
    sparse_col = []
    dense_col = ["age"]
    user_col = []
    item_col = ["genre1", "genre2", "genre3"]
    with pytest.raises(ValueError, match="Please make sure length of columns match*"):
        DatasetFeat.build_trainset(
            train_data=pd_data,
            user_col=user_col,
            item_col=item_col,
            sparse_col=sparse_col,
            dense_col=dense_col,
            multi_sparse_col=multi_sparse_col,
        )


def test_data_info_features():
    data_path = os.path.join(
        str(Path(os.path.realpath(__file__)).parent),
        "sample_data",
        "sample_movielens_merged.csv",
    )
    pd_data = pd.read_csv(data_path, sep=",", header=0)
    np_rng = np.random.default_rng(42)
    pd_data["item_dense_col"] = np_rng.integers(0, 10000, size=len(pd_data))

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
    assert data_info.user_col == ["age"]
    assert data_info.item_col == ["genre1", "genre2", "genre3", "item_dense_col"]
    assert data_info.sparse_col == Feature(
        name=["genre1", "genre2", "genre3"], index=[0, 1, 2]
    )
    assert data_info.dense_col == Feature(name=["age", "item_dense_col"], index=[0, 1])


@pytest.fixture
def feature_data():
    size = 5
    np_rng = np.random.default_rng(88)
    data = pd.DataFrame(
        {
            "user": [4, 1, 10, 11, 12],
            "item": [1, 2, 3, 4, 5],
            "label": np_rng.integers(0, 5, size) + 1,
            "sex": ["M", "F", "M", "M", "F"],
            "occupation": ["c", "a", "a", "b", "a"],
            "age": np_rng.integers(0, 100, size),
            "actor1": [11, 0, 77, 44, 77],
            "actor2": [0, 22, 11, 99, 77],
            "profit": np_rng.random(size) * 10000,
            "genre1": ["x", "y", "z", "x", "missing"],
            "genre2": ["xx", "missing", "xx", "z", "missing"],
            "genre3": ["y", "y", "zz", "x", "missing"],
        }
    )
    return data


def test_sparse_indices(feature_data):
    sparse_cols = [
        "sex",
        "occupation",
        "actor1",
        "actor2",
        "genre1",
        "genre2",
        "genre3",
    ]
    dense_cols = ["age", "profit"]
    user_cols = ["sex", "age", "occupation", "actor1", "actor2"]
    item_cols = ["genre1", "genre2", "genre3", "profit"]
    _ = DatasetFeat.build_trainset(
        train_data=feature_data,
        sparse_col=sparse_cols,
        dense_col=dense_cols,
        user_col=user_cols,
        item_col=item_cols,
    )
    sparse_uniques = DatasetFeat.sparse_unique_vals

    assert DatasetFeat.sparse_col == sparse_cols
    assert DatasetFeat.dense_col == dense_cols
    assert_array_equal(DatasetFeat.sparse_unique_vals["sex"], ["F", "M"])
    assert_array_equal(sparse_uniques["occupation"], ["a", "b", "c"])
    assert_array_equal(sparse_uniques["genre1"], ["missing", "x", "y", "z"])
    assert_array_equal(sparse_uniques["genre2"], ["missing", "xx", "z"])
    assert_array_equal(sparse_uniques["genre3"], ["missing", "x", "y", "zz"])
    assert_array_equal(sparse_uniques["actor1"], [0, 11, 44, 77])
    assert_array_equal(sparse_uniques["actor2"], [0, 11, 22, 77, 99])
    assert DatasetFeat.multi_sparse_col is None
    assert DatasetFeat.multi_sparse_unique_vals is None

    transformed_single_indices = np.array(
        [
            [1, 0, 1, 1, 0],
            [2, 0, 0, 1, 0],
            [1, 0, 3, 2, 3],
            [0, 2, 1, 4, 3],
            [1, 2, 3, 1, 0],
            [1, 0, 1, 2, 0],
            [2, 2, 3, 1, 0],
        ]
    ).transpose()
    indices_from_sorted = get_sparse_indices_matrix(
        feature_data,
        sparse_cols,
        sparse_uniques,
        is_train=True,
        is_ordered=True,
    )
    indices_from_mapping = get_sparse_indices_matrix(
        feature_data,
        sparse_cols,
        sparse_uniques,
        is_train=True,
        is_ordered=False,
    )
    assert_array_equal(transformed_single_indices, indices_from_sorted)
    assert_array_equal(indices_from_sorted, indices_from_mapping)

    with pytest.raises(KeyError):
        column_sparse_indices(
            feature_data["sex"], ["M"], is_train=True, is_ordered=False
        )
    oov_mapping_indices = column_sparse_indices(
        feature_data["sex"], ["M"], is_train=False, is_ordered=False
    )
    assert_array_equal(oov_mapping_indices, [0, 1, 0, 0, 1])

    sparse_offsets = merge_offset(sparse_cols, None, sparse_uniques, None)
    assert_array_equal(sparse_offsets, [0, 3, 7, 12, 18, 23, 27])
    sparse_oovs = get_oov_pos(sparse_cols, None, sparse_uniques, None)
    assert_array_equal(sparse_oovs, [2, 6, 11, 17, 22, 26, 31])
    assert get_last_offset(sparse_cols, sparse_uniques) == 32

    transformed_indices = np.array(
        [
            [1, 0, 1, 1, 0],
            [5, 3, 3, 4, 3],
            [8, 7, 10, 9, 10],
            [12, 14, 13, 16, 15],
            [19, 20, 21, 19, 18],
            [24, 23, 24, 25, 23],
            [29, 29, 30, 28, 27],
        ]
    ).transpose()  # transposed
    indices_from_sorted = merge_sparse_indices(
        feature_data,
        sparse_cols,
        None,
        sparse_uniques,
        None,
        is_train=True,
        is_ordered=False,
    )
    indices_from_mapping = merge_sparse_indices(
        feature_data,
        sparse_cols,
        None,
        sparse_uniques,
        None,
        is_train=True,
        is_ordered=False,
    )
    assert_array_equal(transformed_indices, indices_from_sorted)
    assert_array_equal(indices_from_sorted, indices_from_mapping)


def test_multi_sparse_indices(feature_data):
    sparse_cols = ["sex", "occupation"]
    multi_sparse_cols = [["actor1", "actor2"], ["genre1", "genre2", "genre3"]]
    dense_cols = ["age", "profit"]
    user_cols = ["sex", "age", "occupation", "actor1", "actor2"]
    item_cols = ["genre1", "genre2", "genre3", "profit"]
    with pytest.raises(
        ValueError, match="Length of `multi_sparse_col` and `pad_val` doesn't match"
    ):
        _ = DatasetFeat.build_trainset(
            feature_data,
            sparse_col=sparse_cols,
            multi_sparse_col=multi_sparse_cols,
            dense_col=dense_cols,
            user_col=user_cols,
            item_col=item_cols,
            pad_val=["missing", "a", "b"],
        )

    _, data_info = DatasetFeat.build_trainset(
        train_data=feature_data,
        sparse_col=sparse_cols,
        multi_sparse_col=multi_sparse_cols,
        dense_col=dense_cols,
        user_col=user_cols,
        item_col=item_cols,
        pad_val=[0, "missing"],
    )
    sparse_uniques = DatasetFeat.sparse_unique_vals
    multi_sparse_uniques = DatasetFeat.multi_sparse_unique_vals

    recovered_sparse_cols, recovered_multi_sparse_cols = recover_sparse_cols(data_info)
    assert_array_equal(sparse_cols, recovered_sparse_cols)
    for i, j in zip(multi_sparse_cols, recovered_multi_sparse_cols):
        assert_array_equal(i, j)

    field_offset, field_len, feat_oov, pad_val = astuple(
        data_info.multi_sparse_combine_info
    )
    assert_array_equal(field_offset, [2, 4])
    assert_array_equal(field_len, [2, 3])
    assert_array_equal(feat_oov, [12, 18])
    assert pad_val == {"actor1": 0, "genre1": "missing"}

    assert DatasetFeat.multi_sparse_col == multi_sparse_cols
    assert 0 not in multi_sparse_uniques["actor1"]
    assert_array_equal(multi_sparse_uniques["actor1"], [11, 22, 44, 77, 99])
    assert "missing" not in multi_sparse_uniques["genre1"]
    assert_array_equal(multi_sparse_uniques["genre1"], ["x", "xx", "y", "z", "zz"])
    assert data_info.col_name_mapping["multi_sparse"] == {
        "genre2": "genre1",
        "genre3": "genre1",
        "actor2": "actor1",
    }

    transformed_single_indices = np.array(
        [
            [0, 5, 3, 2, 3],
            [5, 1, 0, 4, 3],
            [0, 2, 3, 0, 5],
            [1, 5, 1, 3, 5],
            [2, 2, 4, 0, 5],
        ]
    ).transpose()
    indices_from_sorted = get_multi_sparse_indices_matrix(
        feature_data,
        multi_sparse_cols,
        multi_sparse_uniques,
        is_train=True,
        is_ordered=True,
    )
    indices_from_mapping = get_multi_sparse_indices_matrix(
        feature_data,
        multi_sparse_cols,
        multi_sparse_uniques,
        is_train=True,
        is_ordered=False,
    )
    assert_array_equal(transformed_single_indices, indices_from_sorted)
    assert_array_equal(indices_from_sorted, indices_from_mapping)

    sparse_offsets = merge_offset(
        sparse_cols, multi_sparse_cols, sparse_uniques, multi_sparse_uniques
    )
    assert_array_equal(sparse_offsets, [0, 3, 7, 7, 13, 13, 13])
    sparse_oovs = get_oov_pos(
        sparse_cols, multi_sparse_cols, sparse_uniques, multi_sparse_uniques
    )
    assert_array_equal(sparse_oovs, [2, 6, 12, 12, 18, 18, 18])
    assert get_last_offset(sparse_cols, sparse_uniques) == 7

    transformed_indices = np.array(
        [
            [1, 0, 1, 1, 0],
            [5, 3, 3, 4, 3],
            [7, 12, 10, 9, 10],
            [12, 8, 7, 11, 10],
            [13, 15, 16, 13, 18],
            [14, 18, 14, 16, 18],
            [15, 15, 17, 13, 18],
        ]
    ).transpose()
    indices_from_sorted = merge_sparse_indices(
        feature_data,
        sparse_cols,
        multi_sparse_cols,
        sparse_uniques,
        multi_sparse_uniques,
        is_train=True,
        is_ordered=False,
    )
    indices_from_mapping = merge_sparse_indices(
        feature_data,
        sparse_cols,
        multi_sparse_cols,
        sparse_uniques,
        multi_sparse_uniques,
        is_train=True,
        is_ordered=False,
    )
    assert_array_equal(transformed_indices, indices_from_sorted)
    assert_array_equal(indices_from_sorted, indices_from_mapping)


@pytest.fixture
def feature_data_pair():
    data = pd.DataFrame(
        {
            "user": [4, 1, 10],
            "item": [1, 2, 8],
            "label": [1, 0, 1],
            "sex": ["M", "F", "M"],
            "occupation": ["c", "a", "a"],
            "age": [1, 2, 3],
            "actor1": [11, 0, 77],
            "actor2": [0, 22, 11],
            "genre1": ["missing", "y", "z"],
            "genre2": ["x", "missing", "x"],
            "genre3": ["y", "y", "z"],
        }
    )
    new_data = pd.DataFrame(
        {
            "user": [11, 1],
            "item": [4, 1],
            "label": [1, 0],
            "sex": ["M", "F"],
            "occupation": ["b", "d"],
            "age": [4, 5],
            "actor1": [11, 88],
            "actor2": [99, 0],  # oov value(0) for user 1, will not update
            "genre1": ["xx", "missing"],  # oov value for item 1, will not update
            "genre2": ["z", "yy"],
            "genre3": ["missing", "x"],
        }
    )
    sparse_cols = ["sex", "occupation"]
    multi_sparse_cols = [["actor1", "actor2"], ["genre1", "genre2", "genre3"]]
    dense_cols = ["age"]
    user_cols = ["sex", "age", "occupation", "actor1", "actor2"]
    item_cols = ["genre1", "genre2", "genre3"]
    _, data_info = DatasetFeat.build_trainset(
        train_data=data,
        sparse_col=sparse_cols,
        multi_sparse_col=multi_sparse_cols,
        dense_col=dense_cols,
        user_col=user_cols,
        item_col=item_cols,
        pad_val=[0, "missing"],
    )
    return sparse_cols, multi_sparse_cols, data_info, new_data


def test_update_features(feature_data_pair):
    sparse_cols, multi_sparse_cols, old_data_info, new_df = feature_data_pair

    sparse_uniques = DatasetFeat.sparse_unique_vals
    assert_array_equal(sparse_uniques["occupation"], ["a", "c"])
    with pytest.raises(ValueError, match="Old column .* doesn't exist in new data"):
        _ = update_sparse_unique(new_df.drop("sex", axis=1), old_data_info)

    new_sparse_uniques = update_sparse_unique(new_df, old_data_info)
    assert_array_equal(new_sparse_uniques["occupation"], ["a", "c", "b", "d"])

    multi_sparse_uniques = DatasetFeat.multi_sparse_unique_vals
    assert 0 not in multi_sparse_uniques["actor1"]
    assert_array_equal(multi_sparse_uniques["actor1"], [11, 22, 77])
    assert "missing" not in multi_sparse_uniques["genre1"]
    assert_array_equal(multi_sparse_uniques["genre1"], ["x", "y", "z"])
    with pytest.raises(ValueError, match="Old column .* doesn't exist in new data"):
        _ = update_sparse_unique(new_df.drop("occupation", axis=1), old_data_info)

    new_multi_sparse_uniques = update_multi_sparse_unique(new_df, old_data_info)
    assert 0 not in new_multi_sparse_uniques["actor1"]
    assert "missing" not in new_multi_sparse_uniques["genre1"]
    assert_array_equal(new_multi_sparse_uniques["actor1"], [11, 22, 77, 88, 99])
    assert_array_equal(new_multi_sparse_uniques["genre1"], ["x", "y", "z", "xx", "yy"])

    new_user_uniques, new_item_uniques = update_id_unique(new_df, old_data_info)
    assert_array_equal(new_user_uniques, [1, 4, 10, 11])
    assert_array_equal(new_item_uniques, [1, 2, 8, 4])

    assert_array_equal(
        old_data_info.user_sparse_unique,  # has oov, length is 4
        np.array(
            [
                [0, 3, 9, 7],
                [1, 4, 6, 9],
                [1, 3, 8, 6],
                [2, 5, 9, 9],
            ]
        ),
    )
    assert_array_equal(
        old_data_info.item_sparse_unique,
        np.array(
            [
                [13, 10, 11],
                [11, 13, 11],
                [12, 10, 12],
                [13, 13, 13],
            ]
        ),
    )

    sparse_offset = merge_offset(
        sparse_cols, multi_sparse_cols, new_sparse_uniques, new_multi_sparse_uniques
    )
    sparse_oov = get_oov_pos(
        sparse_cols, multi_sparse_cols, new_sparse_uniques, new_multi_sparse_uniques
    )
    assert_array_equal(sparse_offset, [0, 3, 8, 8, 14, 14, 14])
    assert_array_equal(sparse_oov, [2, 7, 13, 13, 19, 19, 19])

    new_user_sparse_uniques, new_user_dense_uniques = update_unique_feats(
        new_df,
        old_data_info,
        new_user_uniques,
        new_sparse_uniques,
        new_multi_sparse_uniques,
        sparse_offset,
        sparse_oov,
        is_user=True,
    )
    new_item_sparse_uniques, new_item_dense_uniques = update_unique_feats(
        new_df,
        old_data_info,
        new_item_uniques,
        new_sparse_uniques,
        new_multi_sparse_uniques,
        sparse_offset,
        sparse_oov,
        is_user=False,
    )
    # no oov at this step, length is still 4
    assert_array_equal(
        new_user_sparse_uniques,
        np.array(
            [
                [0, 6, 11, 9],
                [1, 4, 8, 13],
                [1, 3, 10, 8],
                [1, 5, 8, 12],
            ]
        ),
    )
    assert_array_equal(new_user_dense_uniques, np.array([[5], [1], [3], [4]]))
    assert_array_equal(
        new_item_sparse_uniques,
        np.array(
            [
                [19, 18, 14],
                [15, 19, 15],
                [16, 14, 16],
                [17, 16, 19],
            ]
        ),
    )
    assert new_item_dense_uniques is None


def test_assign_features(feature_data_pair):
    _, _, data_info, new_df = feature_data_pair
    old_info = store_old_info(data_info)
    assert old_info.n_users == 3
    assert old_info.n_items == 3

    new_df = new_df.drop("sex", axis=1)
    assert "sex" not in new_df

    new_df.loc[1, "actor1"] = 77
    data_info.assign_user_features(new_df)
    data_info.assign_item_features(new_df)
    assert_array_equal(
        data_info.user_sparse_unique,
        np.array(
            [
                [0, 3, 8, 7],
                [1, 4, 6, 9],
                [1, 3, 8, 6],
                [2, 5, 9, 9],
            ]
        ),
    )
    assert_array_equal(
        data_info.item_sparse_unique,
        np.array(
            [
                [13, 10, 10],
                [11, 13, 11],
                [12, 10, 12],
                [13, 13, 13],
            ]
        ),
    )


def test_get_features_from_data_info(feature_data_pair):
    _, _, data_info, _ = feature_data_pair
    _, _, sparse_indices, dense_values = get_original_feats(
        data_info, user=[2, 3], item=[0, 1], sparse=True, dense=True
    )
    assert_array_equal(
        sparse_indices,
        np.array([[1, 3, 8, 6, 13, 10, 11], [2, 5, 9, 9, 11, 13, 11]]),
    )
    assert_array_equal(dense_values, np.array([[3.0], [2.0]]))

    users, items, sparse_indices, dense_values = get_original_feats(
        data_info, user=0, item=0, sparse=True, dense=False
    )
    assert isinstance(users, list)
    assert isinstance(items, list)
    assert_array_equal(
        sparse_indices,
        np.array([[0, 3, 9, 7, 13, 10, 11]]),
    )
    assert dense_values is None

    _, _, sparse_indices, dense_values = get_original_feats(
        data_info, user=0, item=0, sparse=False, dense=True
    )
    assert sparse_indices is None
    assert_array_equal(dense_values, np.array([[2.0]]))


def test_set_temp_features(feature_data_pair):
    _, _, data_info, _ = feature_data_pair
    _, _, sparse_indices, dense_values = get_original_feats(
        data_info, user=[2], item=1, sparse=True, dense=True
    )
    feats = {"occupation": "xxx", "age": 10, "actor1": 111, "actor2": 77, "genre2": "x"}
    sparse_indices_new, dense_values_new = set_temp_feats(
        data_info, sparse_indices, dense_values, feats
    )
    assert_array_equal(sparse_indices, np.array([[1, 3, 8, 6, 11, 13, 11]]))
    assert_array_equal(dense_values, np.array([[3.0]]))
    assert_array_equal(sparse_indices_new, np.array([[1, 3, 8, 8, 11, 10, 11]]))
    assert_array_equal(dense_values_new, np.array([[10.0]]))


def test_features_from_batch(feature_data_pair):
    _, _, data_info, new_df = feature_data_pair
    with pytest.raises(ValueError, match="Column .* doesn't exist in data"):
        _ = features_from_batch(
            data_info, sparse=True, dense=True, data=new_df.drop("sex", axis=1)
        )
    with pytest.raises(ValueError, match="Column .* doesn't exist in data"):
        _ = features_from_batch(
            data_info, sparse=True, dense=True, data=new_df.drop("actor1", axis=1)
        )

    sparse_indices, dense_values = features_from_batch(
        data_info, sparse=True, dense=True, data=new_df
    )
    assert_array_equal(
        sparse_indices,
        np.array([[1, 5, 6, 9, 13, 12, 13], [0, 5, 9, 9, 13, 13, 10]]),
    )
    assert_array_equal(dense_values, np.array([[4.0], [5.0]]))


def test_get_recommend_features(feature_data_pair):
    _, _, data_info, _ = feature_data_pair
    sparse_indices, dense_values = get_rec_feats(
        data_info, user=0, n_items=3, sparse=True, dense=True
    )
    assert_array_equal(
        sparse_indices,
        np.array(
            [
                [0, 3, 9, 7, 13, 10, 11],
                [0, 3, 9, 7, 11, 13, 11],
                [0, 3, 9, 7, 12, 10, 12],
            ]
        ),
    )
    assert_array_equal(dense_values, np.array([[2.0], [2.0], [2.0]]))

    sparse_indices, dense_values = get_rec_feats(
        data_info, user=3, n_items=3, sparse=True, dense=False
    )
    assert_array_equal(
        sparse_indices,
        np.array(
            [
                [2, 5, 9, 9, 13, 10, 11],
                [2, 5, 9, 9, 11, 13, 11],
                [2, 5, 9, 9, 12, 10, 12],
            ]
        ),
    )
    assert dense_values is None

    sparse_indices, dense_values = get_rec_feats(
        data_info, user=2, n_items=1, sparse=False, dense=True
    )
    assert sparse_indices is None
    assert_array_equal(dense_values, np.array([[3.0]]))


def test_process_embed_feat(feature_data_pair):
    _, _, data_info, _ = feature_data_pair
    user_id = np.array([1])
    user_feats = {"sex": "out", "occ": "a", "actor1": "out", "actor2": 77, "age": 11}
    sparse_indices, dense_values = process_embed_feat(data_info, user_id, user_feats)

    assert_array_equal(sparse_indices, np.array([[1, 4, 6, 8]]))
    assert_array_equal(dense_values, np.array([[11.0]]))
