import os.path
from collections import Counter
from io import StringIO

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from libreco.data import DatasetPure, DatasetFeat, TransformedSet, DataInfo


sparse_col = ["sex", "occupation", "genre1", "genre2", "genre3"]
dense_col = ["age"]
user_col = ["sex", "age", "occupation"]
item_col = ["genre1", "genre2", "genre3"]

raw_data = StringIO("""
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
""")
pd_data = pd.read_csv(raw_data, header=0)


@pytest.fixture(params=[pd_data])
def pure_train_data(request):
    return DatasetPure.build_trainset(request.param)


@pytest.fixture(params=[{
    "train_data": pd_data,
    "sparse_col": sparse_col,
    "dense_col": dense_col,
    "user_col": user_col,
    "item_col": item_col
}])
def feat_train_data(request):
    return DatasetFeat.build_trainset(**request.param)


def test_dataset_pure(pure_train_data):
    data, data_info = pure_train_data
    assert DatasetPure.train_called
    np.testing.assert_array_equal(DatasetPure.user_unique_vals, [
        242, 263, 1298, 2137, 2211, 2461, 3706, 4585, 4617, 5184
    ])

    assert isinstance(data, TransformedSet)
    assert isinstance(data_info, DataInfo)
    with pytest.raises(IndexError):
        assert len(data) == 10
        _ = data[11]
    with pytest.raises(AssertionError):
        _ = DatasetPure.build_trainset(pd_data[["user", "item"]])


def test_negative_samples(pure_train_data):
    data, data_info = pure_train_data
    data.build_negative_samples(data_info, num_neg=5)
    label_frequency = list(Counter(data.labels).values())
    data_len = len(pd_data)
    np.testing.assert_array_equal(label_frequency, [data_len, data_len * 5])


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
    assert len(data_info.col_name_mapping["item_dense_col"]) == 0
    # pprint.pprint(data_info.col_name_mapping)
    with pytest.raises(IndexError):
        _ = data_info.user_consumed[-999][0]
    assert data_info.user_sparse_col.name == ["sex", "occupation"]
    assert data_info.user_dense_col.name == ["age"]
    assert data_info.item_sparse_col.name == ["genre1", "genre2", "genre3"]
    assert data_info.item_dense_col.name == []
    assert data_info.n_users == data_info.n_items == data_info.data_size == 10
    assert data_info.item2id[208] == 0
    with pytest.raises(KeyError):
        _ = data_info.user2id[-999]

    # test save load DataInfo
    data_info.save(os.path.curdir)
    data_info2 = DataInfo.load(os.path.curdir)
    os.remove(os.path.join(os.path.curdir, "data_info.npz"))
    os.remove(os.path.join(os.path.curdir, "data_info_name_mapping.json"))
    assert data_info2.data_size == 10
    assert data_info.col_name_mapping == data_info2.col_name_mapping
