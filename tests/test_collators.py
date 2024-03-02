from io import StringIO

import numpy as np
import pandas as pd
import pytest
import torch

from libreco.algorithms import DIN, LightGCN, PinSageDGL, RNN4Rec
from libreco.batch.batch_data import BatchData
from libreco.batch.batch_unit import (
    PairFeats,
    PairwiseBatch,
    PointwiseBatch,
    PointwiseSepFeatBatch,
    SparseBatch,
    TripleFeats,
)
from libreco.batch.collators import BaseCollator as NormalCollator
from libreco.batch.collators import (
    GraphDGLCollator,
    PairwiseCollator,
    PointwiseCollator,
    SparseCollator,
)
from libreco.batch.enums import Backend
from libreco.data import DatasetFeat
from libreco.graph.message import ItemMessageDGL, UserMessage
from libreco.sampling.negatives import negatives_from_unconsumed
from libreco.tfops import tf

raw_data = """
user,item,label,time,sex,age,occupation,genre1,genre2,genre3
1,296,2,964138229,F,25,6,crime,drama,missing
1,297,2,964138229,F,25,6,crime,drama,missing
1298,208,4,974849526,M,35,6,action,adventure,missing
2,1769,4,964322774,M,35,7,action,thriller,missing
1298,933,4,974607346,M,45,6,romance,missing,missing
3706,1136,5,966376465,M,25,12,comedy,missing,missing
2137,1215,3,974640099,F,1,10,action,adventure,comedy
2,1257,4,974170662,M,18,4,comedy,missing,missing
242,3148,3,977854274,F,18,4,drama,missing,missing
2211,932,4,974607346,M,45,6,romance,missing,missing
263,2115,2,976651827,F,25,7,action,adventure,missing
1,291,2,964138229,F,25,6,crime,drama,missing
5184,866,5,961735308,M,18,20,crime,drama,romance
"""


@pytest.fixture
def config_feat_data(request):
    pd_data = pd.read_csv(StringIO(raw_data), sep=",", header=0)
    pd_data["item_dense_col"] = np.random.default_rng(42).random(len(pd_data))
    train_data, data_info = DatasetFeat.build_trainset(
        train_data=pd_data, **request.param
    )
    return train_data, data_info


@pytest.mark.parametrize(
    "config_feat_data",
    [
        {"sparse_col": [], "item_col": None},
        {"sparse_col": ["sex"], "dense_col": ["age"], "user_col": ["sex", "age"]},
        {
            "sparse_col": ["genre1"],
            "dense_col": ["item_dense_col"],
            "item_col": ["genre1", "item_dense_col"],
        },
        {
            "sparse_col": ["sex"],
            "dense_col": ["item_dense_col"],
            "user_col": ["sex"],
            "item_col": ["item_dense_col"],
        },
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
def test_normal_collator(config_feat_data):
    train_data, data_info = config_feat_data
    tf_model = DIN("ranking", data_info, sampler=None)
    torch_model = LightGCN("ranking", data_info, sampler=None)
    original_data = BatchData(train_data, use_features=True)[[11, 7, 2]]

    tf_collator = NormalCollator(
        tf_model, data_info, Backend.TF, separate_features=False
    )
    tf_batch = tf_collator(original_data)
    assert isinstance(tf_batch, PointwiseBatch)
    assert len(tf_batch.labels) == 3
    assert isinstance(tf_batch.items, np.ndarray)
    if tf_batch.sparse_indices is not None:
        assert isinstance(tf_batch.sparse_indices, np.ndarray)
        assert tf_batch.sparse_indices.shape[1] == len(data_info.sparse_col.index)
    if tf_batch.dense_values is not None:
        assert isinstance(tf_batch.dense_values, np.ndarray)
        assert tf_batch.dense_values.shape[1] == len(data_info.dense_col.index)
    assert isinstance(tf_batch.seqs.interacted_seq, np.ndarray)
    assert tf_batch.seqs.interacted_seq.shape == (3, 10)
    assert isinstance(tf_batch.seqs.interacted_len, np.ndarray)
    assert tf_batch.seqs.interacted_len.shape == (3,)
    tf_batch.seqs.repeat(3)
    assert tf_batch.seqs.interacted_seq.shape == (9, 10)
    assert tf_batch.seqs.interacted_len.shape == (9,)
    tf.reset_default_graph()

    torch_collator = NormalCollator(
        torch_model, data_info, Backend.TORCH, separate_features=True
    )
    torch_batch = torch_collator(original_data)
    assert isinstance(torch_batch, PointwiseSepFeatBatch)
    assert len(torch_batch.labels) == 3
    assert isinstance(torch_batch.items, torch.Tensor)
    if torch_batch.sparse_indices is not None:
        assert isinstance(torch_batch.sparse_indices, PairFeats)
    user_sparse_len = len(data_info.user_sparse_col.index)
    if user_sparse_len > 0:
        assert isinstance(torch_batch.sparse_indices.user_feats, torch.Tensor)
        assert torch_batch.sparse_indices.user_feats.shape[1] == user_sparse_len
    item_sparse_len = len(data_info.item_sparse_col.index)
    if item_sparse_len > 0:
        assert isinstance(torch_batch.sparse_indices.item_feats, torch.Tensor)
        assert torch_batch.sparse_indices.item_feats.shape[1] == item_sparse_len
    user_dense_len = len(data_info.user_dense_col.index)
    if user_dense_len:
        assert isinstance(torch_batch.dense_values.user_feats, torch.Tensor)
        assert torch_batch.dense_values.user_feats.shape[1] == user_dense_len
    item_dense_len = len(data_info.item_dense_col.index)
    if item_dense_len > 0:
        assert isinstance(torch_batch.dense_values.item_feats, torch.Tensor)
        assert torch_batch.dense_values.item_feats.shape[1] == item_dense_len


@pytest.mark.parametrize(
    "config_feat_data",
    [
        {"sparse_col": [], "item_col": None},
        {"sparse_col": ["sex"], "dense_col": ["age"], "user_col": ["sex", "age"]},
        {
            "sparse_col": ["genre1"],
            "dense_col": ["item_dense_col"],
            "item_col": ["genre1", "item_dense_col"],
        },
        {
            "sparse_col": ["sex"],
            "dense_col": ["item_dense_col"],
            "user_col": ["sex"],
            "item_col": ["item_dense_col"],
        },
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
def test_sparse_collator(config_feat_data):
    train_data, data_info = config_feat_data
    tf_model = DIN("ranking", data_info, sampler=None)
    original_data = BatchData(train_data, use_features=True)[[11, 7, 2]]

    tf_collator = SparseCollator(tf_model, data_info, Backend.TF)
    tf_batch = tf_collator(original_data)
    assert isinstance(tf_batch, SparseBatch)
    assert len(tf_batch.items) == 3
    assert isinstance(tf_batch.items, np.ndarray)
    if tf_batch.sparse_indices is not None:
        assert isinstance(tf_batch.sparse_indices, np.ndarray)
        assert tf_batch.sparse_indices.shape[1] == len(data_info.sparse_col.index)
    if tf_batch.dense_values is not None:
        assert isinstance(tf_batch.dense_values, np.ndarray)
        assert tf_batch.dense_values.shape[1] == len(data_info.dense_col.index)
    assert isinstance(tf_batch.seqs.interacted_indices, np.ndarray)
    assert tf_batch.seqs.interacted_indices.shape == (3, 2)
    assert isinstance(tf_batch.seqs.interacted_values, np.ndarray)
    assert tf_batch.seqs.interacted_values.shape == (3,)
    tf.reset_default_graph()


@pytest.mark.parametrize(
    "config_feat_data",
    [
        {"sparse_col": [], "item_col": None},
        {"sparse_col": ["sex"], "dense_col": ["age"], "user_col": ["sex", "age"]},
        {
            "sparse_col": ["genre1"],
            "dense_col": ["item_dense_col"],
            "item_col": ["genre1", "item_dense_col"],
        },
        {
            "sparse_col": ["sex"],
            "dense_col": ["item_dense_col"],
            "user_col": ["sex"],
            "item_col": ["item_dense_col"],
        },
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
def test_pointwise_collator(config_feat_data):
    train_data, data_info = config_feat_data
    tf_model = DIN("ranking", data_info, "cross_entropy", sampler="random", num_neg=2)
    torch_model = LightGCN("ranking", data_info, "focal", sampler="random", num_neg=3)
    original_data = BatchData(train_data, use_features=True)[[11, 7, 2]]

    tf_collator = PointwiseCollator(tf_model, data_info, Backend.TF)
    tf_batch = tf_collator(original_data)
    assert isinstance(tf_batch, PointwiseBatch)
    assert len(tf_batch.users) == len(tf_batch.items) == len(tf_batch.labels) == 9
    assert isinstance(tf_batch.items, np.ndarray)
    assert np.all(tf_batch.labels[0::3] == 1.0)
    assert np.all(tf_batch.labels[1::3] == 0.0)
    assert np.all(tf_batch.labels[2::3] == 0.0)
    if tf_batch.sparse_indices is not None:
        assert isinstance(tf_batch.sparse_indices, np.ndarray)
        assert tf_batch.sparse_indices.shape[1] == len(data_info.sparse_col.index)
    if tf_batch.dense_values is not None:
        assert isinstance(tf_batch.dense_values, np.ndarray)
        assert tf_batch.dense_values.shape[1] == len(data_info.dense_col.index)
    assert isinstance(tf_batch.seqs.interacted_seq, np.ndarray)
    assert tf_batch.seqs.interacted_seq.shape == (9, 10)
    assert isinstance(tf_batch.seqs.interacted_len, np.ndarray)
    assert tf_batch.seqs.interacted_len.shape == (9,)
    tf.reset_default_graph()

    torch_collator = PointwiseCollator(torch_model, data_info, Backend.TORCH)
    torch_batch = torch_collator(original_data)
    assert isinstance(torch_batch, PointwiseBatch)
    assert len(torch_batch.users) == len(torch_batch.labels) == 12
    assert isinstance(torch_batch.items, torch.Tensor)
    sparse_len = len(data_info.sparse_col.index)
    if sparse_len > 0:
        assert isinstance(torch_batch.sparse_indices, torch.Tensor)
        assert torch_batch.sparse_indices.shape[1] == sparse_len
    dense_len = len(data_info.dense_col.index)
    if dense_len > 0:
        assert isinstance(torch_batch.dense_values, torch.Tensor)
        assert torch_batch.dense_values.shape[1] == dense_len


@pytest.mark.parametrize(
    "config_feat_data",
    [
        {"sparse_col": [], "item_col": None},
        {"sparse_col": ["sex"], "dense_col": ["age"], "user_col": ["sex", "age"]},
        {
            "sparse_col": ["genre1"],
            "dense_col": ["item_dense_col"],
            "item_col": ["genre1", "item_dense_col"],
        },
        {
            "sparse_col": ["sex"],
            "dense_col": ["item_dense_col"],
            "user_col": ["sex"],
            "item_col": ["item_dense_col"],
        },
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
def test_pairwise_collator(config_feat_data):
    train_data, data_info = config_feat_data
    tf_model = RNN4Rec("ranking", data_info, "bpr", num_neg=2)
    torch_model = LightGCN("ranking", data_info, "max_margin", num_neg=3)
    original_data = BatchData(train_data, use_features=True)[[11, 7, 2]]

    tf_collator = PairwiseCollator(
        tf_model, data_info, Backend.TF, repeat_positives=True
    )
    tf_batch = tf_collator(original_data)
    assert isinstance(tf_batch, PairwiseBatch)
    assert len(tf_batch.queries) == 6
    assert len(tf_batch.item_pairs[0]) == len(tf_batch.item_pairs[1]) == 6
    assert isinstance(tf_batch.queries, np.ndarray)
    assert np.all(tf_batch.queries[:2] == tf_batch.queries[0])
    assert np.all(tf_batch.item_pairs[0][:2] == tf_batch.item_pairs[0][:1])
    if tf_batch.sparse_indices is not None:
        assert isinstance(tf_batch.sparse_indices, TripleFeats)
    user_sparse_len = len(data_info.user_sparse_col.index)
    if user_sparse_len > 0:
        assert isinstance(tf_batch.sparse_indices.query_feats, np.ndarray)
        assert tf_batch.sparse_indices.query_feats.shape[1] == user_sparse_len
    item_sparse_len = len(data_info.item_sparse_col.index)
    if item_sparse_len > 0:
        assert isinstance(tf_batch.sparse_indices.item_pos_feats, np.ndarray)
        assert tf_batch.sparse_indices.item_pos_feats.shape[1] == item_sparse_len
    user_dense_len = len(data_info.user_dense_col.index)
    if user_dense_len:
        assert isinstance(tf_batch.dense_values.query_feats, np.ndarray)
        assert tf_batch.dense_values.query_feats.shape[1] == user_dense_len
    item_dense_len = len(data_info.item_dense_col.index)
    if item_dense_len > 0:
        assert isinstance(tf_batch.dense_values.item_neg_feats, np.ndarray)
        assert tf_batch.dense_values.item_neg_feats.shape[1] == item_dense_len
    assert isinstance(tf_batch.seqs.interacted_seq, np.ndarray)
    assert tf_batch.seqs.interacted_seq.shape == (6, 10)
    assert isinstance(tf_batch.seqs.interacted_len, np.ndarray)
    assert tf_batch.seqs.interacted_len.shape == (6,)
    tf.reset_default_graph()

    torch_collator = PairwiseCollator(
        torch_model, data_info, Backend.TORCH, repeat_positives=False
    )
    torch_batch = torch_collator(original_data)
    assert isinstance(torch_batch, PairwiseBatch)
    assert len(torch_batch.queries) == len(torch_batch.item_pairs[0]) == 3
    assert len(torch_batch.item_pairs[1]) == 9
    assert isinstance(torch_batch.queries, torch.Tensor)


@pytest.mark.parametrize(
    "config_feat_data",
    [
        {"sparse_col": [], "item_col": None},
        {"sparse_col": ["sex"], "dense_col": ["age"], "user_col": ["sex", "age"]},
        {
            "sparse_col": ["genre1"],
            "dense_col": ["item_dense_col"],
            "item_col": ["genre1", "item_dense_col"],
        },
        {
            "sparse_col": ["sex"],
            "dense_col": ["item_dense_col"],
            "user_col": ["sex"],
            "item_col": ["item_dense_col"],
        },
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
def test_graph_collator(config_feat_data):
    train_data, data_info = config_feat_data
    original_data = BatchData(train_data, use_features=True)[[11, 7, 2]]

    u2i_model = PinSageDGL("ranking", data_info, "bpr", paradigm="u2i", num_neg=3)
    u2i_model.build_model()
    collator = GraphDGLCollator(u2i_model, data_info, Backend.TORCH)
    user_data, item_data, *_ = collator(original_data)
    assert isinstance(user_data, UserMessage)
    assert isinstance(item_data, ItemMessageDGL)
    assert isinstance(user_data.users, torch.Tensor)
    assert isinstance(item_data.items, torch.Tensor)
    assert len(user_data.users) == 3

    i2i_model = PinSageDGL("ranking", data_info, "focal", paradigm="i2i", num_neg=1)
    i2i_model.build_model()
    collator = GraphDGLCollator(i2i_model, data_info, Backend.TORCH)
    item_data, *_ = collator(original_data)
    assert isinstance(item_data, ItemMessageDGL)


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
