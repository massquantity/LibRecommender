import json
import os

import numpy as np
import pytest

# noinspection PyUnresolvedReferences
from tensorflow.core.protobuf.meta_graph_pb2 import MetaGraphDef

from libreco.bases import CfBase, TfBase
from libreco.tfops import tf
from libserving.serialization import (
    embed2redis,
    knn2redis,
    online2redis,
    save_embed,
    save_knn,
    save_online,
    save_tf,
    tf2redis,
)
from tests.utils_data import SAVE_PATH


@pytest.mark.parametrize("knn_model", ["UserCF", "ItemCF"], indirect=True)
def test_knn_serialization(knn_model, redis_client):
    assert isinstance(knn_model, CfBase)
    save_knn(SAVE_PATH, knn_model, k=10)
    knn2redis(SAVE_PATH)
    check_model_name(SAVE_PATH, knn_model, redis_client)
    check_id_mapping(SAVE_PATH, knn_model, redis_client)
    check_user_consumed(SAVE_PATH, knn_model, redis_client)

    sim_path = os.path.join(SAVE_PATH, "sim.json")
    with open(sim_path) as f:
        k_sims = json.load(f, object_hook=lambda d: {int(k): v for k, v in d.items()})
    model_num = (
        knn_model.n_users if knn_model.model_name == "UserCF" else knn_model.n_items
    )
    assert len(k_sims) == model_num
    assert min(k_sims.keys()) == 0
    assert max(k_sims.keys()) == model_num - 1
    k_sims_redis = load_from_redis(redis_client, name="k_sims", mode="hlist")
    assert k_sims == k_sims_redis


def test_embed_serialization(embed_model, redis_client):
    save_embed(SAVE_PATH, embed_model)
    embed2redis(SAVE_PATH)
    check_model_name(SAVE_PATH, embed_model, redis_client)
    check_id_mapping(SAVE_PATH, embed_model, redis_client)
    check_user_consumed(SAVE_PATH, embed_model, redis_client)
    user_embed_path = os.path.join(SAVE_PATH, "user_embed.json")
    item_embed_path = os.path.join(SAVE_PATH, "item_embed.json")
    with open(user_embed_path) as f1, open(item_embed_path) as f2:
        user_embed = json.load(
            f1, object_hook=lambda d: {int(k): v for k, v in d.items()}
        )
        item_embed = json.load(f2)
    assert len(user_embed) == embed_model.n_users
    assert len(item_embed) == embed_model.n_items
    user_embed_redis = load_from_redis(redis_client, name="user_embed", mode="hlist")
    assert user_embed == user_embed_redis


@pytest.mark.parametrize("tf_model", ["pure", "feat"], indirect=True)
def test_tf_serialization(tf_model, redis_client):
    assert isinstance(tf_model, TfBase)
    save_tf(SAVE_PATH, tf_model, version=1)
    tf2redis(SAVE_PATH)
    check_model_name(SAVE_PATH, tf_model, redis_client)
    check_id_mapping(SAVE_PATH, tf_model, redis_client)
    check_user_consumed(SAVE_PATH, tf_model, redis_client)
    check_features(SAVE_PATH, tf_model, redis_client)

    model_name = tf_model.model_name.lower()
    SAVE_MODEL_PATH = os.path.join(SAVE_PATH, model_name, "1")
    with tf.Session(graph=tf.Graph()) as sess:
        loaded_model = tf.saved_model.load(
            sess, [tf.saved_model.tag_constants.SERVING], SAVE_MODEL_PATH
        )
    assert isinstance(loaded_model, MetaGraphDef)


@pytest.mark.parametrize(
    "online_model",
    ["pure", "user_feat", "separate", "multi_sparse", "item_feat", "all"],
    indirect=True,
)
def test_online_serialization(online_model, redis_client):
    save_online(SAVE_PATH, online_model, version=1)
    online2redis(SAVE_PATH)
    check_model_name(SAVE_PATH, online_model, redis_client)
    check_id_mapping(SAVE_PATH, online_model, redis_client)
    check_user_consumed(SAVE_PATH, online_model, redis_client)
    check_features(SAVE_PATH, online_model, redis_client)
    check_user_sparse_mapping(online_model, redis_client)


def check_model_name(path, model, redis_client):
    model_name_path = os.path.join(path, "model_name.json")
    with open(model_name_path) as f:
        m = json.load(f)
    assert model.model_name == m["model_name"]

    model_name_redis = redis_client.get("model_name")
    assert model_name_redis == m["model_name"]


def check_id_mapping(path, model, redis_client):
    user2id_path = os.path.join(path, "user2id.json")
    with open(user2id_path) as f:
        user2id = json.load(
            f, object_hook=lambda d: {int(k): int(v) for k, v in d.items()}
        )
    assert len(user2id) == model.n_users
    assert min(user2id.values()) == 0
    assert max(user2id.values()) == model.n_users - 1

    user2id_redis = load_from_redis(redis_client, name="user2id", mode="hdict")
    assert user2id_redis == user2id

    id2item_path = os.path.join(path, "id2item.json")
    with open(id2item_path) as f:
        id2item = json.load(
            f, object_hook=lambda d: {int(k): int(v) for k, v in d.items()}
        )
    assert len(id2item) == model.n_items
    assert min(id2item) == 0
    assert max(id2item) == model.n_items - 1

    id2item_redis = load_from_redis(redis_client, name="id2item", mode="hdict")
    assert id2item_redis == id2item


def check_user_consumed(path, model, redis_client):
    user_consumed_path = os.path.join(path, "user_consumed.json")
    with open(user_consumed_path) as f:
        user_consumed = json.load(
            f, object_hook=lambda d: {int(k): v for k, v in d.items()}
        )
    assert len(user_consumed) == model.n_users
    assert min(user_consumed) == 0
    assert max(user_consumed) == model.n_users - 1

    user_consumed_redis = load_from_redis(
        redis_client, name="user_consumed", mode="hlist"
    )
    assert user_consumed_redis == user_consumed


def check_features(path, model, redis_client):
    data_info = model.data_info
    feature_path = os.path.join(path, "features.json")
    with open(feature_path) as f:
        feats = json.load(f)

    assert feats["n_users"] == data_info.n_users
    assert feats["n_items"] == data_info.n_items
    n_items_redis = load_from_redis(redis_client, name="n_items", mode="dict")
    assert n_items_redis == data_info.n_items

    if hasattr(model, "max_seq_len"):
        assert feats["max_seq_len"] == model.max_seq_len
        max_seq_len_redis = load_from_redis(
            redis_client, name="max_seq_len", mode="dict"
        )
        assert max_seq_len_redis == model.max_seq_len

    if "user_sparse_col_index" in feats:
        assert len(feats["user_sparse_col_index"]) == len(feats["user_sparse_values"][0])  # fmt: skip
        assert len(feats["user_sparse_values"]) == data_info.n_users + 1

        user_sparse_col_index_redis = load_from_redis(
            redis_client, name="user_sparse_col_index", mode="dict-list"
        )
        user_sparse_values_redis = load_from_redis(
            redis_client, name="user_sparse_values", mode="hlist-value"
        )
        assert user_sparse_col_index_redis == feats["user_sparse_col_index"]
        assert user_sparse_values_redis == feats["user_sparse_values"]
        assert redis_client.hexists("feature", "user_sparse")

    if "item_sparse_col_index" in feats:
        assert len(feats["item_sparse_col_index"]) == len(feats["item_sparse_values"][0])  # fmt: skip
        assert len(feats["item_sparse_values"]) == data_info.n_items + 1

        item_sparse_col_index_redis = load_from_redis(
            redis_client, name="item_sparse_col_index", mode="dict-list"
        )
        item_sparse_values_redis = load_from_redis(
            redis_client, name="item_sparse_values", mode="list"
        )
        assert item_sparse_col_index_redis == feats["item_sparse_col_index"]
        assert item_sparse_values_redis == feats["item_sparse_values"]
        assert redis_client.hexists("feature", "item_sparse")

    if "user_dense_col_index" in feats:
        assert len(feats["user_dense_col_index"]) == len(feats["user_dense_values"][0])
        assert len(feats["user_dense_values"]) == data_info.n_users + 1

        user_dense_col_index_redis = load_from_redis(
            redis_client, name="user_dense_col_index", mode="dict-list"
        )
        user_dense_values_redis = load_from_redis(
            redis_client, name="user_dense_values", mode="hlist-value"
        )
        assert user_dense_col_index_redis == feats["user_dense_col_index"]
        assert user_dense_values_redis == feats["user_dense_values"]
        assert redis_client.hexists("feature", "user_dense")

    if "item_dense_col_index" in feats:
        assert len(feats["item_dense_col_index"]) == len(feats["item_dense_values"][0])
        assert len(feats["item_dense_values"]) == data_info.n_items + 1

        item_dense_col_index_redis = load_from_redis(
            redis_client, name="item_dense_col_index", mode="dict-list"
        )
        item_dense_values_redis = load_from_redis(
            redis_client, name="item_dense_values", mode="list"
        )
        assert item_dense_col_index_redis == feats["item_dense_col_index"]
        assert item_dense_values_redis == feats["item_dense_values"]
        assert redis_client.hexists("feature", "item_dense")


def load_from_redis(redis_client, name, mode):
    if redis_client.exists(name):
        if mode == "dict":
            d = redis_client.get(name)
            return int(d)
        elif mode == "dict-list":
            d = redis_client.get(name)
            return json.loads(d)
        elif mode == "hdict":
            d = redis_client.hgetall(name)
            return {int(k): int(v) for k, v in d.items()}
        elif mode == "hlist":
            d = redis_client.hgetall(name)
            return {int(k): json.loads(v) for k, v in d.items()}
        elif mode == "hlist-value":
            d = redis_client.hgetall(name)
            keys = sorted(int(i) for i in d)
            return [json.loads(d[str(i)]) for i in keys]
        elif mode == "list":
            d = redis_client.lrange(name, 0, -1)
            return [json.loads(v) for v in d]
        else:
            raise ValueError(f"unknown mode: {mode}")
    else:
        raise KeyError(f"{name} doesn't exist in redis")


def check_user_sparse_mapping(model, redis_client):
    data_info = model.data_info
    if data_info.user_sparse_col.name:
        for field_idx, col in enumerate(data_info.user_sparse_col.name):
            assert field_idx == int(redis_client.hget("user_sparse_fields", col))
            idx_mapping_redis = redis_client.hgetall(f"user_sparse_idx_mapping__{col}")
            idx_mapping_redis = {k: int(v) for k, v in idx_mapping_redis.items()}
            assert get_val_idx_mapping(data_info, col) == idx_mapping_redis


def get_val_idx_mapping(data_info, col):
    col_mapping = data_info.col_name_mapping
    sparse_idx_mapping = data_info.sparse_idx_mapping  # {col: {val: idx}}
    if "multi_sparse" in col_mapping and col in col_mapping["multi_sparse"]:
        main_col = col_mapping["multi_sparse"][col]
        idx_mapping = sparse_idx_mapping[main_col]
    else:
        idx_mapping = sparse_idx_mapping[col]

    all_field_idx = col_mapping["sparse_col"][col]
    feat_offset = data_info.sparse_offset[all_field_idx]
    val_idx_mapping = dict()
    for val, idx in idx_mapping.items():
        val = val.item() if isinstance(val, np.integer) else val
        idx = int(idx + feat_offset)
        val_idx_mapping[val] = idx
    return val_idx_mapping
