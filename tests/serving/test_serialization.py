import json
import os

import pytest
import tensorflow
from tensorflow.core.protobuf.meta_graph_pb2 import MetaGraphDef

from libreco.algorithms import ALS, DIN, ItemCF, NCF, UserCF
from libserving.serialization import (
    embed2redis,
    knn2redis,
    save_embed,
    save_knn,
    save_tf,
    tf2redis,
)
from tests.utils_path import SAVE_PATH, remove_path


def test_knn_serialization(knn_models, redis_client):
    user_cf_model, item_cf_model = knn_models
    # userCF
    save_knn(SAVE_PATH, user_cf_model, k=10)
    knn2redis(SAVE_PATH)
    check_model_name(SAVE_PATH, user_cf_model, redis_client)
    check_id_mapping(SAVE_PATH, user_cf_model, redis_client)
    check_user_consumed(SAVE_PATH, user_cf_model, redis_client)

    sim_path = os.path.join(SAVE_PATH, "sim.json")
    with open(sim_path) as f:
        k_sims = json.load(f, object_hook=lambda d: {int(k): v for k, v in d.items()})
    assert len(k_sims) == user_cf_model.n_users
    assert min(k_sims.keys()) == 0
    assert max(k_sims.keys()) == user_cf_model.n_users - 1
    k_sims_redis = load_from_redis(redis_client, name="k_sims", mode="hlist")
    assert k_sims == k_sims_redis

    # itemCF
    redis_client.flushdb()
    save_knn(SAVE_PATH, item_cf_model, k=10)
    knn2redis(SAVE_PATH)
    check_model_name(SAVE_PATH, item_cf_model, redis_client)
    check_id_mapping(SAVE_PATH, item_cf_model, redis_client)
    check_user_consumed(SAVE_PATH, item_cf_model, redis_client)

    sim_path = os.path.join(SAVE_PATH, "sim.json")
    with open(sim_path) as f:
        k_sims = json.load(f, object_hook=lambda d: {int(k): v for k, v in d.items()})
    assert len(k_sims) == item_cf_model.n_items
    assert min(k_sims.keys()) == 0
    assert max(k_sims.keys()) == item_cf_model.n_items - 1
    k_sims_redis = load_from_redis(redis_client, name="k_sims", mode="hlist")
    assert k_sims == k_sims_redis


@pytest.fixture
def knn_models(prepare_pure_data):
    _, train_data, _, data_info = prepare_pure_data
    train_data.build_negative_samples(data_info, seed=2022)
    user_cf_model = UserCF("ranking", data_info)
    user_cf_model.fit(train_data, verbose=2)
    item_cf_model = ItemCF("ranking", data_info)
    item_cf_model.fit(train_data, verbose=2)
    return user_cf_model, item_cf_model


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


@pytest.fixture
def embed_model(prepare_pure_data):
    _, train_data, _, data_info = prepare_pure_data
    train_data.build_negative_samples(data_info, seed=2022)
    model = ALS("ranking", data_info, n_epochs=1, use_cg=False, reg=0.1)
    model.fit(train_data, verbose=2)
    return model


def test_tf_pure_serialization(ncf_model, redis_client):
    tf = tensorflow.compat.v1
    save_tf(SAVE_PATH, ncf_model, version=1)
    tf2redis(SAVE_PATH)
    check_model_name(SAVE_PATH, ncf_model, redis_client)
    check_id_mapping(SAVE_PATH, ncf_model, redis_client)
    check_user_consumed(SAVE_PATH, ncf_model, redis_client)
    check_features(SAVE_PATH, ncf_model, redis_client)

    SAVE_MODEL_PATH = os.path.join(SAVE_PATH, "ncf", "1")
    with tf.Session(graph=tf.Graph()) as sess:
        loaded_model = tf.saved_model.load(
            sess, [tf.saved_model.tag_constants.SERVING], SAVE_MODEL_PATH
        )
    assert isinstance(loaded_model, MetaGraphDef)


@pytest.fixture
def ncf_model(prepare_pure_data):
    tensorflow.compat.v1.reset_default_graph()
    remove_path(SAVE_PATH)
    _, train_data, _, data_info = prepare_pure_data
    train_data.build_negative_samples(data_info, seed=2022)
    model = NCF("ranking", data_info, n_epochs=1, batch_size=2048)
    model.fit(train_data, verbose=2)
    return model


def test_tf_feat_serialization(din_model, redis_client):
    tf = tensorflow.compat.v1
    save_tf(SAVE_PATH, din_model, version=1)
    tf2redis(SAVE_PATH)
    check_model_name(SAVE_PATH, din_model, redis_client)
    check_id_mapping(SAVE_PATH, din_model, redis_client)
    check_user_consumed(SAVE_PATH, din_model, redis_client)
    check_features(SAVE_PATH, din_model, redis_client)

    SAVE_MODEL_PATH = os.path.join(SAVE_PATH, "din", "1")
    with tf.Session(graph=tf.Graph()) as sess:
        loaded_model = tf.saved_model.load(
            sess, [tf.saved_model.tag_constants.SERVING], SAVE_MODEL_PATH
        )
    assert isinstance(loaded_model, MetaGraphDef)


@pytest.fixture
def din_model(prepare_feat_data):
    tensorflow.compat.v1.reset_default_graph()
    remove_path(SAVE_PATH)
    _, train_data, _, data_info = prepare_feat_data
    train_data.build_negative_samples(data_info, seed=2022)
    model = DIN("ranking", data_info, n_epochs=1, batch_size=2048)
    model.fit(train_data, verbose=2)
    return model


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
        assert len(feats["user_sparse_col_index"]) == len(
            feats["user_sparse_values"][0]
        )
        assert len(feats["user_sparse_values"]) == data_info.n_users

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
        assert len(feats["item_sparse_col_index"]) == len(
            feats["item_sparse_values"][0]
        )
        assert len(feats["item_sparse_values"]) == data_info.n_items

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
        assert len(feats["user_dense_values"]) == data_info.n_users

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
        assert len(feats["item_dense_values"]) == data_info.n_items

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
