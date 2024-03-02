import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import redis
import requests
import tensorflow as tf
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from libreco.algorithms import (
    ALS,
    DIN,
    FM,
    NCF,
    ItemCF,
    RNN4Rec,
    TwoTower,
    UserCF,
    WideDeep,
    YouTubeRetrieval,
)
from libreco.data import DatasetFeat
from tests.utils_data import SAVE_PATH, remove_path


@pytest.fixture
def redis_client():
    pool = redis.ConnectionPool(host="localhost", port=6379, decode_responses=True)
    r = redis.Redis(connection_pool=pool)
    r.flushdb()
    yield r
    r.flushdb()
    r.close()


@pytest.fixture
def session():
    Retry.DEFAULT_BACKOFF_MAX = 0.8
    retries = Retry(total=50, backoff_factor=0.08)
    s = requests.Session()
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    yield s
    s.close()


@pytest.fixture
def close_server():
    yield
    # subprocess.run("kill $(lsof -t -i:8000 -sTCP:LISTEN)", shell=True, check=True)
    subprocess.run(["pkill", "sanic"], check=True)
    subprocess.run("kill $(lsof -t -i:8501 -sTCP:LISTEN)", shell=True, check=False)
    r = redis.Redis()
    r.flushdb()
    r.close()
    # time.sleep(1)


@pytest.fixture
def knn_model(prepare_pure_data, request):
    _, train_data, _, data_info = prepare_pure_data
    if request.param == "UserCF":
        user_cf_model = UserCF("ranking", data_info)
        user_cf_model.fit(train_data, neg_sampling=True, verbose=2)
        return user_cf_model
    elif request.param == "ItemCF":
        item_cf_model = ItemCF("ranking", data_info)
        item_cf_model.fit(train_data, neg_sampling=True, verbose=2)
        return item_cf_model


@pytest.fixture
def embed_model(prepare_pure_data):
    _, train_data, _, data_info = prepare_pure_data
    model = ALS("ranking", data_info, n_epochs=1, use_cg=False, reg=0.1)
    model.fit(train_data, neg_sampling=True, verbose=2)
    return model


@pytest.fixture
def tf_model(prepare_pure_data, request):
    tf.compat.v1.reset_default_graph()
    remove_path(SAVE_PATH)
    if request.param == "pure":
        _, train_data, _, data_info = prepare_pure_data
        model = NCF("ranking", data_info, n_epochs=1, batch_size=2048)
        model.fit(train_data, neg_sampling=True, verbose=2)
        return model
    else:
        if "user" in request.param:
            features = {
                "sparse_col": ["sex", "occupation"],
                "dense_col": ["age"],
                "user_col": ["sex", "age", "occupation"],
            }
        elif "item" in request.param:
            features = {
                "sparse_col": ["genre1", "genre2", "genre3"],
                "dense_col": ["item_dense_feat"],
                "item_col": ["genre1", "genre2", "genre3", "item_dense_feat"],
            }
        else:
            features = {
                "sparse_col": ["sex", "occupation", "genre1", "genre2", "genre3"],
                "dense_col": ["age", "item_dense_feat"],
                "user_col": ["sex", "age", "occupation"],
                "item_col": ["genre1", "genre2", "genre3", "item_dense_feat"],
            }

        data_path = (
            Path(__file__).parents[1] / "sample_data" / "sample_movielens_merged.csv"
        )
        pd_data = pd.read_csv(data_path, sep=",", header=0)
        pd_data["item_dense_feat"] = np.random.default_rng(42).random(len(pd_data))
        train_data, data_info = DatasetFeat.build_trainset(pd_data, **features)
        model = DIN("ranking", data_info, n_epochs=1, batch_size=2048)
        model.fit(train_data, neg_sampling=True, verbose=2)
        return model


@pytest.fixture
def online_model(make_synthetic_data, request):
    tf.compat.v1.reset_default_graph()

    pd_data = make_synthetic_data
    if request.param == "pure":
        features = dict()
        model_cls = RNN4Rec
    elif request.param == "user_feat":
        features = {
            "sparse_col": ["sex", "occupation"],
            "dense_col": ["age"],
            "user_col": ["sex", "occupation", "age"],
        }
        model_cls = YouTubeRetrieval
    elif request.param == "separate":
        features = {
            "sparse_col": ["genre3", "genre2", "occupation", "genre1", "sex"],
            "dense_col": ["age", "profit"],
            "user_col": ["age", "sex", "occupation"],
            "item_col": ["genre1", "genre3", "genre2", "profit"],
        }
        model_cls = TwoTower
    elif request.param == "multi_sparse":
        features = {
            "sparse_col": ["sex", "occupation"],
            "multi_sparse_col": [["genre1", "genre2", "genre3"]],
            "dense_col": ["age", "profit"],
            "user_col": ["genre1", "genre2", "genre3", "sex"],
            "item_col": ["age", "occupation", "profit"],
        }
        model_cls = WideDeep
    elif request.param == "item_feat":
        features = {
            "sparse_col": ["genre3", "genre2", "sex", "genre1"],
            "dense_col": ["age", "profit"],
            "user_col": ["age"],
            "item_col": ["genre1", "genre3", "genre2", "sex", "profit"],
        }
        model_cls = FM
    elif request.param == "all":
        features = {
            "sparse_col": ["genre3", "genre2", "sex", "occupation", "genre1"],
            "dense_col": ["age", "profit"],
            "user_col": ["sex", "age", "occupation"],
            "item_col": ["genre1", "genre3", "genre2", "profit"],
        }
        model_cls = DIN
    else:
        raise ValueError(f"Unknown type `{request.param}`")

    train_data, data_info = DatasetFeat.build_trainset(
        train_data=pd_data,
        pad_val=["missing"],
        **features,
    )
    model = model_cls("ranking", data_info, embed_size=4, n_epochs=1, batch_size=50)
    model.fit(train_data, neg_sampling=True, verbose=2)
    yield model
    remove_path(SAVE_PATH)
