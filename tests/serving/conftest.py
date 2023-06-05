import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import redis
import tensorflow as tf

from libreco.algorithms import ALS, DIN, NCF, ItemCF, UserCF
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
def close_server():
    yield
    subprocess.run(["pkill", "sanic"], check=False)
    subprocess.run("kill $(lsof -t -i:8501 -sTCP:LISTEN)", shell=True, check=False)
    r = redis.Redis()
    r.flushdb()
    r.close()
    time.sleep(1)


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
        pd_data["item_dense_feat"] = np.random.random(len(pd_data))
        train_data, data_info = DatasetFeat.build_trainset(pd_data, **features)
        model = DIN("ranking", data_info, n_epochs=1, batch_size=2048)
        model.fit(train_data, neg_sampling=True, verbose=2)
        return model
