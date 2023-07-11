import sys

import pytest

from libreco.algorithms import ALS, LightGCN, RNN4Rec
from tests.utils_data import set_ranking_labels


@pytest.mark.skipif(
    sys.platform.startswith("win") or sys.platform.startswith("darwin"),
    reason="Possible issue on Windows and MaxOS platform using `nmslib`",
)
@pytest.mark.skipif(
    sys.version_info[:2] >= (3, 10),
    reason="Possible issue for Python >= 3.10 using `nmslib`",
)
def test_knn_embed(pure_data_small, monkeypatch):
    pd_data, train_data, eval_data, data_info = pure_data_small
    set_ranking_labels(train_data)
    set_ranking_labels(eval_data)

    als = ALS("ranking", data_info, embed_size=16, n_epochs=2, reg=5.0)
    als.fit(train_data, neg_sampling=True, verbose=2, shuffle=True)
    ptest_knn(als, pd_data)

    lightgcn = LightGCN(
        "ranking",
        data_info,
        loss_type="cross_entropy",
        embed_size=8,
        n_epochs=1,
        lr=1e-4,
        batch_size=20,
    )
    lightgcn.fit(train_data, neg_sampling=True, verbose=2, shuffle=True)
    ptest_knn(lightgcn, pd_data)

    with pytest.raises(ValueError):
        lightgcn.get_user_id(-1)
    with pytest.raises(ValueError):
        lightgcn.get_item_id(-1)

    # test `approximate=True` when `nmslib` is unavailable
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "nmslib", None)
        with pytest.raises((ImportError, ModuleNotFoundError)):
            lightgcn.init_knn(approximate=True, sim_type="cosine")


def ptest_knn(model, pd_data):
    assert model.get_user_embedding().shape[0] == model.n_users
    assert model.get_user_embedding().shape[1] == model.embed_size
    assert model.get_item_embedding().shape[0] == model.n_items
    assert model.get_item_embedding().shape[1] == model.embed_size
    with pytest.raises(ValueError):
        model.init_knn(approximate=True, sim_type="whatever")

    user = pd_data.user[0].item()
    item = pd_data.item[0].item()
    model.init_knn(approximate=True, sim_type="cosine")
    sim_cosine_users_approx = model.search_knn_users(user, 10)
    sim_cosine_items_approx = model.search_knn_items(item, 10)
    model.init_knn(approximate=False, sim_type="cosine")
    sim_cosine_users = model.search_knn_users(user, 10)
    sim_cosine_items = model.search_knn_items(item, 10)
    assert compare_diff(sim_cosine_users_approx, sim_cosine_users) <= 1
    assert compare_diff(sim_cosine_items_approx, sim_cosine_items) <= 1
    assert model.sim_type == "cosine"

    model.init_knn(approximate=True, sim_type="inner-product")
    sim_ip_users_approx = model.search_knn_users(user, 10)
    sim_ip_items_approx = model.search_knn_items(item, 10)
    model.init_knn(approximate=False, sim_type="inner-product")
    sim_ip_users = model.search_knn_users(user, 10)
    sim_ip_items = model.search_knn_items(item, 10)
    assert compare_diff(sim_ip_users_approx, sim_ip_users) <= 1
    assert compare_diff(sim_ip_items_approx, sim_ip_items) <= 1
    assert model.sim_type == "inner-product"


def compare_diff(a, b):
    diff = set(a).symmetric_difference(b)
    return len(diff)


def test_get_embeddings(pure_data_small):
    pd_data, train_data, eval_data, data_info = pure_data_small

    model = RNN4Rec("ranking", data_info, embed_size=16, n_epochs=1)
    model.fit(train_data, neg_sampling=True, verbose=2, shuffle=True)
    assert model.get_user_embedding().shape[0] == model.n_users
    assert model.get_user_embedding(include_bias=False).shape[1] == model.embed_size
    assert model.get_user_embedding(include_bias=True).shape[1] == model.embed_size + 1

    assert model.get_user_embedding(user=1, include_bias=False).size == model.embed_size
    assert model.get_user_embedding(user=2, include_bias=True).size == model.embed_size + 1  # fmt: skip

    assert model.get_item_embedding().shape[0] == model.n_items
    assert model.get_item_embedding(include_bias=False).shape[1] == model.embed_size
    assert model.get_item_embedding(include_bias=True).shape[1] == model.embed_size + 1

    assert model.get_item_embedding(item=3, include_bias=False).size == model.embed_size
    assert model.get_item_embedding(item=4, include_bias=True).size == model.embed_size + 1  # fmt: skip

    all_embed = model.get_user_embedding(user=None)
    dyn_embed = model.dyn_user_embedding(user=None)
    assert all_embed.shape == dyn_embed.shape
    with pytest.raises(ValueError):
        model.dyn_user_embedding(user=[1, 2, 3], seq=[0, 0])

    # test squeezed single user embedding
    user = pd_data.user.tolist()[0]
    user_embed = model.dyn_user_embedding(user, seq=[0])
    assert len(user_embed) == dyn_embed.shape[1]

    user_embed = model.dyn_user_embedding(
        user, user_feats={"aa": "bb"}, seq=[0, 10], include_bias=True, inner_id=True
    )
    assert len(user_embed) == dyn_embed.shape[1] + 1
