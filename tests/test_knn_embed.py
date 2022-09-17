import pytest

from libreco.algorithms import ALS, RNN4Rec


def test_knn_embed(prepare_pure_data):
    pd_data, train_data, eval_data, data_info = prepare_pure_data
    train_data.build_negative_samples(data_info, seed=2022)
    eval_data.build_negative_samples(data_info, seed=2222)

    als = ALS("ranking", data_info, embed_size=16, n_epochs=2, reg=5.0)
    als.fit(train_data, verbose=2, shuffle=True)
    ptest_knn(als)

    rnn = RNN4Rec(
        "ranking",
        data_info,
        rnn_type="lstm",
        loss_type="cross_entropy",
        embed_size=16,
        n_epochs=4,
    )
    rnn.fit(train_data, verbose=2, shuffle=True)
    ptest_knn(rnn)

    with pytest.raises(ValueError):
        rnn.get_user_id(-1)
    with pytest.raises(ValueError):
        rnn.get_item_id(-1)


def ptest_knn(model):
    assert model.get_user_embedding().shape[0] == model.n_users
    assert model.get_user_embedding().shape[1] == model.embed_size
    assert model.get_item_embedding().shape[0] == model.n_items
    assert model.get_item_embedding().shape[1] == model.embed_size
    with pytest.raises(ValueError):
        model.init_knn(approximate=True, sim_type="whatever")

    model.init_knn(approximate=True, sim_type="cosine")
    sim_cosine_users_approx = model.search_knn_users(5000, 10)
    sim_cosine_items_approx = model.search_knn_items(3000, 10)
    model.init_knn(approximate=False, sim_type="cosine")
    sim_cosine_users = model.search_knn_users(5000, 10)
    sim_cosine_items = model.search_knn_items(3000, 10)
    assert compare_diff(sim_cosine_users_approx, sim_cosine_users) <= 5
    assert compare_diff(sim_cosine_items_approx, sim_cosine_items) <= 5
    assert model.sim_type == "cosine"

    model.init_knn(approximate=True, sim_type="inner-product")
    sim_ip_users_approx = model.search_knn_users(5000, 10)
    sim_ip_items_approx = model.search_knn_items(3000, 10)
    model.init_knn(approximate=False, sim_type="inner-product")
    sim_ip_users = model.search_knn_users(5000, 10)
    sim_ip_items = model.search_knn_items(3000, 10)
    assert compare_diff(sim_ip_users_approx, sim_ip_users) <= 5
    assert compare_diff(sim_ip_items_approx, sim_ip_items) <= 5
    assert model.sim_type == "inner-product"


def compare_diff(a, b):
    diff = set(a).symmetric_difference(b)
    return len(diff)
