import numpy as np
import pytest


def recommend_in_former_consumed(data_info, reco, user):
    user_id = data_info.user2id[user]
    user_consumed = data_info.user_consumed[user_id]
    user_consumed_id = [data_info.id2item[i] for i in user_consumed]
    return any([r in user_consumed_id for r in reco])


def ptest_recommends(model, data_info, pd_data, with_feats):
    # cold start strategy
    with pytest.raises(ValueError):
        model.recommend_user(user=-99999, n_rec=7, cold_start="sss")
    # different recommendation for different users
    users = pd_data.user.tolist()
    user1, user2 = users[0], users[1]
    reco_take_one = model.recommend_user(user=user1, n_rec=7)[user1]
    reco_take_two = model.recommend_user(user=user2, n_rec=7)[user2]
    assert len(reco_take_one) == len(reco_take_two) == 7
    # assert reco_take_one != reco_take_two
    assert not recommend_in_former_consumed(data_info, reco_take_one, user1)
    assert not recommend_in_former_consumed(data_info, reco_take_two, user2)

    assert len(model.recommend_user(user=-1, n_rec=10, cold_start="popular")[-1]) == 10

    cold_user1, cold_user2 = -99999, -1
    cold_reco1 = model.recommend_user(user=cold_user1, n_rec=3)[cold_user1]
    cold_reco2 = model.recommend_user(user=cold_user2, n_rec=3)[cold_user2]
    assert len(cold_reco1) == len(cold_reco2) == 3
    if hasattr(model, "default_recs") and model.default_recs is not None:
        default_recs = model.default_recs
    else:
        default_recs = [data_info.item2id[i] for i in data_info.popular_items]
    cold_reco1 = [data_info.item2id[i] for i in cold_reco1]
    cold_reco2 = [data_info.item2id[i] for i in cold_reco2]
    assert np.all(np.isin(cold_reco1, default_recs))
    assert np.all(np.isin(cold_reco2, default_recs))

    u1, u2, u3 = users[2], users[3], users[4]
    random_reco = model.recommend_user(
        user=u1, n_rec=10, filter_consumed=False, random_rec=True
    )
    no_random_reco = model.recommend_user(
        user=u1, n_rec=10, filter_consumed=False, random_rec=False
    )
    assert len(random_reco[u1]) == len(no_random_reco[u1]) == 10

    batch_recs = model.recommend_user(
        user=[u1, u2, u3, -1],
        n_rec=3,
        filter_consumed=True,
        random_rec=False,
        cold_start="popular",
    )
    assert len(batch_recs[u1]) == len(batch_recs[u2]) == len(batch_recs[u3])
    assert np.all(np.isin(batch_recs[-1], data_info.popular_items))

    if with_feats:
        model.recommend_user(
            user=u3,
            n_rec=7,
            inner_id=False,
            cold_start="average",
            user_feats={"sex": "F", "occupation": 2, "age": 23},
        )
        # fails in batch recommend with provided features
        with pytest.raises(ValueError):
            model.recommend_user(
                user=[u1, u2, u3],
                n_rec=7,
                inner_id=False,
                cold_start="average",
                user_feats={"sex": "F", "occupation": 2, "age": 23},
            )


def ptest_seq_recommends(model, pd_data):
    users = pd_data.user.tolist()
    user1, user2 = users[0], users[1]
    with pytest.raises(
        ValueError,
        match="Batch recommend doesn't support arbitrary item sequence*",
    ):
        model.recommend_user([user1, user2], 3, seq=[1, 2, 3])
    with pytest.raises(AssertionError):
        model.recommend_user(user1, 3, seq=(1, 2))

    seq1 = [1, 23, "898", 0, -1, -3, 7]
    seq2 = []
    reco_take_one = model.recommend_user(user=user1, n_rec=7, seq=seq1)[user1]
    reco_take_two = model.recommend_user(user=user2, n_rec=7, seq=seq2)[user2]
    reco_take_the = model.recommend_user(user=[user2], n_rec=7, seq=seq2)[user2]
    assert len(reco_take_one) == len(reco_take_two) == len(reco_take_the) == 7
    return reco_take_one
