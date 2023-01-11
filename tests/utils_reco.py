import numpy as np
import pandas as pd
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
    user1, user2 = 1, 2
    reco_take_one = model.recommend_user(user=user1, n_rec=7)[user1]
    reco_take_two = model.recommend_user(user=user2, n_rec=7)[user2]
    assert len(reco_take_one) == len(reco_take_two) == 7
    # assert reco_take_one != reco_take_two
    assert not recommend_in_former_consumed(data_info, reco_take_one, 1)
    assert not recommend_in_former_consumed(data_info, reco_take_two, 2)

    assert len(model.recommend_user(user=-1, n_rec=10, cold_start="popular")[-1]) == 10

    cold_user1, cold_user2 = -99999, -1
    cold_reco1 = model.recommend_user(user=cold_user1, n_rec=3)[cold_user1]
    cold_reco2 = model.recommend_user(user=cold_user2, n_rec=3)[cold_user2]
    # np.testing.assert_array_equal(cold_reco1, cold_reco2)
    assert len(cold_reco1) == len(cold_reco2) == 3
    assert np.any(np.sort(cold_reco1) != np.sort(cold_reco2))

    random_reco = model.recommend_user(
        user=1, n_rec=10, filter_consumed=False, random_rec=True
    )[1]
    no_random_reco = model.recommend_user(
        user=1, n_rec=10, filter_consumed=False, random_rec=False
    )[1]
    assert np.any(random_reco != no_random_reco)

    batch_recs = model.recommend_user(
        user=[1, 2, 3, -1],
        n_rec=3,
        filter_consumed=True,
        random_rec=False,
        cold_start="popular",
    )
    assert len(batch_recs[1]) == len(batch_recs[2]) == len(batch_recs[3])
    assert np.all(np.isin(batch_recs[-1], data_info.popular_items))

    if with_feats:
        model.recommend_user(
            user=2211,
            n_rec=7,
            inner_id=False,
            cold_start="average",
            user_feats=pd.Series({"sex": "F", "occupation": 2, "age": 23}),
            item_data=pd_data.iloc[4:10],
        )
        # fails in batch recommend with provided features
        with pytest.raises(ValueError):
            model.recommend_user(
                user=[1, 2, 3],
                n_rec=7,
                inner_id=False,
                cold_start="average",
                user_feats=pd.Series({"sex": "F", "occupation": 2, "age": 23}),
                item_data=pd_data.iloc[4:10],
            )
