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
    reco_take_one = [i for i in model.recommend_user(user=1, n_rec=7)]
    reco_take_two = [i for i in model.recommend_user(user=2, n_rec=7)]
    assert len(reco_take_one) == len(reco_take_two) == 7
    # assert reco_take_one != reco_take_two
    assert not recommend_in_former_consumed(data_info, reco_take_one, 1)
    assert not recommend_in_former_consumed(data_info, reco_take_two, 2)
    model.recommend_user(user=-1, n_rec=10, cold_start="popular")

    cold_reco1 = model.recommend_user(user=-99999, n_rec=3)
    cold_reco2 = model.recommend_user(user=-1, n_rec=3)
    np.testing.assert_array_equal(cold_reco1, cold_reco2)

    random_reco = model.recommend_user(
        user=1, n_rec=10, filter_consumed=False, random_rec=True
    )
    no_random_reco = model.recommend_user(
        user=1, n_rec=10, filter_consumed=False, random_rec=False
    )
    assert np.any(np.sort(random_reco) != np.sort(no_random_reco))

    if with_feats:
        model.recommend_user(
            user=2211,
            n_rec=7,
            inner_id=False,
            cold_start="average",
            user_feats=pd.Series({"sex": "F", "occupation": 2, "age": 23}),
            item_data=pd_data.iloc[4:10],
        )
