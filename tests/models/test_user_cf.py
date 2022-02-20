import pytest

from libreco.algorithms import UserCF
from tests.utils_data import prepare_pure_data
from tests.utils_reco import recommend_in_former_consumed


@pytest.mark.parametrize("task", ["rating", "ranking"])
@pytest.mark.parametrize("sim_type", ["cosine", "pearson", "jaccard"])
def test_user_cf(prepare_pure_data, task, sim_type):
    pd_data, train_data, eval_data, data_info = prepare_pure_data
    if task == "ranking":
        train_data.build_negative_samples(data_info, item_gen_mode="random",
                                          num_neg=1, seed=2022)
        eval_data.build_negative_samples(data_info, item_gen_mode="random",
                                         num_neg=1, seed=2222)
    metrics = (
        ["rmse", "mae", "r2"]
        if task == "rating"
        else ["roc_auc", "precision", "ndcg"]
    )
    model = UserCF(
        task=task,
        data_info=data_info,
        sim_type=sim_type,
        k=10
    )
    model.fit(
        train_data,
        verbose=2,
        eval_data=eval_data,
        metrics=metrics
    )
    pred = model.predict(user=1, item=2333)
    # prediction in range
    if task == "rating":
        assert 1 <= pred <= 5
    else:
        assert 0 <= pred <= 1

    cold_pred1 = model.predict(user="cold user1", item="cold item2")
    cold_pred2 = model.predict(user="cold user2", item="cold item2")
    assert cold_pred1 == cold_pred2

    # cold start strategy
    with pytest.raises(ValueError):
        model.recommend_user(user=-99999, n_rec=7, cold_start="sss")
    # different recommendation for different users
    reco_take_one = [i[0] for i in model.recommend_user(user=1, n_rec=7)]
    reco_take_two = [i[0] for i in model.recommend_user(user=2, n_rec=7)]
    assert len(reco_take_one) == len(reco_take_two) == 7
    assert reco_take_one != reco_take_two
    assert not recommend_in_former_consumed(data_info, reco_take_one, 1)
    assert not recommend_in_former_consumed(data_info, reco_take_two, 2)
    cold_reco1 = model.recommend_user(user=-99999, n_rec=3)
    cold_reco2 = model.recommend_user(user=-1, n_rec=3)
    assert cold_reco1 == cold_reco2
