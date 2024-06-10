import sys

import numpy as np
import pytest

from libreco.algorithms import RsItemCF
from tests.utils_data import remove_path, set_ranking_labels
from tests.utils_metrics import get_metrics
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_recommends
from tests.utils_save_load import save_load_model


@pytest.mark.parametrize("task", ["rating", "ranking"])
@pytest.mark.parametrize("neg_sampling", [True, False, None])
def test_item_cf_rs(pure_data_small, task, neg_sampling):
    if sys.version_info[:2] < (3, 7):
        pytest.skip("Rust implementation only supports Python >= 3.7.")

    pd_data, train_data, eval_data, data_info = pure_data_small
    if neg_sampling is False:
        set_ranking_labels(train_data)
        set_ranking_labels(eval_data)

    model = RsItemCF(task=task, data_info=data_info, k_sim=20)
    if neg_sampling is None:
        with pytest.raises(AssertionError):
            model.fit(train_data, neg_sampling)
    elif task == "rating" and neg_sampling:
        with pytest.raises(ValueError):
            model.fit(train_data, neg_sampling)
    else:
        model.fit(
            train_data,
            neg_sampling,
            verbose=2,
            eval_data=eval_data,
            metrics=get_metrics(task),
            k=10,
            eval_user_num=100,
        )
        ptest_preds(model, task, pd_data, with_feats=False)
        ptest_recommends(model, data_info, pd_data, with_feats=False)
        model.recommend_user(1, 10, random_rec=True)
        with pytest.raises(ValueError):
            model.predict(user="cold user1", item="cold item2", cold_start="other")

        # test save and load model
        loaded_model, loaded_data_info = save_load_model(RsItemCF, model, data_info)
        ptest_preds(loaded_model, task, pd_data, with_feats=False)
        ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=False)
        model.save("not_existed_path", "item_cf2")
        remove_path("not_existed_path")


@pytest.mark.skipif(
    sys.version_info[:2] < (3, 7),
    reason="Rust implementation only supports Python >= 3.7.",
)
def test_all_consumed_recommend(all_consumed_data, monkeypatch):
    _, train_data, eval_data, data_info = all_consumed_data
    set_ranking_labels(train_data)
    set_ranking_labels(eval_data)

    model = RsItemCF(task="ranking", data_info=data_info)
    model.fit(train_data, neg_sampling=False, verbose=0)
    model.save("not_existed_path", "item_cf2")
    remove_path("not_existed_path")
    for user in range(model.n_users):
        model.rs_model.user_consumed = {user: list(range(model.n_items))}
        recos = model.recommend_user(user=user, n_rec=7)
        assert np.all(np.isin(recos[user], data_info.popular_items))
