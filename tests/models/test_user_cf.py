import numpy as np
import pandas as pd
import pytest

from libreco.algorithms import UserCF
from libreco.data import DatasetPure
from tests.utils_data import remove_path, set_ranking_labels
from tests.utils_metrics import get_metrics
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_recommends
from tests.utils_save_load import save_load_model


@pytest.mark.parametrize("task", ["rating", "ranking"])
@pytest.mark.parametrize("sim_type", ["cosine", "pearson", "jaccard", "unknown"])
@pytest.mark.parametrize("store_top_k", [True, False])
@pytest.mark.parametrize("neg_sampling", [True, False, None])
def test_user_cf(pure_data_small, task, sim_type, store_top_k, neg_sampling):
    pd_data, train_data, eval_data, data_info = pure_data_small
    if neg_sampling is False:
        set_ranking_labels(train_data)
        set_ranking_labels(eval_data)

    model = UserCF(
        task=task,
        data_info=data_info,
        sim_type=sim_type,
        k_sim=20,
        store_top_k=store_top_k,
    )

    if neg_sampling is None:
        with pytest.raises(AssertionError):
            model.fit(train_data, neg_sampling)
    elif task == "rating" and neg_sampling:
        with pytest.raises(ValueError):
            model.fit(train_data, neg_sampling)
    elif sim_type == "unknown":
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
        loaded_model, loaded_data_info = save_load_model(UserCF, model, data_info)
        ptest_preds(loaded_model, task, pd_data, with_feats=False)
        ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=False)
        model.save("not_existed_path", "user_cf2")
        remove_path("not_existed_path")

        with pytest.raises(NotImplementedError):
            model.rebuild_model("model_path", "user_cf")


def test_all_consumed_recommend(pure_data_small, monkeypatch):
    _, train_data, eval_data, data_info = pure_data_small
    set_ranking_labels(train_data)
    set_ranking_labels(eval_data)

    model = UserCF(task="ranking", data_info=data_info)
    model.fit(train_data, neg_sampling=False, verbose=0)
    model.save("not_existed_path", "user_cf2")
    remove_path("not_existed_path")
    with monkeypatch.context() as m:
        user = 1
        m.setitem(model.user_consumed, user, list(range(model.n_items)))
        recos = model.recommend_user(user, n_rec=7)
        assert np.all(np.isin(recos[user], data_info.popular_items))


def test_no_sim_recommend(pure_data_small):
    size, unique_num = 50000, 1000
    out_id = 1001
    out_inner_id = out_id - 1
    np_rng = np.random.default_rng(999)
    train_data = pd.DataFrame(
        {
            "user": np_rng.integers(1, unique_num, size, endpoint=True),
            "item": np_rng.integers(1, unique_num, size, endpoint=True),
            "label": np_rng.integers(0, 1, size, endpoint=True),
        }
    )
    train_data.loc[size] = [out_id, out_id, 1]
    train_data, data_info = DatasetPure.build_trainset(train_data)
    model = UserCF(task="ranking", data_info=data_info)
    model.fit(train_data, neg_sampling=False, verbose=0)
    recos = model.recommend_user(out_id, n_rec=7, filter_consumed=True)
    assert len(data_info.popular_items) == 100
    assert np.all(np.isin(recos[out_id], data_info.popular_items))
    # no sim users
    indptr = model.sim_matrix.indptr
    assert indptr[out_inner_id] == indptr[out_inner_id + 1]
