import numpy as np
import pandas as pd
import pytest

from libreco.algorithms import Swing
from libreco.data import DatasetPure
from tests.utils_data import SAVE_PATH, remove_path, set_ranking_labels
from tests.utils_metrics import get_metrics
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_recommends
from tests.utils_save_load import save_load_model


@pytest.mark.parametrize("task", ["rating", "ranking"])
@pytest.mark.parametrize("neg_sampling", [True, False, None])
def test_swing(all_consumed_data, task, neg_sampling):
    pd_data, train_data, eval_data, data_info = all_consumed_data
    if neg_sampling is False:
        set_ranking_labels(train_data)
        set_ranking_labels(eval_data)

    if neg_sampling is None:
        with pytest.raises(AssertionError):
            model = Swing(task=task, data_info=data_info)
            model.fit(train_data, neg_sampling)
    elif task == "rating":
        with pytest.raises(AssertionError):
            _ = Swing(task=task, data_info=data_info)
    else:
        model = Swing(task=task, data_info=data_info)
        model.fit(
            train_data,
            neg_sampling,
            verbose=2,
            eval_data=eval_data,
            metrics=get_metrics(task),
            k=10,
            eval_user_num=200,
        )
        ptest_preds(model, task, pd_data, with_feats=False)
        ptest_recommends(model, data_info, pd_data, with_feats=False)
        model.recommend_user(1, 10, random_rec=True)
        with pytest.raises(ValueError):
            model.predict(user="cold user1", item="cold item2", cold_start="other")
        with pytest.raises(TypeError):
            model.recommend_user(1, 7, seq=[1, 2, 3])

        # test save and load model
        loaded_model, loaded_data_info = save_load_model(Swing, model, data_info)
        ptest_preds(loaded_model, task, pd_data, with_feats=False)
        ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=False)
        remove_path(SAVE_PATH)


def test_all_consumed_recommend(all_consumed_data, monkeypatch):
    _, train_data, eval_data, data_info = all_consumed_data

    model = Swing(task="ranking", data_info=data_info)
    model.fit(train_data, neg_sampling=True, verbose=0)
    model.save("not_existed_path", "swing")
    remove_path("not_existed_path")
    user = 1
    model.rs_model.user_consumed = {user: list(range(model.n_items))}
    recos = model.recommend_user(user=user, n_rec=7)
    assert np.all(np.isin(recos[user], data_info.popular_items))


def test_cold_start():
    size, unique_num = 50000, 1000
    out_id = 1001
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
    model = Swing(task="ranking", data_info=data_info)
    model.fit(train_data, neg_sampling=False, verbose=0)
    recos = model.recommend_user(out_id, n_rec=7, filter_consumed=True)
    assert len(data_info.popular_items) == 100
    assert np.all(np.isin(recos[out_id], data_info.popular_items))
