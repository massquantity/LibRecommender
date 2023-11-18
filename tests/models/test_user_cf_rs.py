import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from libreco.algorithms import RsUserCF
from libreco.data import DataInfo, DatasetPure, split_by_ratio_chrono
from libreco.evaluation import evaluate
from tests.utils_data import SAVE_PATH, remove_path, set_ranking_labels
from tests.utils_metrics import get_metrics
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_recommends
from tests.utils_save_load import save_load_model


@pytest.mark.parametrize("task", ["rating", "ranking"])
@pytest.mark.parametrize("neg_sampling", [True, False, None])
def test_user_cf_rs(pure_data_small, task, neg_sampling):
    if sys.version_info[:2] < (3, 7):
        pytest.skip("Rust implementation only supports Python >= 3.7.")

    pd_data, train_data, eval_data, data_info = pure_data_small
    if neg_sampling is False:
        set_ranking_labels(train_data)
        set_ranking_labels(eval_data)

    model = RsUserCF(task=task, data_info=data_info, k_sim=20)
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
        loaded_model, loaded_data_info = save_load_model(RsUserCF, model, data_info)
        ptest_preds(loaded_model, task, pd_data, with_feats=False)
        ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=False)
        model.save("not_existed_path", "user_cf2")
        remove_path("not_existed_path")


@pytest.mark.skipif(
    sys.version_info[:2] < (3, 7),
    reason="Rust implementation only supports Python >= 3.7.",
)
def test_all_consumed_recommend(pure_data_small, monkeypatch):
    _, train_data, eval_data, data_info = pure_data_small
    set_ranking_labels(train_data)
    set_ranking_labels(eval_data)

    model = RsUserCF(task="ranking", data_info=data_info)
    model.fit(train_data, neg_sampling=False, verbose=0)
    model.save("not_existed_path", "user_cf2")
    remove_path("not_existed_path")
    with monkeypatch.context() as m:
        m.setitem(model.user_consumed, 0, list(range(model.n_items)))
        recos = model.recommend_user(user=1, n_rec=7)
        assert np.all(np.isin(recos[1], data_info.popular_items))


@pytest.mark.skipif(
    sys.version_info[:2] < (3, 7),
    reason="Rust implementation only supports Python >= 3.7.",
)
def test_retrain():
    data_path = Path(__file__).parents[1] / "sample_data" / "sample_movielens_rating.dat"  # fmt: skip
    all_data = pd.read_csv(
        data_path, sep="::", names=["user", "item", "label", "time"], engine="python"
    )
    # use first half data as first training part
    first_half_data = all_data[: (len(all_data) // 2)]
    train_data, eval_data = split_by_ratio_chrono(first_half_data, test_size=0.2)
    train_data, data_info = DatasetPure.build_trainset(train_data)
    eval_data = DatasetPure.build_evalset(eval_data)

    model = RsUserCF(
        task="ranking",
        data_info=data_info,
        k_sim=20,
        mode="invert",
        num_threads=1,
        min_common=1,
        seed=42,
    )
    model.fit(
        train_data,
        neg_sampling=True,
        verbose=2,
        eval_data=eval_data,
        metrics=[
            "loss",
            "balanced_accuracy",
            "roc_auc",
            "pr_auc",
            "precision",
            "recall",
            "map",
            "ndcg",
        ],
    )
    eval_result = evaluate(
        model,
        eval_data,
        neg_sampling=True,
        eval_batch_size=8192,
        k=10,
        metrics=["roc_auc", "pr_auc", "precision", "recall", "map", "ndcg"],
        seed=2222,
    )

    data_info.save(path=SAVE_PATH, model_name="usercf_model")
    model.save(
        path=SAVE_PATH, model_name="usercf_model", manual=True, inference_only=False
    )

    # ========================== load and retrain =============================
    new_data_info = DataInfo.load(SAVE_PATH, model_name="usercf_model")

    # use first half of second half data as second training part
    second_half_data = all_data[(len(all_data) // 2) : (len(all_data) * 3 // 4)]
    train_data_orig, eval_data_orig = split_by_ratio_chrono(
        second_half_data, test_size=0.2
    )
    train_data, new_data_info = DatasetPure.merge_trainset(
        train_data_orig, new_data_info, merge_behavior=True
    )
    eval_data = DatasetPure.merge_evalset(eval_data_orig, new_data_info)

    new_model = RsUserCF(
        task="ranking",
        data_info=new_data_info,
        k_sim=20,
        mode="invert",
        num_threads=1,
        min_common=1,
        seed=42,
    )
    new_model.rebuild_model(path=SAVE_PATH, model_name="usercf_model")
    new_model.fit(
        train_data,
        neg_sampling=True,
        verbose=2,
        eval_data=eval_data,
        metrics=[
            "loss",
            "balanced_accuracy",
            "roc_auc",
            "pr_auc",
            "precision",
            "recall",
            "map",
            "ndcg",
        ],
    )
    ptest_preds(new_model, "ranking", second_half_data, with_feats=False)
    ptest_recommends(new_model, new_data_info, second_half_data, with_feats=False)

    new_eval_result = evaluate(
        new_model,
        eval_data_orig,
        neg_sampling=True,
        eval_batch_size=100000,
        k=2,
        metrics=["roc_auc", "pr_auc", "precision", "recall", "map", "ndcg"],
        seed=2222,
    )

    assert new_eval_result["roc_auc"] != eval_result["roc_auc"]

    new_data_info.save(path=SAVE_PATH, model_name="usercf_model")
    new_model.save(
        path=SAVE_PATH, model_name="usercf_model", manual=True, inference_only=False
    )

    # ========================== load and retrain 2 =============================
    new_data_info = DataInfo.load(SAVE_PATH, model_name="usercf_model")

    # use second half of second half data as second training part
    third_half_data = all_data[(len(all_data) * 3 // 4) :]
    train_data_orig, eval_data_orig = split_by_ratio_chrono(
        third_half_data, test_size=0.2
    )
    train_data, new_data_info = DatasetPure.merge_trainset(
        train_data_orig, new_data_info, merge_behavior=True
    )
    eval_data = DatasetPure.merge_evalset(eval_data_orig, new_data_info)

    new_model = RsUserCF(
        task="ranking",
        data_info=new_data_info,
        k_sim=20,
        mode="invert",
        num_threads=1,
        min_common=1,
        seed=42,
    )
    new_model.rebuild_model(path=SAVE_PATH, model_name="usercf_model")
    new_model.fit(
        train_data,
        neg_sampling=True,
        verbose=2,
        eval_data=eval_data,
        metrics=[
            "loss",
            "balanced_accuracy",
            "roc_auc",
            "pr_auc",
            "precision",
            "recall",
            "map",
            "ndcg",
        ],
    )
    ptest_preds(new_model, "ranking", third_half_data, with_feats=False)
    ptest_recommends(new_model, new_data_info, third_half_data, with_feats=False)

    remove_path(SAVE_PATH)
