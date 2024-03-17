import sys
from pathlib import Path

import pandas as pd
import pytest

from libreco.algorithms import RsItemCF, RsUserCF
from libreco.data import DataInfo, DatasetPure, split_by_ratio_chrono
from libreco.evaluation import evaluate
from tests.utils_data import SAVE_PATH, remove_path
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_recommends


@pytest.fixture
def cf_model(request):
    if request.param == "user_cf":
        return RsUserCF, "usercf_model"
    else:
        return RsItemCF, "itemcf_model"


@pytest.mark.skipif(
    sys.version_info[:2] < (3, 7),
    reason="Rust implementation only supports Python >= 3.7.",
)
@pytest.mark.parametrize(
    "cf_model",
    ["user_cf", "item_cf"],
    indirect=True,
)
def test_rs_cf_retrain(cf_model):
    data_path = Path(__file__).parents[1] / "sample_data" / "sample_movielens_rating.dat"  # fmt: skip
    all_data = pd.read_csv(
        data_path, sep="::", names=["user", "item", "label", "time"], engine="python"
    )
    # use first half data as first training part
    first_half_data = all_data[: (len(all_data) // 2)]
    train_data, eval_data = split_by_ratio_chrono(first_half_data, test_size=0.2)
    train_data, data_info = DatasetPure.build_trainset(train_data)
    eval_data = DatasetPure.build_evalset(eval_data)
    rs_model_cls, model_name = cf_model

    model = rs_model_cls(
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

    data_info.save(path=SAVE_PATH, model_name=model_name)
    model.save(path=SAVE_PATH, model_name=model_name, manual=True, inference_only=False)

    # ========================== load and retrain =============================
    new_data_info = DataInfo.load(SAVE_PATH, model_name=model_name)

    # use first half of second half data as second training part
    second_half_data = all_data[(len(all_data) // 2) : (len(all_data) * 3 // 4)]
    train_data_orig, eval_data_orig = split_by_ratio_chrono(
        second_half_data, test_size=0.2
    )
    train_data, new_data_info = DatasetPure.merge_trainset(
        train_data_orig, new_data_info, merge_behavior=True
    )
    eval_data = DatasetPure.merge_evalset(eval_data_orig, new_data_info)

    new_model = rs_model_cls(
        task="ranking",
        data_info=new_data_info,
        k_sim=20,
        mode="invert",
        num_threads=1,
        min_common=1,
        seed=42,
    )
    new_model.rebuild_model(path=SAVE_PATH, model_name=model_name)
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

    new_data_info.save(path=SAVE_PATH, model_name=model_name)
    new_model.save(
        path=SAVE_PATH, model_name=model_name, manual=True, inference_only=False
    )

    # ========================== load and retrain 2 =============================
    new_data_info = DataInfo.load(SAVE_PATH, model_name=model_name)

    # use second half of second half data as second training part
    third_half_data = all_data[(len(all_data) * 3 // 4) :]
    train_data_orig, eval_data_orig = split_by_ratio_chrono(
        third_half_data, test_size=0.2
    )
    train_data, new_data_info = DatasetPure.merge_trainset(
        train_data_orig, new_data_info, merge_behavior=True
    )
    eval_data = DatasetPure.merge_evalset(eval_data_orig, new_data_info)

    new_model = rs_model_cls(
        task="ranking",
        data_info=new_data_info,
        k_sim=20,
        mode="invert",
        num_threads=1,
        min_common=1,
        seed=42,
    )
    new_model.rebuild_model(path=SAVE_PATH, model_name=model_name)
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
