import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from libreco.algorithms import Swing
from libreco.data import DataInfo, DatasetPure, split_by_ratio_chrono
from libreco.evaluation import evaluate
from tests.utils_data import SAVE_PATH, remove_path
from tests.utils_reco import ptest_recommends


@pytest.mark.skipif(
    sys.version_info[:2] < (3, 7),
    reason="Rust implementation only supports Python >= 3.7.",
)
def test_rs_cf_retrain():
    data_path = Path(__file__).parents[1] / "sample_data" / "sample_movielens_rating.dat"  # fmt: skip
    all_data = pd.read_csv(
        data_path, sep="::", names=["user", "item", "label", "time"], engine="python"
    )
    # use first half data as first training part
    first_half_data = all_data[: (len(all_data) // 2)]
    train_data, eval_data = split_by_ratio_chrono(first_half_data, test_size=0.2)
    train_data, data_info = DatasetPure.build_trainset(train_data)
    eval_data = DatasetPure.build_evalset(eval_data)
    model_name = "swing_model"

    model = Swing(
        task="ranking",
        data_info=data_info,
        top_k=20,
        alpha=1.0,
        num_threads=2,
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

    new_model = Swing(
        task="ranking",
        data_info=new_data_info,
        top_k=20,
        alpha=1.0,
        num_threads=1,
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
    ptest_preds(new_model, second_half_data)
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

    new_model = Swing(
        task="ranking",
        data_info=new_data_info,
        top_k=20,
        alpha=1.0,
        num_threads=1,
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
    ptest_preds(new_model, third_half_data)
    ptest_recommends(new_model, new_data_info, third_half_data, with_feats=False)

    remove_path(SAVE_PATH)


def ptest_preds(model, pd_data):
    user = pd_data.user.iloc[0]
    item = pd_data.item.iloc[0]
    pred = model.predict(user=user, item=item)
    assert pred >= 0

    popular_pred = model.predict(
        user="cold user2", item="cold item2", cold_start="popular"
    )
    assert np.allclose(popular_pred, model.default_pred)

    cold_pred1 = model.predict(user="cold user1", item="cold item2")
    cold_pred2 = model.predict(user="cold user2", item="cold item2")
    assert cold_pred1 == cold_pred2
