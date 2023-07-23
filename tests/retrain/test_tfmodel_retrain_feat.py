from pathlib import Path

import pandas as pd
import pytest
import tensorflow as tf

from libreco.algorithms import DIN
from libreco.data import DataInfo, DatasetFeat, split_by_ratio_chrono
from libreco.evaluation import evaluate
from tests.utils_data import SAVE_PATH, remove_path
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_recommends


def test_tfmodel_retrain_feat():
    tf.compat.v1.reset_default_graph()
    data_path = Path(__file__).parents[1] / "sample_data" / "sample_movielens_merged.csv"  # fmt: skip
    all_data = pd.read_csv(data_path, sep=",", header=0)
    # use first half data as first training part
    first_half_data = all_data[: (len(all_data) // 2)]
    train_data, eval_data = split_by_ratio_chrono(first_half_data, test_size=0.2)

    sparse_col = ["sex", "occupation", "genre1", "genre2", "genre3"]
    dense_col = ["age"]
    user_col = ["sex", "age", "occupation"]
    item_col = ["genre1", "genre2", "genre3"]
    train_data, data_info = DatasetFeat.build_trainset(
        train_data,
        user_col,
        item_col,
        sparse_col,
        dense_col,
        shuffle=False,
    )
    eval_data = DatasetFeat.build_evalset(eval_data)

    model = DIN(
        "ranking",
        data_info,
        loss_type="cross_entropy",
        embed_size=16,
        n_epochs=1,
        lr=1e-4,
        lr_decay=False,
        reg=None,
        batch_size=2048,
        hidden_units=(32, 16),
        recent_num=10,
        use_tf_attention=True,
    )
    model.fit(
        train_data,
        neg_sampling=True,
        verbose=2,
        shuffle=True,
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
        eval_user_num=20,
    )
    eval_result = evaluate(
        model,
        eval_data,
        neg_sampling=True,
        sample_user_num=200,
        eval_batch_size=8192,
        k=10,
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
        seed=2222,
    )

    data_info.save(path=SAVE_PATH, model_name="din_model")
    model.save(
        path=SAVE_PATH, model_name="din_model", manual=True, inference_only=False
    )

    # ========================== load and retrain =============================
    tf.compat.v1.reset_default_graph()
    new_data_info = DataInfo.load(SAVE_PATH, model_name="din_model")

    # use first half of second half data as second training part
    second_half_data = all_data[(len(all_data) // 2) : (len(all_data) * 3 // 4)]
    train_data_orig, eval_data_orig = split_by_ratio_chrono(
        second_half_data, test_size=0.2
    )
    train_data, new_data_info = DatasetFeat.merge_trainset(
        train_data_orig, new_data_info, merge_behavior=True
    )
    eval_data = DatasetFeat.merge_evalset(eval_data_orig, new_data_info)

    new_model = DIN(
        "ranking",
        new_data_info,
        loss_type="focal",  # change loss
        embed_size=16,
        n_epochs=1,
        lr=1e-4,
        lr_decay=False,
        reg=None,
        batch_size=2048,
        hidden_units=(32, 16),
        recent_num=10,
        use_tf_attention=True,
    )
    new_model.rebuild_model(path=SAVE_PATH, model_name="din_model", full_assign=True)
    new_model.fit(
        train_data,
        neg_sampling=True,
        verbose=2,
        shuffle=True,
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
        eval_user_num=20,
    )
    ptest_preds(new_model, "ranking", second_half_data, with_feats=False)
    ptest_recommends(new_model, new_data_info, second_half_data, with_feats=False)

    with pytest.raises(ValueError, match="`data` must be `pandas.DataFrame` or `TransformedEvalSet`"):  # fmt: skip
        _ = evaluate(new_model, eval_data_orig["user"], neg_sampling=True)

    new_eval_result = evaluate(
        new_model,
        eval_data_orig,
        neg_sampling=True,
        sample_user_num=200,
        eval_batch_size=100000,
        k=20,
        metrics=["roc_auc", "pr_auc", "precision", "recall", "map", "ndcg"],
        seed=2222,
    )
    _ = evaluate(new_model, eval_data, neg_sampling=False)

    assert new_eval_result["roc_auc"] != eval_result["roc_auc"]

    new_data_info.save(path=SAVE_PATH, model_name="din_model")
    new_model.save(
        path=SAVE_PATH, model_name="din_model", manual=True, inference_only=False
    )

    # ========================== load and retrain 2 =============================
    tf.compat.v1.reset_default_graph()
    new_data_info = DataInfo.load(SAVE_PATH, model_name="din_model")

    # use second half of second half data as second training part
    third_half_data = all_data[(len(all_data) * 3 // 4) :]
    train_data_orig, eval_data_orig = split_by_ratio_chrono(
        third_half_data, test_size=0.2
    )
    train_data, new_data_info = DatasetFeat.merge_trainset(
        train_data_orig, new_data_info, merge_behavior=True
    )
    eval_data = DatasetFeat.merge_evalset(eval_data_orig, new_data_info)
    print(new_data_info)

    new_model = DIN(
        "ranking",
        new_data_info,
        loss_type="focal",  # change loss
        embed_size=16,
        n_epochs=1,
        lr=1e-4,
        lr_decay=False,
        reg=None,
        batch_size=2048,
        hidden_units=(32, 16),
        recent_num=10,
        use_tf_attention=True,
    )
    new_model.rebuild_model(path=SAVE_PATH, model_name="din_model", full_assign=True)
    new_model.fit(
        train_data,
        neg_sampling=True,
        verbose=2,
        shuffle=True,
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
        eval_user_num=20,
    )
    ptest_preds(new_model, "ranking", third_half_data, with_feats=False)
    ptest_recommends(new_model, new_data_info, third_half_data, with_feats=False)

    remove_path(SAVE_PATH)
