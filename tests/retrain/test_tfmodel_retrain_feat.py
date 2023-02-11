import os
from pathlib import Path

import pandas as pd
import tensorflow as tf

from libreco.algorithms import DIN
from libreco.data import DataInfo, DatasetFeat, split_by_ratio_chrono
from libreco.evaluation import evaluate
from tests.utils_path import SAVE_PATH, remove_path
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_recommends


def test_tfmodel_retrain_feat():
    tf.compat.v1.reset_default_graph()
    data_path = os.path.join(
        str(Path(os.path.realpath(__file__)).parent.parent),
        "sample_data",
        "sample_movielens_merged.csv",
    )
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
    train_data.build_negative_samples(data_info, seed=2022)
    eval_data.build_negative_samples(data_info, seed=2222)

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
        neg_sample=True,
        update_features=False,
        seed=2222,
    )

    data_info.save(path=SAVE_PATH, model_name="din_model")
    model.save(
        path=SAVE_PATH, model_name="din_model", manual=True, inference_only=False
    )

    # ========================== load and retrain =============================
    tf.compat.v1.reset_default_graph()
    new_data_info = DataInfo.load(SAVE_PATH, model_name="din_model")

    # use second half data as second training part
    second_half_data = all_data[(len(all_data) // 2) :]
    train_data_orig, eval_data_orig = split_by_ratio_chrono(
        second_half_data, test_size=0.2
    )
    train_data, new_data_info = DatasetFeat.build_trainset(
        train_data_orig, revolution=True, data_info=new_data_info, merge_behavior=True
    )
    eval_data = DatasetFeat.build_evalset(
        eval_data_orig, revolution=True, data_info=new_data_info
    )
    train_data.build_negative_samples(new_data_info, seed=2022)
    eval_data.build_negative_samples(new_data_info, seed=2222)

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

    new_eval_result = evaluate(
        new_model,
        eval_data_orig,
        sample_user_num=200,
        eval_batch_size=100000,
        k=20,
        metrics=["roc_auc", "pr_auc", "precision", "recall", "map", "ndcg"],
        neg_sample=True,
        update_features=True,
        seed=2222,
    )

    assert new_eval_result["roc_auc"] != eval_result["roc_auc"]

    remove_path(SAVE_PATH)
