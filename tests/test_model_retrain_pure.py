import os
import shutil
from pathlib import Path

import pandas as pd
import tensorflow as tf

from libreco.algorithms import WaveNet
from libreco.data import DataInfo, DatasetPure, split_by_ratio_chrono
from libreco.evaluation import evaluate
from tests.utils_path import SAVE_PATH
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_recommends


def test_model_retrain():
    tf.compat.v1.reset_default_graph()
    data_path = os.path.join(
        str(Path(os.path.realpath(__file__)).parent),
        "sample_data",
        "sample_movielens_rating.dat",
    )
    all_data = pd.read_csv(data_path, sep="::", names=["user", "item", "label", "time"])
    # use first half data as first training part
    first_half_data = all_data[: (len(all_data) // 2)]
    train_data, eval_data = split_by_ratio_chrono(first_half_data, test_size=0.2)
    train_data, data_info = DatasetPure.build_trainset(train_data, reset_state=True)
    eval_data = DatasetPure.build_evalset(eval_data)
    train_data.build_negative_samples(
        data_info, item_gen_mode="random", num_neg=1, seed=2022
    )
    eval_data.build_negative_samples(
        data_info, item_gen_mode="random", num_neg=1, seed=2222
    )

    model = WaveNet(
        "ranking",
        data_info,
        loss_type="cross_entropy",
        embed_size=16,
        n_epochs=1,
        lr=1e-4,
        lr_decay=None,
        reg=None,
        batch_size=2048,
        n_filters=16,
        n_blocks=2,
        n_layers_per_block=4,
        recent_num=10,
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
    )
    eval_result = evaluate(
        model,
        eval_data,
        eval_batch_size=8192,
        k=10,
        metrics=["roc_auc", "pr_auc", "precision", "recall", "map", "ndcg"],
        neg_sample=True,
        update_features=False,
        seed=2222,
    )

    data_info.save(path=SAVE_PATH, model_name="wavenet_model")
    model.save(
        path=SAVE_PATH, model_name="wavenet_model", manual=True, inference_only=False
    )

    # ========================== load and retrain =============================
    tf.compat.v1.reset_default_graph()
    new_data_info = DataInfo.load(SAVE_PATH, model_name="wavenet_model")

    # use second half data as second training part
    second_half_data = all_data[(len(all_data) // 2) :]
    train_data_orig, eval_data_orig = split_by_ratio_chrono(
        second_half_data, test_size=0.2
    )
    train_data, new_data_info = DatasetPure.build_trainset(
        train_data_orig, revolution=True, data_info=new_data_info, merge_behavior=True
    )
    eval_data = DatasetPure.build_evalset(
        eval_data_orig, revolution=True, data_info=new_data_info
    )
    train_data.build_negative_samples(
        new_data_info, item_gen_mode="random", num_neg=1, seed=2022
    )
    eval_data.build_negative_samples(
        new_data_info, item_gen_mode="random", num_neg=1, seed=2222
    )

    new_model = WaveNet(
        "ranking",
        new_data_info,
        loss_type="focal",  # change loss
        embed_size=16,
        n_epochs=1,
        lr=1e-4,
        lr_decay=None,
        reg=None,
        batch_size=2048,
        n_filters=16,
        n_blocks=2,
        n_layers_per_block=4,
        recent_num=10,
    )
    new_model.rebuild_model(
        path=SAVE_PATH, model_name="wavenet_model", full_assign=True
    )
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
    )
    ptest_preds(new_model, "ranking", second_half_data, with_feats=False)
    ptest_recommends(new_model, new_data_info, second_half_data, with_feats=False)

    new_eval_result = evaluate(
        new_model,
        eval_data_orig,
        eval_batch_size=8192,
        k=10,
        metrics=["roc_auc", "pr_auc", "precision", "recall", "map", "ndcg"],
        neg_sample=True,
        update_features=False,
        seed=2222,
    )

    assert new_eval_result["roc_auc"] != eval_result["roc_auc"]

    if os.path.exists(SAVE_PATH) and os.path.isdir(SAVE_PATH):
        shutil.rmtree(SAVE_PATH)
