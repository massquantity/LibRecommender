from pathlib import Path

import pandas as pd

from libreco.algorithms import DeepWalk, Item2Vec
from libreco.data import DataInfo, DatasetPure, split_by_ratio_chrono
from libreco.evaluation import evaluate
from tests.utils_data import SAVE_PATH, remove_path
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_recommends


def test_gensim_model_retrain_pure():
    data_path = (
        Path(__file__).parents[1].joinpath("sample_data", "sample_movielens_rating.dat")
    )
    all_data = pd.read_csv(
        data_path, sep="::", names=["user", "item", "label", "time"], engine="python"
    )
    # use first half data as first training part
    first_half_data = all_data[: (len(all_data) // 2)]
    train_data, eval_data = split_by_ratio_chrono(first_half_data, test_size=0.2)
    train_data, data_info = DatasetPure.build_trainset(train_data)
    eval_data = DatasetPure.build_evalset(eval_data)
    retrain("item2vec", train_data, eval_data, data_info, all_data)
    retrain("deepwalk", train_data, eval_data, data_info, all_data)


def retrain(model_name, train_data, eval_data, data_info, all_data):
    if model_name == "item2vec":
        model = Item2Vec(
            "ranking",
            data_info,
            embed_size=16,
            norm_embed=False,
            window_size=5,
            n_epochs=2,
            n_threads=0,
        )
    else:
        model = DeepWalk(
            "ranking",
            data_info,
            embed_size=16,
            norm_embed=False,
            n_walks=10,
            walk_length=3,
            window_size=5,
            n_epochs=2,
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

    # use second half data as second training part
    second_half_data = all_data[(len(all_data) // 2) :]
    train_data_orig, eval_data_orig = split_by_ratio_chrono(
        second_half_data, test_size=0.2
    )
    train_data, new_data_info = DatasetPure.merge_trainset(
        train_data_orig, new_data_info, merge_behavior=False
    )
    eval_data = DatasetPure.merge_evalset(eval_data_orig, new_data_info)

    if model_name == "item2vec":
        new_model = Item2Vec(
            "ranking",
            new_data_info,
            embed_size=16,
            norm_embed=False,
            window_size=5,
            n_epochs=2,
            n_threads=0,
        )
    else:
        new_model = DeepWalk(
            "ranking",
            new_data_info,
            embed_size=16,
            norm_embed=False,
            n_walks=10,
            walk_length=3,
            window_size=5,
            n_epochs=2,
        )

    new_model.rebuild_model(path=SAVE_PATH, model_name=model_name)
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
    )
    ptest_preds(new_model, "ranking", second_half_data, with_feats=False)
    ptest_recommends(new_model, new_data_info, second_half_data, with_feats=False)

    new_eval_result = evaluate(
        new_model,
        eval_data_orig,
        neg_sampling=True,
        eval_batch_size=8192,
        k=10,
        metrics=["roc_auc", "pr_auc", "precision", "recall", "map", "ndcg"],
        seed=2222,
    )

    assert new_eval_result["roc_auc"] != eval_result["roc_auc"]

    remove_path(SAVE_PATH)
