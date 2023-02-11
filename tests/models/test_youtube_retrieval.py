import os
from pathlib import Path

import pandas as pd
import pytest
import tensorflow as tf

from libreco.algorithms import YouTubeRetrieval
from libreco.data import DatasetFeat, split_by_ratio_chrono
from tests.utils_metrics import get_metrics
from tests.utils_path import SAVE_PATH, remove_path
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_recommends
from tests.utils_save_load import save_load_model


# According to the paper, the YouTuBeRetrieval model can not use item features.
def prepare_youtube_retrieval_data(multi_sparse=False):
    data_path = os.path.join(
        str(Path(os.path.realpath(__file__)).parent.parent),
        "sample_data",
        "sample_movielens_merged.csv",
    )
    pd_data = pd.read_csv(data_path, sep=",", header=0)
    train_data, eval_data = split_by_ratio_chrono(pd_data, test_size=0.2)
    if multi_sparse:
        train_data, data_info = DatasetFeat.build_trainset(
            train_data=train_data,
            sparse_col=["sex", "occupation"],
            multi_sparse_col=[["genre1", "genre2", "genre3"]],
            dense_col=["age"],
            user_col=["sex", "age", "occupation", "genre1", "genre2", "genre3"],
            item_col=[],
        )
    else:
        train_data, data_info = DatasetFeat.build_trainset(
            train_data=train_data,
            sparse_col=["sex", "occupation"],
            dense_col=["age"],
            user_col=["sex", "age", "occupation"],
            item_col=[],
        )
    eval_data = DatasetFeat.build_testset(eval_data)
    return pd_data, train_data, eval_data, data_info


@pytest.mark.parametrize("task", ["rating", "ranking"])
@pytest.mark.parametrize(
    "lr_decay, reg, use_bn, dropout_rate, recent_num, random_num, hidden_units",
    [
        (False, None, False, None, 10, None, 1),
        (True, 0.001, True, 0.5, None, 10, [16, 16]),
        (True, 0.001, False, None, None, None, (4, 4, 4)),
        (False, None, False, None, 10, None, "64,64"),
        (True, 0.001, True, 0.5, None, 10, [1, 2, 4.22]),
    ],
)
@pytest.mark.parametrize("num_sampled_per_batch", [None, 1, 64])
@pytest.mark.parametrize("loss_type", ["nce", "sampled_softmax", "unknown"])
def test_youtube_retrieval(
    task,
    lr_decay,
    reg,
    use_bn,
    dropout_rate,
    num_sampled_per_batch,
    loss_type,
    recent_num,
    random_num,
    hidden_units,
):
    tf.compat.v1.reset_default_graph()
    pd_data, train_data, eval_data, data_info = prepare_youtube_retrieval_data()
    if task == "ranking":
        train_data.build_negative_samples(data_info, seed=2022)
        eval_data.build_negative_samples(data_info, seed=2222)

    if task == "rating":
        with pytest.raises(AssertionError):
            _ = YouTubeRetrieval(task, data_info, loss_type)
    elif task == "ranking" and loss_type not in ("nce", "sampled_softmax"):
        with pytest.raises(ValueError):
            YouTubeRetrieval(task, data_info, loss_type).fit(train_data)
    elif hidden_units in ("64,64", [1, 2, 4.22]):
        with pytest.raises(ValueError):
            YouTubeRetrieval(task, data_info, hidden_units=hidden_units)
    else:
        model = YouTubeRetrieval(
            task=task,
            data_info=data_info,
            loss_type=loss_type,
            embed_size=4,
            n_epochs=1,
            lr=1e-4,
            lr_decay=lr_decay,
            reg=reg,
            batch_size=2048,
            use_bn=use_bn,
            dropout_rate=dropout_rate,
            hidden_units=hidden_units,
            num_sampled_per_batch=num_sampled_per_batch,
            recent_num=recent_num,
            random_num=random_num,
            tf_sess_config=None,
        )
        model.fit(
            train_data,
            verbose=2,
            shuffle=True,
            eval_data=eval_data,
            metrics=get_metrics(task),
            eval_user_num=200,
        )
        ptest_preds(model, task, pd_data, with_feats=False)
        ptest_recommends(model, data_info, pd_data, with_feats=False)

        # test save and load model
        loaded_model, loaded_data_info = save_load_model(
            YouTubeRetrieval, model, data_info
        )
        ptest_preds(loaded_model, task, pd_data, with_feats=False)
        ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=False)

        remove_path(SAVE_PATH)


def test_youtube_retrieval_multi_sparse():
    tf.compat.v1.reset_default_graph()
    task = "ranking"
    pd_data, train_data, eval_data, data_info = prepare_youtube_retrieval_data(
        multi_sparse=True
    )
    train_data.build_negative_samples(
        data_info, item_gen_mode="random", num_neg=1, seed=2022
    )
    eval_data.build_negative_samples(
        data_info, item_gen_mode="random", num_neg=1, seed=2222
    )
    model = YouTubeRetrieval(
        task=task,
        data_info=data_info,
        loss_type="sampled_softmax",
        embed_size=16,
        n_epochs=1,
        lr=1e-4,
        lr_decay=True,
        reg=None,
        batch_size=2048,
        use_bn=True,
        dropout_rate=None,
        num_sampled_per_batch=None,
        tf_sess_config=None,
    )
    model.fit(
        train_data,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=get_metrics(task),
        eval_user_num=200,
    )
    ptest_preds(model, task, pd_data, with_feats=False)
    ptest_recommends(model, data_info, pd_data, with_feats=False)

    # test save and load model
    loaded_model, loaded_data_info = save_load_model(YouTubeRetrieval, model, data_info)
    ptest_preds(loaded_model, task, pd_data, with_feats=False)
    ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=False)

    remove_path(SAVE_PATH)
