import os
from pathlib import Path

import pandas as pd
import pytest
import tensorflow as tf

from libreco.algorithms import DIN
from libreco.data import DatasetFeat, split_by_ratio_chrono
from tests.utils_metrics import get_metrics
from tests.utils_multi_sparse_models import fit_multi_sparse
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_recommends
from tests.utils_save_load import save_load_model


@pytest.mark.parametrize(
    "task, loss_type",
    [
        ("rating", "whatever"),
        ("ranking", "cross_entropy"),
        ("ranking", "focal"),
        ("ranking", "unknown"),
    ],
)
@pytest.mark.parametrize(
    "lr_decay, reg, num_neg, use_bn, dropout_rate, hidden_units, "
    "recent_num, use_tf_attention",
    [
        (False, None, 1, False, None, "128,64,32", 10, False),
        (True, 0.001, 3, True, 0.5, "1,1,1", 6, True),
    ],
)
def test_din(
    prepare_feat_data,
    task,
    loss_type,
    lr_decay,
    reg,
    num_neg,
    use_bn,
    dropout_rate,
    hidden_units,
    recent_num,
    use_tf_attention,
):
    tf.compat.v1.reset_default_graph()
    pd_data, train_data, eval_data, data_info = prepare_feat_data
    if task == "ranking":
        train_data.build_negative_samples(data_info, seed=2022)
        eval_data.build_negative_samples(data_info, seed=2222)

    if task == "ranking" and loss_type not in ("cross_entropy", "focal"):
        with pytest.raises(ValueError):
            _ = DIN(task, data_info, loss_type)
    else:
        model = DIN(
            task=task,
            data_info=data_info,
            loss_type=loss_type,
            embed_size=4,
            n_epochs=1,
            lr=1e-4,
            lr_decay=lr_decay,
            reg=reg,
            batch_size=8192,
            num_neg=num_neg,
            use_bn=use_bn,
            dropout_rate=dropout_rate,
            hidden_units=hidden_units,
            recent_num=recent_num,
            use_tf_attention=use_tf_attention,
            tf_sess_config=None,
        )
        model.fit(
            train_data,
            verbose=2,
            shuffle=True,
            eval_data=eval_data,
            metrics=get_metrics(task),
            eval_user_num=40,
        )
        ptest_preds(model, task, pd_data, with_feats=True)
        ptest_recommends(model, data_info, pd_data, with_feats=True)


def test_din_multi_sparse(prepare_multi_sparse_data):
    task = "ranking"
    pd_data, train_data, eval_data, data_info = prepare_multi_sparse_data
    model = fit_multi_sparse(DIN, train_data, eval_data, data_info)
    ptest_preds(model, task, pd_data, with_feats=True)
    ptest_recommends(model, data_info, pd_data, with_feats=True)

    # test save and load model
    loaded_model, loaded_data_info = save_load_model(DIN, model, data_info)
    ptest_preds(loaded_model, task, pd_data, with_feats=True)
    ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=True)


def test_item_dense_feature():
    tf.compat.v1.reset_default_graph()
    data_path = os.path.join(
        str(Path(os.path.realpath(__file__)).parent.parent),
        "sample_data",
        "sample_movielens_merged.csv",
    )
    pd_data = pd.read_csv(data_path, sep=",", header=0)
    train_data, eval_data = split_by_ratio_chrono(pd_data, test_size=0.2)
    train_data, data_info = DatasetFeat.build_trainset(
        train_data=train_data,
        sparse_col=["sex", "occupation", "genre1", "genre2", "genre3"],
        dense_col=["age"],
        user_col=["sex", "occupation"],
        item_col=["genre1", "genre2", "genre3", "age"],  # assign `age` to item feature
    )
    eval_data = DatasetFeat.build_testset(eval_data)
    model = DIN(
        "rating",
        data_info,
        embed_size=4,
        lr=3e-4,
        n_epochs=1,
        batch_size=8192,
    )
    model.fit(
        train_data,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=get_metrics("rating"),
        eval_user_num=40,
    )
    ptest_preds(model, "rating", pd_data, with_feats=True)
    ptest_recommends(model, data_info, pd_data, with_feats=True)
