import os
import sys
from pathlib import Path

import pandas as pd
import pytest
import tensorflow as tf
from numpy.testing import assert_array_equal

from libreco.algorithms import DIN
from libreco.data import DatasetFeat, split_by_ratio_chrono
from tests.models.utils_tf import ptest_tf_variables
from tests.utils_data import set_ranking_labels
from tests.utils_metrics import get_metrics
from tests.utils_multi_sparse_models import fit_multi_sparse
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_dyn_recommends, ptest_recommends
from tests.utils_save_load import save_load_model


@pytest.mark.parametrize(
    "task, loss_type, sampler, neg_sampling",
    [
        ("rating", "focal", "random", None),
        ("rating", "focal", None, True),
        ("rating", "focal", "random", True),
        ("ranking", "cross_entropy", "random", False),
        ("ranking", "focal", "unconsumed", False),
        ("ranking", "cross_entropy", "random", True),
        ("ranking", "cross_entropy", "unconsumed", True),
        ("ranking", "focal", "popular", True),
        ("ranking", "unknown", "popular", True),
    ],
)
@pytest.mark.parametrize(
    "lr_decay, reg, num_neg, use_bn, dropout_rate, hidden_units, "
    "recent_num, use_tf_attention, num_workers",
    [
        (False, None, 1, False, None, (64, 32), 10, False, 0),
        (True, 0.001, 3, True, 0.5, 1, 6, True, 2),
    ],
)
def test_din(
    feat_data_small,
    task,
    loss_type,
    sampler,
    neg_sampling,
    lr_decay,
    reg,
    num_neg,
    use_bn,
    dropout_rate,
    hidden_units,
    recent_num,
    use_tf_attention,
    num_workers,
):
    if not sys.platform.startswith("linux") and num_workers > 0:
        pytest.skip(
            "Windows and macOS use `spawn` in multiprocessing, which does not work well in pytest"
        )
    tf.compat.v1.reset_default_graph()
    pd_data, train_data, eval_data, data_info = feat_data_small
    if task == "ranking" and neg_sampling is False and loss_type == "cross_entropy":
        set_ranking_labels(train_data)
        set_ranking_labels(eval_data)

    if neg_sampling is None:
        with pytest.raises(AssertionError):
            DIN(task, data_info).fit(train_data, neg_sampling)
    elif task == "rating" and neg_sampling:
        with pytest.raises(ValueError):
            DIN(task, data_info).fit(train_data, neg_sampling)
    elif loss_type == "focal" and (neg_sampling is False or sampler is None):
        with pytest.raises(ValueError):
            DIN(task, data_info, sampler=sampler).fit(train_data, neg_sampling)
    elif task == "ranking" and loss_type not in ("cross_entropy", "focal"):
        with pytest.raises(ValueError):
            DIN(task, data_info, loss_type).fit(train_data, neg_sampling)
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
            batch_size=100,
            sampler=sampler,
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
            neg_sampling,
            verbose=2,
            shuffle=True,
            eval_data=eval_data,
            metrics=get_metrics(task),
            eval_user_num=40,
            num_workers=num_workers,
        )
        ptest_tf_variables(model)
        ptest_preds(model, task, pd_data, with_feats=True)
        ptest_recommends(model, data_info, pd_data, with_feats=True)
        ptest_dyn_recommends(model, pd_data)


def test_din_multi_sparse(multi_sparse_data_small):
    task = "ranking"
    pd_data, train_data, eval_data, data_info = multi_sparse_data_small
    model = fit_multi_sparse(DIN, train_data, eval_data, data_info)
    ptest_preds(model, task, pd_data, with_feats=True)
    ptest_recommends(model, data_info, pd_data, with_feats=True)
    dyn_rec = ptest_dyn_recommends(model, pd_data)

    # test save and load model
    loaded_model, loaded_data_info = save_load_model(DIN, model, data_info)
    ptest_preds(loaded_model, task, pd_data, with_feats=True)
    ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=True)
    loaded_dyn_rec = ptest_dyn_recommends(loaded_model, pd_data)
    assert_array_equal(dyn_rec, loaded_dyn_rec)
    with pytest.raises(RuntimeError):
        loaded_model.fit(train_data, neg_sampling=True)


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
        neg_sampling=False,
        verbose=2,
        shuffle=True,
        eval_data=eval_data,
        metrics=get_metrics("rating"),
        eval_user_num=40,
    )
    ptest_preds(model, "rating", pd_data, with_feats=True)
    ptest_recommends(model, data_info, pd_data, with_feats=True)
    ptest_dyn_recommends(model, pd_data)
