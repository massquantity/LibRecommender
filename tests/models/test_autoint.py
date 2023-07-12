import sys

import pytest
import tensorflow as tf

from libreco.algorithms import AutoInt
from tests.models.utils_tf import ptest_tf_variables
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
    "lr_decay, reg, use_residual, att_embed_size, num_heads, num_workers",
    [
        (False, None, True, None, 1, 0),
        (True, 0.001, False, 16, 2, 1),
        (True, None, False, (4, 8), 2, 0),
    ],
)
def test_autoint(
    feat_data_small,
    task,
    loss_type,
    lr_decay,
    reg,
    use_residual,
    att_embed_size,
    num_heads,
    num_workers,
):
    if not sys.platform.startswith("linux") and num_workers > 0:
        pytest.skip(
            "Windows and macOS use `spawn` in multiprocessing, which does not work well in pytest"
        )
    tf.compat.v1.reset_default_graph()
    pd_data, train_data, eval_data, data_info = feat_data_small

    neg_sampling = True if task == "ranking" else False
    if task == "ranking" and loss_type not in ("cross_entropy", "focal"):
        with pytest.raises(ValueError):
            AutoInt(task, data_info, loss_type).fit(train_data, neg_sampling)
    else:
        model = AutoInt(
            task=task,
            data_info=data_info,
            loss_type=loss_type,
            embed_size=4,
            n_epochs=1,
            lr=1e-4,
            lr_decay=lr_decay,
            reg=reg,
            batch_size=100,
            use_bn=True,
            dropout_rate=None,
            use_residual=use_residual,
            att_embed_size=att_embed_size,
            num_heads=num_heads,
            tf_sess_config=None,
        )
        model.fit(
            train_data,
            neg_sampling=neg_sampling,
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


def test_autoint_multi_sparse(multi_sparse_data_small):
    task = "ranking"
    pd_data, train_data, eval_data, data_info = multi_sparse_data_small
    model = fit_multi_sparse(AutoInt, train_data, eval_data, data_info)
    ptest_preds(model, task, pd_data, with_feats=True)
    ptest_recommends(model, data_info, pd_data, with_feats=True)

    # test save and load model
    loaded_model, loaded_data_info = save_load_model(AutoInt, model, data_info)
    ptest_preds(loaded_model, task, pd_data, with_feats=True)
    ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=True)
    with pytest.raises(RuntimeError):
        loaded_model.fit(train_data, neg_sampling=True)
