import sys

import pytest
import tensorflow as tf

from libreco.algorithms import SVD
from tests.models.utils_tf import ptest_tf_variables
from tests.utils_data import set_ranking_labels
from tests.utils_metrics import get_metrics
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_recommends
from tests.utils_save_load import save_load_model


@pytest.mark.parametrize(
    "task, loss_type, sampler, neg_sampling",
    [
        ("rating", "focal", "random", None),
        ("rating", "focal", None, True),
        ("rating", "focal", "random", True),
        ("ranking", "cross_entropy", "random", False),
        ("ranking", "focal", "random", False),
        ("ranking", "cross_entropy", "random", True),
        ("ranking", "cross_entropy", "unconsumed", True),
        ("ranking", "focal", "popular", True),
        ("ranking", "unknown", "popular", True),
    ],
)
@pytest.mark.parametrize(
    "norm_embed, reg, num_neg, num_workers", [(True, None, 1, 0), (False, 0.001, 3, 2)]
)
def test_svd(
    pure_data_small,
    task,
    loss_type,
    sampler,
    neg_sampling,
    norm_embed,
    reg,
    num_neg,
    num_workers,
):
    if not sys.platform.startswith("linux") and num_workers > 0:
        pytest.skip(
            "Windows and macOS use `spawn` in multiprocessing, which does not work well in pytest"
        )
    tf.compat.v1.reset_default_graph()
    pd_data, train_data, eval_data, data_info = pure_data_small
    if task == "ranking" and neg_sampling is False and loss_type == "cross_entropy":
        set_ranking_labels(train_data)
        set_ranking_labels(eval_data)

    if neg_sampling is None:
        with pytest.raises(AssertionError):
            SVD(task, data_info).fit(train_data, neg_sampling)
    elif task == "rating" and neg_sampling:
        with pytest.raises(ValueError):
            SVD(task, data_info).fit(train_data, neg_sampling)
    elif loss_type == "focal" and (neg_sampling is False or sampler is None):
        with pytest.raises(ValueError):
            SVD(task, data_info, sampler=sampler).fit(train_data, neg_sampling)
    elif task == "ranking" and loss_type not in ("cross_entropy", "focal"):
        with pytest.raises(ValueError):
            SVD(task, data_info, loss_type).fit(train_data, neg_sampling)
    else:
        model = SVD(
            task=task,
            data_info=data_info,
            loss_type=loss_type,
            embed_size=16,
            norm_embed=norm_embed,
            n_epochs=2,
            lr=1e-4,
            reg=reg,
            batch_size=256,
            sampler=sampler,
            num_neg=num_neg,
            tf_sess_config=None,
        )
        model.fit(
            train_data,
            neg_sampling,
            verbose=2,
            shuffle=True,
            eval_data=eval_data,
            metrics=get_metrics(task),
            num_workers=num_workers,
        )
        ptest_tf_variables(model)
        ptest_preds(model, task, pd_data, with_feats=False)
        ptest_recommends(model, data_info, pd_data, with_feats=False)

        # test save and load model manual
        loaded_model, loaded_data_info = save_load_model(SVD, model, data_info)
        ptest_preds(loaded_model, task, pd_data, with_feats=False)
        ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=False)
