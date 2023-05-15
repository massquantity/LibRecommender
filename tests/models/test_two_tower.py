import sys

import pytest
import tensorflow as tf

from libreco.algorithms import TwoTower
from tests.utils_data import set_ranking_labels
from tests.utils_metrics import get_metrics
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_recommends
from tests.utils_save_load import save_load_model


@pytest.mark.parametrize(
    "task, neg_sampling",
    [
        ("rating", None),
        ("ranking", False),
        ("ranking", True),
    ],
)
@pytest.mark.parametrize(
    "norm_embed, lr_decay, reg, dropout_rate, hidden_units, num_workers",
    [
        (True, False, None, None, 1, 0),
        (False, True, 0.001, 0.5, (32, 16), 2),
    ],
)
def test_two_tower(
    feat_data_small,
    task,
    neg_sampling,
    norm_embed,
    lr_decay,
    reg,
    dropout_rate,
    hidden_units,
    num_workers,
):
    if not sys.platform.startswith("linux") and num_workers > 0:
        pytest.skip(
            "Windows and macOS use `spawn` in multiprocessing, which does not work well in pytest"
        )
    tf.compat.v1.reset_default_graph()
    pd_data, train_data, eval_data, data_info = feat_data_small
    if neg_sampling is False:
        set_ranking_labels(train_data)
        set_ranking_labels(eval_data)

    if task == "rating":
        with pytest.raises(AssertionError):
            _ = TwoTower(task, data_info)
    else:
        model = TwoTower(
            task=task,
            data_info=data_info,
            embed_size=4,
            norm_embed=norm_embed,
            n_epochs=1,
            lr=1e-4,
            lr_decay=lr_decay,
            reg=reg,
            batch_size=100,
            dropout_rate=dropout_rate,
            hidden_units=hidden_units,
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
        ptest_preds(model, task, pd_data, with_feats=False)
        ptest_recommends(model, data_info, pd_data, with_feats=False)

        # test save and load model
        loaded_model, loaded_data_info = save_load_model(TwoTower, model, data_info)
        ptest_preds(loaded_model, task, pd_data, with_feats=False)
        ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=False)
