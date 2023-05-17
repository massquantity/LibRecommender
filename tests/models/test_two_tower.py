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
    "task, loss_type, use_correction, temperature, sampler, neg_sampling",
    [
        ("rating", "softmax", True, 1.0, "random", True),
        ("ranking", "whatever", True, 1.0, "random", True),
        ("ranking", "softmax", True, 1.0, "random", None),
        ("ranking", "cross_entropy", True, 1.0, "random", False),
        ("ranking", "cross_entropy", True, 1.0, "random", True),
        ("ranking", "max_margin", True, 1.0, "unconsumed", True),
        ("ranking", "max_margin", True, 1.0, "unconsumed", False),
        ("ranking", "softmax", True, 2.0, "popular", True),
        ("ranking", "softmax", False, 0.0, "unconsumed", True),
        ("ranking", "softmax", True, -1, "random", False),
    ],
)
@pytest.mark.parametrize(
    "norm_embed, lr_decay, reg, use_bn, dropout_rate, hidden_units, num_workers",
    [
        (True, False, None, True, None, 1, 0),
        (False, True, 0.001, False, 0.5, (32, 16), 2),
    ],
)
def test_two_tower(
    feat_data_small,
    task,
    loss_type,
    use_correction,
    temperature,
    sampler,
    neg_sampling,
    norm_embed,
    lr_decay,
    reg,
    use_bn,
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

    if task == "rating" or loss_type == "whatever":
        with pytest.raises(ValueError):
            _ = TwoTower(task, data_info, loss_type)
    elif neg_sampling is None:
        with pytest.raises(AssertionError):
            TwoTower(task, data_info).fit(train_data, neg_sampling)
    elif loss_type == "max_margin" and not neg_sampling:
        with pytest.raises(ValueError):
            TwoTower(task, data_info, loss_type).fit(train_data, neg_sampling)
    else:
        model = TwoTower(
            task=task,
            data_info=data_info,
            loss_type=loss_type,
            embed_size=4,
            norm_embed=norm_embed,
            n_epochs=1,
            lr=1e-4,
            lr_decay=lr_decay,
            reg=reg,
            batch_size=100,
            sampler=sampler,
            use_bn=use_bn,
            dropout_rate=dropout_rate,
            hidden_units=hidden_units,
            use_correction=use_correction,
            temperature=temperature,
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
