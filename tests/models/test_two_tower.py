import sys

import pytest
import tensorflow as tf
from numpy.testing import assert_array_equal

from libreco.algorithms import TwoTower
from tests.models.utils_tf import ptest_tf_variables
from tests.utils_data import set_ranking_labels
from tests.utils_metrics import get_metrics
from tests.utils_pred import ptest_preds
from tests.utils_reco import ptest_dyn_recommends, ptest_recommends
from tests.utils_save_load import save_load_model


@pytest.mark.parametrize(
    "task, loss_type, use_correction, temperature, ssl_pattern, sampler, neg_sampling",
    [
        ("rating", "softmax", True, 1.0, None, "random", True),
        ("ranking", "whatever", True, 1.0, None, "random", True),
        ("ranking", "softmax", True, 1.0, None, "random", None),
        ("ranking", "cross_entropy", True, 1.0, None, "random", False),
        ("ranking", "cross_entropy", True, 1.0, None, "random", True),
        ("ranking", "max_margin", True, 1.0, None, "unconsumed", True),
        ("ranking", "max_margin", True, 1.0, None, "unconsumed", False),
        ("ranking", "cross_entropy", True, 1.0, "rfm", "random", False),
        ("ranking", "softmax", True, 2.0, "whatever", "popular", True),
        ("ranking", "softmax", True, 0.1, "cfm", "random", True),
        ("ranking", "softmax", True, 2.0, "rfm-complementary", "popular", True),
        ("ranking", "softmax", False, 0.0, "rfm", "unconsumed", True),
        ("ranking", "softmax", True, -1, None, "random", False),
    ],
)
@pytest.mark.parametrize(
    "norm_embed, lr_decay, reg, use_bn, dropout_rate, hidden_units, remove_accidental_hits, num_workers",
    [
        (True, False, None, True, None, 1, True, 0),
        (False, True, 0.001, False, 0.5, (32, 16), False, 2),
    ],
)
def test_two_tower(
    feat_data_small,
    task,
    loss_type,
    use_correction,
    temperature,
    ssl_pattern,
    sampler,
    neg_sampling,
    norm_embed,
    lr_decay,
    reg,
    use_bn,
    dropout_rate,
    hidden_units,
    remove_accidental_hits,
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
    elif ssl_pattern == "whatever" or (ssl_pattern and loss_type != "softmax"):
        with pytest.raises(ValueError):
            _ = TwoTower(task, data_info, loss_type, ssl_pattern=ssl_pattern)
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
            remove_accidental_hits=remove_accidental_hits,
            ssl_pattern=ssl_pattern,
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
        ptest_recommends(model, data_info, pd_data, with_feats=True)
        dyn_rec = ptest_dyn_recommends(model, pd_data)

        model._assign_user_oov("user_embeds_var", "embedding")
        with pytest.raises(ValueError):
            model._assign_user_oov("user_embeds_fake", "embedding")

        # test save and load model
        loaded_model, loaded_data_info = save_load_model(TwoTower, model, data_info)
        ptest_preds(loaded_model, task, pd_data, with_feats=False)
        ptest_recommends(loaded_model, loaded_data_info, pd_data, with_feats=True)
        loaded_dyn_rec = ptest_dyn_recommends(loaded_model, pd_data)
        assert_array_equal(dyn_rec, loaded_dyn_rec)
